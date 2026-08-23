#!/usr/bin/env python3
"""Entraîne le Transformer caractère IvoireSLM v0.2 (~4,76 M paramètres).

Le script est autonome pour Google Colab : il lit les fichiers uint16 par
memory mapping, sélectionne le meilleur modèle sur validation et n'évalue le
test gelé qu'après la fin de l'entraînement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EXPECTED_FILES = {
    "train.uint16.bin": "536d1402c335910aaf536bb4d976fd9797701bccf6c8fd8672d6aac2d6cfe004",
    "validation.uint16.bin": "bafd4bebeecf28097700beed85dad91ae8e01c7928fb62d302315b9a6a2425b3",
    "test.uint16.bin": "8da17a3584b97dfbb8c8452ba5921a4fc1dfa265a53269e51c0bdca550bfc9b9",
    "tokenizer.json": "e3a6c91d7b4d766573346d363d486e60170f30081848b95e404d0a10afb97f95",
}


@dataclass
class TrainingConfig:
    model_id: str = "microivoire_transformer_v0.2_5m"
    vocab_size: int = 722
    block_size: int = 256
    embedding_dim: int = 192
    attention_heads: int = 6
    transformer_layers: int = 10
    dropout: float = 0.15
    batch_size: int = 32
    gradient_accumulation: int = 1
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 300
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    evaluation_interval: int = 250
    checkpoint_interval: int = 500
    log_interval: int = 50
    seed: int = 20260823


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data(data_dir: Path) -> None:
    for name, expected in EXPECTED_FILES.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Fichier manquant : {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"SHA256 incorrect pour {name}: {observed}")
    print("Données v0.2 et empreintes SHA256 validées ✅", flush=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        if config.embedding_dim % config.attention_heads:
            raise ValueError("embedding_dim doit être divisible par attention_heads")
        self.heads = config.attention_heads
        self.head_dim = config.embedding_dim // config.attention_heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=False)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.projection(y)


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForwardNetwork(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class MicroIvoireTransformer(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.transformer_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        length = tokens.shape[1]
        if length > self.config.block_size:
            raise ValueError("séquence plus longue que block_size")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.language_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss


def load_tokens(data_dir: Path, split: str) -> np.memmap:
    return np.memmap(data_dir / f"{split}.uint16.bin", mode="r", dtype=np.uint16)


def random_batch(
    tokens: np.memmap,
    config: TrainingConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, len(tokens) - config.block_size - 1, size=config.batch_size)
    offsets = np.arange(config.block_size)
    x = np.asarray(tokens[starts[:, None] + offsets], dtype=np.int64)
    y = np.asarray(tokens[starts[:, None] + offsets + 1], dtype=np.int64)
    return (
        torch.from_numpy(x).pin_memory().to(device, non_blocking=True),
        torch.from_numpy(y).pin_memory().to(device, non_blocking=True),
    )


@torch.inference_mode()
def evaluate_complete(
    model: MicroIvoireTransformer,
    tokens: np.memmap,
    config: TrainingConfig,
    device: torch.device,
    autocast_context,
) -> dict[str, float | int]:
    """Évalue des blocs disjoints; aucune sélection aléatoire n'est utilisée."""
    model.eval()
    usable = ((len(tokens) - 1) // config.block_size) * config.block_size
    total_loss = 0.0
    evaluated = 0
    sequences_per_batch = config.batch_size
    for first in range(0, usable, sequences_per_batch * config.block_size):
        count = min(sequences_per_batch, (usable - first) // config.block_size)
        starts = first + np.arange(count) * config.block_size
        offsets = np.arange(config.block_size)
        x = np.asarray(tokens[starts[:, None] + offsets], dtype=np.int64)
        y = np.asarray(tokens[starts[:, None] + offsets + 1], dtype=np.int64)
        x_tensor = torch.from_numpy(x).to(device)
        y_tensor = torch.from_numpy(y).to(device)
        with autocast_context():
            logits, _ = model(x_tensor)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y_tensor.reshape(-1),
                reduction="sum",
            )
        total_loss += float(loss.item())
        evaluated += y_tensor.numel()
    mean_loss = total_loss / evaluated
    return {
        "evaluated_tokens": evaluated,
        "loss_nats": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def learning_rate(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = min(
        1.0,
        (step - config.warmup_steps) / (config.max_steps - config.warmup_steps),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + cosine * (
        config.learning_rate - config.minimum_learning_rate
    )


def save_checkpoint(
    path: Path,
    model: MicroIvoireTransformer,
    config: TrainingConfig,
    step: int,
    best_validation_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "step": step,
        "best_validation_loss": best_validation_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def upload(path: Path, gcs_output: str | None) -> None:
    if not gcs_output:
        return
    destination = f"{gcs_output.rstrip('/')}/{path.name}"
    subprocess.run(["gcloud", "storage", "cp", str(path), destination], check=True)
    print(f"Sauvegarde distante : {destination} ✅", flush=True)


def read_tokenizer(path: Path) -> tuple[list[str], dict[str, int]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    id_to_token = record["id_to_token"]
    return id_to_token, {token: index for index, token in enumerate(id_to_token)}


@torch.inference_mode()
def generate(
    model: MicroIvoireTransformer,
    prompt: str,
    id_to_token: list[str],
    token_to_id: dict[str, int],
    device: torch.device,
    seed: int,
    new_tokens: int = 800,
    temperature: float = 0.7,
    top_k: int = 30,
) -> str:
    unknown = token_to_id["<UNK>"]
    ids = [token_to_id.get(character, unknown) for character in prompt]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for _ in range(new_tokens):
        logits, _ = model(x[:, -model.config.block_size :])
        logits = logits[:, -1, :] / temperature
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits[logits < values[:, [-1]]] = -float("inf")
        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1, generator=generator)
        x = torch.cat((x, next_token), dim=1)
    special = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
    return "".join(
        id_to_token[token_id]
        for token_id in x[0].tolist()
        if id_to_token[token_id] not in special
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/checkpoints_v02"))
    parser.add_argument("--gcs-output")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = TrainingConfig(max_steps=args.max_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_data(args.data_dir)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un accélérateur T4 dans Colab")
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    supports_bfloat16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
    autocast_context = lambda: torch.autocast("cuda", dtype=amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=not supports_bfloat16)

    train_tokens = load_tokens(args.data_dir, "train")
    validation_tokens = load_tokens(args.data_dir, "validation")
    test_tokens = load_tokens(args.data_dir, "test")
    id_to_token, token_to_id = read_tokenizer(args.data_dir / "tokenizer.json")
    if len(id_to_token) != config.vocab_size:
        raise ValueError(f"vocabulaire attendu {config.vocab_size}, obtenu {len(id_to_token)}")

    model = MicroIvoireTransformer(config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 4_758_144:
        raise AssertionError(f"nombre de paramètres inattendu : {parameters:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    start_step = 0
    best_validation_loss = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        print(f"Reprise depuis l'étape {start_step:,} ✅", flush=True)

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"Paramètres : {parameters:,}")
    print(f"Tokens train/validation/test : {len(train_tokens):,} / {len(validation_tokens):,} / {len(test_tokens):,}")
    print("Le test reste fermé jusqu'à la fin de l'entraînement.\n", flush=True)

    rng = np.random.default_rng(config.seed + start_step)
    started = time.monotonic()
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, config.max_steps + 1):
        model.train()
        lr = learning_rate(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation):
            x, y = random_batch(train_tokens, config, rng, device)
            with autocast_context():
                _, loss = model(x, y)
                scaled_loss = loss / config.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        should_evaluate = step == 1 or step % config.evaluation_interval == 0
        if should_evaluate:
            validation = evaluate_complete(
                model, validation_tokens, config, device, autocast_context
            )
            print(
                f"étape {step:5d}/{config.max_steps} | "
                f"train {accumulated_loss / config.gradient_accumulation:.4f} | "
                f"validation {validation['loss_nats']:.4f} | "
                f"ppl {validation['perplexity']:.3f} | lr {lr:.2e} | "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
            if float(validation["loss_nats"]) < best_validation_loss:
                best_validation_loss = float(validation["loss_nats"])
                save_checkpoint(
                    best_path, model, config, step, best_validation_loss
                )
                upload(best_path, args.gcs_output)

        if step % config.checkpoint_interval == 0 or step == config.max_steps:
            save_checkpoint(
                latest_path,
                model,
                config,
                step,
                best_validation_loss,
                optimizer,
                scaler,
            )
            upload(latest_path, args.gcs_output)
        elif step % config.log_interval == 0 and not should_evaluate:
            print(
                f"étape {step:5d}/{config.max_steps} | "
                f"train {accumulated_loss / config.gradient_accumulation:.4f} | "
                f"lr {lr:.2e}",
                flush=True,
            )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    validation = evaluate_complete(model, validation_tokens, config, device, autocast_context)
    print("Meilleur checkpoint restauré. Ouverture unique du test gelé…", flush=True)
    test = evaluate_complete(model, test_tokens, config, device, autocast_context)

    prompts = [
        "La Côte d’Ivoire ",
        "Problème : Résoudre l’équation 3x + 5 = 20.\nMéthode : ",
        "Dans une conversation à Abidjan, ",
    ]
    samples = [
        generate(model, prompt, id_to_token, token_to_id, device, config.seed + index)
        for index, prompt in enumerate(prompts)
    ]
    sample_path = args.output_dir / "sample.txt"
    sample_path.write_text("\n\n---\n\n".join(samples) + "\n", encoding="utf-8")
    report = {
        "model_id": config.model_id,
        "model_type": "decoder_only_character_transformer",
        "parameters": parameters,
        "config": asdict(config),
        "best_step": int(best_checkpoint["step"]),
        "train_only_fit": True,
        "checkpoint_selection_split": "validation",
        "test_opened_once_after_selection": True,
        "validation": validation,
        "test": test,
        "best_checkpoint_sha256": sha256_file(best_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    upload(report_path, args.gcs_output)
    upload(sample_path, args.gcs_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nEntraînement, test gelé et sauvegarde terminés ✅")


if __name__ == "__main__":
    main()
