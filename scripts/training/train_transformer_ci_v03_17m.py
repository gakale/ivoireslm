#!/usr/bin/env python3
"""Entraîne MicroIvoire v0.3 dense (~17,13 M) par paliers reprenables."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from tokenizers import Tokenizer


@dataclass
class TrainingConfig:
    model_id: str = "microivoire_transformer_v0.3_17m"
    tokenizer_id: str = "ivoireslm_bpe_v0.3"
    corpus_id: str = "ivoireslm_corpus_v0.7.0"
    vocab_size: int = 8_192
    block_size: int = 512
    embedding_dim: int = 320
    attention_heads: int = 8
    transformer_layers: int = 12
    feed_forward_dim: int = 832
    dropout: float = 0.10
    batch_size: int = 16
    gradient_accumulation: int = 2
    max_steps: int = 20_800
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    evaluation_interval: int = 250
    checkpoint_interval: int = 500
    log_interval: int = 25
    seed: int = 20260829


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data(data_dir: Path, config: TrainingConfig) -> dict:
    report_path = data_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["tokenizer_id"] != config.tokenizer_id:
        raise ValueError("identifiant de tokenizer inattendu")
    if report["corpus_id"] != config.corpus_id:
        raise ValueError("identifiant de corpus inattendu")
    if report["vocab_size"] != config.vocab_size:
        raise ValueError("taille de vocabulaire inattendue")
    expected = {"tokenizer.json": report["tokenizer_sha256"]}
    for split in ("train", "validation", "test"):
        expected[f"{split}.uint16.bin"] = report["splits"][split][
            "token_sha256"
        ]
    for name, digest in expected.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != digest:
            raise ValueError(f"SHA256 incorrect pour {name}: {observed}")
    print(
        f"Données {config.tokenizer_id} / {config.corpus_id} et empreintes SHA256 validées ✅",
        flush=True,
    )
    return report


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, maximum_length: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("la dimension de tête RoPE doit être paire")
        inverse = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(maximum_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cosine", embedding.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sine", embedding.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-2]
        cosine = self.cosine[:, :, :length].to(dtype=x.dtype)
        sine = self.sine[:, :, :length].to(dtype=x.dtype)
        return x * cosine + self.rotate_half(x) * sine


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        if config.embedding_dim % config.attention_heads:
            raise ValueError("embedding_dim doit être divisible par attention_heads")
        self.heads = config.attention_heads
        self.head_dim = config.embedding_dim // config.attention_heads
        self.qkv = nn.Linear(
            config.embedding_dim, 3 * config.embedding_dim, bias=False
        )
        self.projection = nn.Linear(
            config.embedding_dim, config.embedding_dim, bias=False
        )
        self.rope = RotaryEmbedding(self.head_dim, config.block_size)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q = self.rope.apply_rotary(q)
        k = self.rope.apply_rotary(k)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.projection(y)


class SwiGLU(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.gate = nn.Linear(
            config.embedding_dim, config.feed_forward_dim, bias=False
        )
        self.up = nn.Linear(
            config.embedding_dim, config.feed_forward_dim, bias=False
        )
        self.down = nn.Linear(
            config.feed_forward_dim, config.embedding_dim, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.embedding_dim)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class MicroIvoireTransformer17M(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.transformer_layers)
        )
        self.final_norm = RMSNorm(config.embedding_dim)
        self.language_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )
        self.language_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.shape[1] > self.config.block_size:
            raise ValueError("séquence plus longue que block_size")
        x = self.dropout(self.token_embedding(tokens))
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
    starts = rng.integers(
        0, len(tokens) - config.block_size - 1, size=config.batch_size
    )
    offsets = np.arange(config.block_size)
    x = np.asarray(tokens[starts[:, None] + offsets], dtype=np.int64)
    y = np.asarray(tokens[starts[:, None] + offsets + 1], dtype=np.int64)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    if device.type == "cuda":
        x_tensor = x_tensor.pin_memory()
        y_tensor = y_tensor.pin_memory()
    return (
        x_tensor.to(device, non_blocking=True),
        y_tensor.to(device, non_blocking=True),
    )


@torch.inference_mode()
def evaluate_complete(
    model: MicroIvoireTransformer17M,
    tokens: np.memmap,
    config: TrainingConfig,
    device: torch.device,
    autocast_context,
) -> dict[str, float | int]:
    model.eval()
    usable = ((len(tokens) - 1) // config.block_size) * config.block_size
    total_loss = 0.0
    evaluated = 0
    for first in range(0, usable, config.batch_size * config.block_size):
        count = min(config.batch_size, (usable - first) // config.block_size)
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


def checkpoint_payload(
    model: MicroIvoireTransformer17M,
    config: TrainingConfig,
    step: int,
    best_validation_loss: float,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def upload(path: Path, gcs_output: str | None) -> None:
    if not gcs_output:
        return
    destination = f"{gcs_output.rstrip('/')}/{path.name}"
    subprocess.run(["gcloud", "storage", "cp", str(path), destination], check=True)
    print(f"Sauvegarde distante : {destination} ✅", flush=True)


@torch.inference_mode()
def generate(
    model: MicroIvoireTransformer17M,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    seed: int,
    new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
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
        if int(next_token.item()) == tokenizer.token_to_id("<EOS>"):
            break
    return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)


def parse_arguments(default_output: Path = Path("/content/checkpoints_v03")) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--gcs-output")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-step", type=int, default=5_200)
    parser.add_argument("--allow-cpu-smoke-test", action="store_true")
    return parser.parse_args()


def main(config: TrainingConfig | None = None, default_output: Path | None = None) -> None:
    config = config or TrainingConfig()
    args = parse_arguments(default_output or Path("/content/checkpoints_v03"))
    if not 1 <= args.stop_step <= config.max_steps:
        raise ValueError(f"stop-step doit être entre 1 et {config.max_steps}")
    if args.allow_cpu_smoke_test and args.stop_step > 5:
        raise ValueError("le smoke test CPU est limité à 5 étapes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_report = verify_data(args.data_dir, config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu_smoke_test:
        device = torch.device("cpu")
    else:
        raise RuntimeError("GPU CUDA absent : active un accélérateur T4 dans Colab")

    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        supports_bfloat16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
        autocast_context = lambda: torch.autocast("cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=not supports_bfloat16)
    else:
        autocast_context = nullcontext
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    train_tokens = load_tokens(args.data_dir, "train")
    validation_tokens = load_tokens(args.data_dir, "validation")
    test_tokens = load_tokens(args.data_dir, "test")
    tokenizer = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))

    model = MicroIvoireTransformer17M(config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = 17_129_280
    if parameters != expected_parameters:
        raise AssertionError(
            f"paramètres attendus {expected_parameters:,}, obtenus {parameters:,}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    start_step = 0
    best_validation_loss = math.inf
    rng = np.random.default_rng(config.seed)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = TrainingConfig(**checkpoint["config"])
        if saved_config != config:
            raise ValueError("la configuration du checkpoint diffère de l'expérience")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        print(f"Reprise exacte depuis l'étape {start_step:,} ✅", flush=True)
    if start_step >= args.stop_step:
        raise ValueError("le checkpoint a déjà atteint stop-step")

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Appareil : {device_name}")
    print(f"Paramètres exacts : {parameters:,}")
    print(
        "Tokens train/validation/test : "
        f"{len(train_tokens):,} / {len(validation_tokens):,} / {len(test_tokens):,}"
    )
    print(
        f"Palier demandé : {args.stop_step:,}/{config.max_steps:,}; "
        "le test reste fermé avant le dernier palier.\n",
        flush=True,
    )

    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    progress_path = args.output_dir / "progress.json"
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    last_validation = None

    for step in range(start_step + 1, args.stop_step + 1):
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
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss non finie à l'étape {step}")
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"gradient non fini à l'étape {step}")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        should_evaluate = step == 1 or step % config.evaluation_interval == 0
        if should_evaluate:
            last_validation = evaluate_complete(
                model, validation_tokens, config, device, autocast_context
            )
            print(
                f"étape {step:5d}/{config.max_steps} | "
                f"train {accumulated_loss / config.gradient_accumulation:.4f} | "
                f"validation {last_validation['loss_nats']:.4f} | "
                f"ppl {last_validation['perplexity']:.3f} | lr {lr:.2e} | "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
            if float(last_validation["loss_nats"]) < best_validation_loss:
                best_validation_loss = float(last_validation["loss_nats"])
                torch.save(
                    checkpoint_payload(
                        model, config, step, best_validation_loss, rng
                    ),
                    best_path,
                )
                upload(best_path, args.gcs_output)
        elif step % config.log_interval == 0:
            print(
                f"étape {step:5d}/{config.max_steps} | "
                f"train {accumulated_loss / config.gradient_accumulation:.4f} | "
                f"lr {lr:.2e}",
                flush=True,
            )

        if step % config.checkpoint_interval == 0 or step == args.stop_step:
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    step,
                    best_validation_loss,
                    rng,
                    optimizer,
                    scaler,
                ),
                latest_path,
            )
            upload(latest_path, args.gcs_output)

    tokens_seen = (
        args.stop_step
        * config.batch_size
        * config.gradient_accumulation
        * config.block_size
    )
    progress = {
        "model_id": config.model_id,
        "parameters": parameters,
        "config": asdict(config),
        "current_step": args.stop_step,
        "target_step": config.max_steps,
        "training_tokens_seen": tokens_seen,
        "unique_train_tokens": len(train_tokens),
        "estimated_train_passes": tokens_seen / len(train_tokens),
        "best_validation_loss": best_validation_loss,
        "last_validation": last_validation,
        "test_opened": False,
        "data_report_sha256": sha256_file(args.data_dir / "report.json"),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "elapsed_seconds_this_run": time.monotonic() - started,
    }

    if args.stop_step == config.max_steps:
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        validation = evaluate_complete(
            model, validation_tokens, config, device, autocast_context
        )
        print("Ouverture unique du test gelé après sélection du modèle…", flush=True)
        test = evaluate_complete(model, test_tokens, config, device, autocast_context)
        prompts = [
            "La Côte d’Ivoire est",
            "À Abidjan, une étudiante explique que",
            "Question : Pourquoi faut-il vérifier une information ?\nRéponse :",
        ]
        samples = [
            generate(model, tokenizer, prompt, device, config.seed + index)
            for index, prompt in enumerate(prompts)
        ]
        sample_path = args.output_dir / "sample.txt"
        sample_path.write_text("\n\n---\n\n".join(samples) + "\n", encoding="utf-8")
        progress.update(
            {
                "best_step": int(best_checkpoint["step"]),
                "validation": validation,
                "test": test,
                "test_opened": True,
                "best_checkpoint_sha256": sha256_file(best_path),
            }
        )
        upload(sample_path, args.gcs_output)

    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    upload(progress_path, args.gcs_output)
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    if args.stop_step < config.max_steps:
        print("\nPalier terminé ; test gelé toujours fermé ✅")
    else:
        print("\nEntraînement complet et test gelé terminés ✅")


if __name__ == "__main__":
    main()
