#!/usr/bin/env python3
"""Fine-tuning mathématique supervisé du Transformer IvoireSLM 5M."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class SFTConfig:
    model_id: str = "microivoire_transformer_v0.2_5m_math_sft_v0.1"
    max_steps: int = 3_000
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 2e-5
    minimum_learning_rate: float = 2e-6
    warmup_steps: int = 100
    weight_decay: float = 0.05
    gradient_clip: float = 1.0
    raw_language_probability: float = 0.20
    evaluation_interval: int = 250
    checkpoint_interval: int = 500
    log_interval: int = 50
    seed: int = 20260827


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location("ivoireslm_transformer_v02_sft", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def tokenizer_mapping(path: Path) -> tuple[dict[str, int], int, int]:
    record = json.loads(path.read_text(encoding="utf-8"))
    mapping = {token: index for index, token in enumerate(record["id_to_token"])}
    return mapping, mapping["<PAD>"], mapping["<EOS>"]


def encode(text: str, mapping: dict[str, int]) -> list[int]:
    unknown = mapping["<UNK>"]
    return [mapping.get(character, unknown) for character in text]


def supervised_batch(
    rows: list[dict],
    indices: list[int],
    mapping: dict[str, int],
    pad_id: int,
    eos_id: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    examples = []
    supervised_tokens = 0
    for index in indices:
        row = rows[index]
        prompt = encode(row["prompt"], mapping)
        target = encode(row["target"], mapping) + [eos_id]
        if len(target) > block_size:
            raise ValueError(f"cible trop longue : {row['example_id']} ({len(target)})")
        prompt = prompt[-max(1, block_size + 1 - len(target)) :]
        sequence = prompt + target
        x = sequence[:-1]
        y = sequence[1:]
        ignored = max(0, len(prompt) - 1)
        y[:ignored] = [-100] * ignored
        supervised_tokens += len(y) - ignored
        examples.append((x, y))

    maximum = max(len(x) for x, _ in examples)
    x_batch, y_batch = [], []
    for x, y in examples:
        padding = maximum - len(x)
        x_batch.append(x + [pad_id] * padding)
        y_batch.append(y + [-100] * padding)
    return (
        torch.tensor(x_batch, dtype=torch.long, device=device),
        torch.tensor(y_batch, dtype=torch.long, device=device),
        supervised_tokens,
    )


def raw_language_batch(
    tokens: np.memmap,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, len(tokens) - block_size - 1, size=batch_size)
    offsets = np.arange(block_size)
    x = np.asarray(tokens[starts[:, None] + offsets], dtype=np.int64)
    y = np.asarray(tokens[starts[:, None] + offsets + 1], dtype=np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.inference_mode()
def evaluate_supervised(
    model,
    rows: list[dict],
    mapping: dict[str, int],
    pad_id: int,
    eos_id: int,
    batch_size: int,
    device: torch.device,
    autocast_context,
) -> dict:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(rows), batch_size):
        indices = list(range(start, min(start + batch_size, len(rows))))
        x, y, count = supervised_batch(rows, indices, mapping, pad_id, eos_id, model.config.block_size, device)
        with autocast_context():
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum"
            )
        total_loss += float(loss.item())
        total_tokens += count
    mean = total_loss / total_tokens
    return {"supervised_tokens": total_tokens, "loss_nats": mean, "perplexity": math.exp(mean)}


def learning_rate(step: int, config: SFTConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = min(1.0, (step - config.warmup_steps) / (config.max_steps - config.warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + cosine * (config.learning_rate - config.minimum_learning_rate)


def save_checkpoint(path: Path, model, transformer_config, sft_config: SFTConfig, step: int, best_loss: float, optimizer=None, scaler=None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(transformer_config),
        "sft_config": asdict(sft_config),
        "step": step,
        "best_validation_loss": best_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def upload(path: Path, destination_root: str | None) -> None:
    if destination_root:
        subprocess.run(["gcloud", "storage", "cp", str(path), f"{destination_root.rstrip('/')}/{path.name}"], check=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/checkpoints_math_sft_v01"))
    parser.add_argument("--gcs-output")
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = SFTConfig(max_steps=args.max_steps)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda")
    supports_bfloat16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
    autocast_context = lambda: torch.autocast("cuda", dtype=amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=not supports_bfloat16)

    module = load_module(args.model_script)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    transformer_config = module.TrainingConfig(**base["config"])
    transformer_config.model_id = config.model_id
    model = module.MicroIvoireTransformer(transformer_config)
    model.load_state_dict(base["model_state_dict"])
    model.to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    mapping, pad_id, eos_id = tokenizer_mapping(args.data_dir / "tokenizer.json")
    train_rows = read_jsonl(args.train_jsonl)
    validation_rows = read_jsonl(args.validation_jsonl)
    raw_tokens = np.memmap(args.data_dir / "train.uint16.bin", mode="r", dtype=np.uint16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay)

    start_step, best_loss = 0, math.inf
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume.get("scaler_state_dict", {}))
        start_step = int(resume["step"])
        best_loss = float(resume["best_validation_loss"])

    print(f"Modèle de base : étape {base['step']:,}")
    print(f"Paramètres : {parameters:,}")
    print(f"SFT train/validation : {len(train_rows):,} / {len(validation_rows):,}")
    print(f"Rappel de langue générale : {config.raw_language_probability:.0%}")
    rng = np.random.default_rng(config.seed + start_step)
    started = time.monotonic()
    best_path = args.output_dir / "best.pt"
    latest_path = args.output_dir / "latest.pt"

    for step in range(start_step + 1, config.max_steps + 1):
        model.train()
        lr = learning_rate(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses, modes = [], []
        for _ in range(config.gradient_accumulation):
            raw_mode = rng.random() < config.raw_language_probability
            if raw_mode:
                x, y = raw_language_batch(raw_tokens, config.batch_size, transformer_config.block_size, rng, device)
                modes.append("raw")
            else:
                indices = rng.integers(0, len(train_rows), size=config.batch_size).tolist()
                x, y, _ = supervised_batch(train_rows, indices, mapping, pad_id, eos_id, transformer_config.block_size, device)
                modes.append("sft")
            with autocast_context():
                _, loss = model(x, y)
                scaled = loss / config.gradient_accumulation
            scaler.scale(scaled).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        evaluate = step == 1 or step % config.evaluation_interval == 0
        if evaluate:
            validation = evaluate_supervised(model, validation_rows, mapping, pad_id, eos_id, 32, device, autocast_context)
            print(f"étape {step:4d}/{config.max_steps} | train {sum(losses)/len(losses):.4f} | validation {validation['loss_nats']:.4f} | ppl {validation['perplexity']:.3f} | lr {lr:.2e} | {time.monotonic()-started:.1f}s", flush=True)
            if validation["loss_nats"] < best_loss:
                best_loss = validation["loss_nats"]
                save_checkpoint(best_path, model, transformer_config, config, step, best_loss)
                upload(best_path, args.gcs_output)
        elif step % config.log_interval == 0:
            print(f"étape {step:4d}/{config.max_steps} | train {sum(losses)/len(losses):.4f} | modes {','.join(modes)} | lr {lr:.2e}", flush=True)

        if step % config.checkpoint_interval == 0 or step == config.max_steps:
            save_checkpoint(latest_path, model, transformer_config, config, step, best_loss, optimizer, scaler)
            upload(latest_path, args.gcs_output)

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    validation = evaluate_supervised(model, validation_rows, mapping, pad_id, eos_id, 32, device, autocast_context)
    report = {
        "model_id": config.model_id,
        "parameters": parameters,
        "base_checkpoint_step": int(base["step"]),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "best_sft_step": int(best["step"]),
        "sft_config": asdict(config),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_sha256": sha256(args.train_jsonl),
        "validation_sha256": sha256(args.validation_jsonl),
        "validation": validation,
        "best_checkpoint_sha256": sha256(best_path),
        "elapsed_seconds": time.monotonic() - started,
        "final_math_benchmark_opened": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    upload(report_path, args.gcs_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Fine-tuning mathématique terminé; benchmark final toujours fermé ✅")


if __name__ == "__main__":
    main()
