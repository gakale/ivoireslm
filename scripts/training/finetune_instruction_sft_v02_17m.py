#!/usr/bin/env python3
"""SFT équilibré et reproductible du Transformer IvoireSLM 17M."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


@dataclass
class SFTConfig:
    model_id: str = "microivoire_transformer_v0.4_17m_sft_v0.2"
    max_steps: int = 4_000
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 2e-5
    minimum_learning_rate: float = 2e-6
    warmup_steps: int = 100
    weight_decay: float = 0.05
    gradient_clip: float = 1.0
    raw_language_probability: float = 0.15
    evaluation_interval: int = 250
    checkpoint_interval: int = 500
    log_interval: int = 50
    seed: int = 20260831


TASK_WEIGHTS = {
    "assistant_core": 0.02,
    "grounded_wdi": 0.23,
    "math_verified": 0.30,
    "translation_dyu_fr": 0.225,
    "translation_fr_dyu": 0.225,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location("ivoireslm_transformer_v04_sft", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_dataset(path: Path, report: dict, split: str) -> list[dict]:
    expected = report["splits"][split]
    if sha256(path) != expected["sha256"]:
        raise RuntimeError(f"SHA256 invalide : {path}")
    rows = read_jsonl(path)
    if len(rows) != expected["examples"]:
        raise RuntimeError(f"nombre d'exemples invalide : {split}")
    return rows


def supervised_batch(rows, indices, tokenizer, pad_id, eos_id, block_size, device):
    examples, supervised_tokens = [], 0
    for index in indices:
        row = rows[index]
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
        target = tokenizer.encode(row["target"], add_special_tokens=False).ids + [eos_id]
        if len(target) > block_size:
            raise ValueError(f"cible trop longue : {row['example_id']}")
        prompt = prompt[-max(1, block_size + 1 - len(target)) :]
        sequence = prompt + target
        x, y = sequence[:-1], sequence[1:]
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


def raw_batch(tokens, batch_size, block_size, rng, device):
    starts = rng.integers(0, len(tokens) - block_size - 1, size=batch_size)
    offsets = np.arange(block_size)
    x = np.asarray(tokens[starts[:, None] + offsets], dtype=np.int64)
    y = np.asarray(tokens[starts[:, None] + offsets + 1], dtype=np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.inference_mode()
def evaluate_raw_language(model, tokens, device, autocast_context, *, windows=128):
    """Mesure l'oubli sur un échantillon fixe du corpus général de validation."""
    model.eval()
    block_size = model.config.block_size
    starts = np.linspace(0, len(tokens) - block_size - 1, num=windows, dtype=np.int64)
    offsets = np.arange(block_size)
    loss_sum, token_count = 0.0, 0
    for start in range(0, windows, 16):
        batch_starts = starts[start : start + 16]
        x = np.asarray(tokens[batch_starts[:, None] + offsets], dtype=np.int64)
        y = np.asarray(tokens[batch_starts[:, None] + offsets + 1], dtype=np.int64)
        x_tensor = torch.from_numpy(x).to(device)
        y_tensor = torch.from_numpy(y).to(device)
        with autocast_context():
            logits, _ = model(x_tensor)
        loss_sum += float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y_tensor.reshape(-1), reduction="sum"
            ).item()
        )
        token_count += y_tensor.numel()
    mean = loss_sum / token_count
    return {"tokens": token_count, "loss_nats": mean, "perplexity": math.exp(mean)}


@torch.inference_mode()
def evaluate(model, rows, tokenizer, pad_id, eos_id, device, autocast_context):
    model.eval()
    totals = defaultdict(lambda: [0.0, 0])
    for start in range(0, len(rows), 32):
        batch_rows = rows[start : start + 32]
        indices = list(range(len(batch_rows)))
        x, y, _ = supervised_batch(
            batch_rows, indices, tokenizer, pad_id, eos_id, model.config.block_size, device
        )
        with autocast_context():
            logits, _ = model(x)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape(y.shape)
        for row_index, row in enumerate(batch_rows):
            mask = y[row_index] != -100
            count = int(mask.sum().item())
            totals[row["task_family"]][0] += float(losses[row_index][mask].sum().item())
            totals[row["task_family"]][1] += count
    families, total_loss, total_tokens = {}, 0.0, 0
    for family, (loss_sum, tokens) in sorted(totals.items()):
        mean = loss_sum / tokens
        families[family] = {"tokens": tokens, "loss_nats": mean, "perplexity": math.exp(mean)}
        total_loss += loss_sum
        total_tokens += tokens
    mean = total_loss / total_tokens
    return {"tokens": total_tokens, "loss_nats": mean, "perplexity": math.exp(mean), "families": families}


def learning_rate(step: int, config: SFTConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = min(1.0, (step - config.warmup_steps) / (config.max_steps - config.warmup_steps))
    return config.minimum_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        config.learning_rate - config.minimum_learning_rate
    )


def weighted_validation_loss(validation: dict) -> float:
    """Aligne la sélection du checkpoint sur le mélange réellement entraîné."""
    return sum(
        TASK_WEIGHTS[family] * validation["families"][family]["loss_nats"]
        for family in TASK_WEIGHTS
    )


def checkpoint_payload(model, transformer_config, config, step, best_loss, optimizer=None, scaler=None, rng=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(transformer_config),
        "sft_config": asdict(config),
        "step": step,
        "best_validation_loss": best_loss,
    }
    if optimizer is not None:
        payload.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            }
        )
    return payload


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-step", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args, config = arguments(), SFTConfig()
    if not 1 <= args.stop_step <= config.max_steps:
        raise ValueError("stop-step invalide")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    report = json.loads((args.dataset_dir / "report.json").read_text(encoding="utf-8"))
    train_rows = validate_dataset(args.dataset_dir / "train.jsonl", report, "train")
    validation_rows = validate_dataset(args.dataset_dir / "validation.jsonl", report, "validation")
    groups = defaultdict(list)
    for row in train_rows:
        groups[row["task_family"]].append(row)
    if set(groups) != set(TASK_WEIGHTS):
        raise RuntimeError(f"familles inattendues : {sorted(groups)}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    autocast_context = lambda: torch.autocast("cuda", dtype=amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    module = load_module(args.model_script)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    transformer_config = module.TrainingConfig(**base["config"])
    transformer_config.model_id = config.model_id
    model = module.MicroIvoireTransformer17M(transformer_config)
    model.load_state_dict(base["model_state_dict"])
    model.to(device)
    tokenizer = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))
    pad_id, eos_id = tokenizer.token_to_id("<PAD>"), tokenizer.token_to_id("<EOS>")
    if pad_id is None or eos_id is None:
        raise RuntimeError("tokens spéciaux absents")
    raw_tokens = np.memmap(args.data_dir / "train.uint16.bin", mode="r", dtype=np.uint16)
    raw_validation_tokens = np.memmap(
        args.data_dir / "validation.uint16.bin", mode="r", dtype=np.uint16
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    rng, start_step, best_loss = np.random.default_rng(config.seed), 0, math.inf
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        rng.bit_generator.state = resume["numpy_rng_state"]
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])
        start_step, best_loss = int(resume["step"]), float(resume["best_validation_loss"])
        print(f"Reprise SFT exacte depuis l’étape {start_step:,} ✅")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = args.output_dir / "best.pt", args.output_dir / "latest.pt"
    families, probabilities = zip(*TASK_WEIGHTS.items())
    started = time.monotonic()
    print(f"Base : étape {base['step']:,} | paramètres : {sum(p.numel() for p in model.parameters()):,}")
    print(f"SFT train/validation : {len(train_rows):,}/{len(validation_rows):,} | arrêt : {args.stop_step:,}/{config.max_steps:,}")
    last_validation = None
    last_raw_validation = None
    if start_step == 0:
        last_validation = evaluate(
            model, validation_rows, tokenizer, pad_id, eos_id, device, autocast_context
        )
        last_raw_validation = evaluate_raw_language(
            model, raw_validation_tokens, device, autocast_context
        )
        baseline_score = weighted_validation_loss(last_validation)
        print(
            "Référence avant SFT | "
            f"score pondéré {baseline_score:.4f} | "
            f"langue générale {last_raw_validation['loss_nats']:.4f}",
            flush=True,
        )

    for step in range(start_step + 1, args.stop_step + 1):
        model.train()
        lr = learning_rate(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses, modes = [], []
        for _ in range(config.gradient_accumulation):
            if rng.random() < config.raw_language_probability:
                x, y = raw_batch(raw_tokens, config.batch_size, transformer_config.block_size, rng, device)
                mode = "raw"
            else:
                mode = str(rng.choice(families, p=probabilities))
                selected = groups[mode]
                indices = rng.integers(0, len(selected), size=config.batch_size).tolist()
                x, y, _ = supervised_batch(selected, indices, tokenizer, pad_id, eos_id, transformer_config.block_size, device)
            with autocast_context():
                _, loss = model(x, y)
                scaled = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss non finie à l’étape {step}")
            scaler.scale(scaled).backward()
            losses.append(float(loss.detach()))
            modes.append(mode)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"gradient non fini à l’étape {step}")
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % config.evaluation_interval == 0:
            last_validation = evaluate(
                model, validation_rows, tokenizer, pad_id, eos_id, device, autocast_context
            )
            last_raw_validation = evaluate_raw_language(
                model, raw_validation_tokens, device, autocast_context
            )
            validation_score = weighted_validation_loss(last_validation)
            print(f"étape {step:4d}/{config.max_steps} | train {np.mean(losses):.4f} | score {validation_score:.4f} | ppl {last_validation['perplexity']:.3f} | langue {last_raw_validation['loss_nats']:.4f} | lr {lr:.2e} | {time.monotonic()-started:.1f}s", flush=True)
            if validation_score < best_loss:
                best_loss = validation_score
                torch.save(checkpoint_payload(model, transformer_config, config, step, best_loss), best_path)
        elif step % config.log_interval == 0:
            print(f"étape {step:4d}/{config.max_steps} | train {np.mean(losses):.4f} | modes {','.join(modes)} | lr {lr:.2e}", flush=True)
        if step % config.checkpoint_interval == 0 or step == args.stop_step:
            torch.save(checkpoint_payload(model, transformer_config, config, step, best_loss, optimizer, scaler, rng), latest_path)

    if last_validation is None:
        last_validation = evaluate(
            model, validation_rows, tokenizer, pad_id, eos_id, device, autocast_context
        )
        last_raw_validation = evaluate_raw_language(
            model, raw_validation_tokens, device, autocast_context
        )

    progress = {
        "model_id": config.model_id,
        "base_checkpoint_step": int(base["step"]),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "current_step": args.stop_step,
        "target_step": config.max_steps,
        "best_validation_loss": best_loss,
        "selection_metric": "task_weighted_validation_loss",
        "last_supervised_validation": last_validation,
        "last_general_language_validation": last_raw_validation,
        "task_sampling_weights": TASK_WEIGHTS,
        "raw_language_probability": config.raw_language_probability,
        "latest_checkpoint_sha256": sha256(latest_path),
        "sealed_test_opened": False,
        "elapsed_seconds_this_run": time.monotonic() - started,
    }
    (args.output_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    print("Palier SFT terminé ; tests finaux toujours scellés ✅")


if __name__ == "__main__":
    main()
