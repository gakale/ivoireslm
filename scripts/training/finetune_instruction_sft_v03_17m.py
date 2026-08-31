#!/usr/bin/env python3
"""SFT v0.3 du Transformer 17M, sélectionné par générations de validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from finetune_instruction_sft_v02_17m import (
    evaluate,
    evaluate_raw_language,
    load_module,
    raw_batch,
    sha256,
    supervised_batch,
    validate_dataset,
)


@dataclass
class SFTConfig:
    model_id: str = "microivoire_transformer_v0.4_17m_sft_v0.3"
    max_steps: int = 4_000
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 2e-5
    minimum_learning_rate: float = 2e-6
    warmup_steps: int = 100
    weight_decay: float = 0.05
    gradient_clip: float = 1.0
    raw_language_probability: float = 0.20
    evaluation_interval: int = 250
    checkpoint_interval: int = 250
    log_interval: int = 50
    generation_examples_per_family: int = 20
    seed: int = 20260901


TASK_WEIGHTS = {
    "assistant_core": 0.10,
    "grounded_wdi": 0.05,
    "math_exact": 0.30,
    "translation_dyu_fr": 0.275,
    "translation_fr_dyu": 0.275,
}
EXACT_FAMILIES = {"grounded_wdi", "math_exact"}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def learning_rate(step: int, config: SFTConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = min(1.0, (step - config.warmup_steps) / (config.max_steps - config.warmup_steps))
    return config.minimum_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        config.learning_rate - config.minimum_learning_rate
    )


def fixed_generation_rows(rows: list[dict], limit: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["task_family"]].append(row)
    selected = []
    for family, family_rows in sorted(groups.items()):
        ordered = sorted(
            family_rows,
            key=lambda row: hashlib.sha256(
                f"generation-v03:{row['example_id']}".encode()
            ).digest(),
        )
        selected.extend(ordered[:limit])
    return selected


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt: str, eos_id: int, device, autocast_context, limit: int) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    generated: list[int] = []
    model.eval()
    for _ in range(limit):
        context = (prompt_ids + generated)[-model.config.block_size :]
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        with autocast_context():
            logits, _ = model(inputs)
        next_id = int(torch.argmax(logits[0, -1]).item())
        if next_id == eos_id:
            break
        generated.append(next_id)
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_generation(
    model,
    rows,
    tokenizer,
    eos_id,
    device,
    autocast_context,
    examples_per_family,
):
    selected = fixed_generation_rows(rows, examples_per_family)
    totals = defaultdict(lambda: {"examples": 0, "exact": 0, "similarity_sum": 0.0})
    samples = []
    for row in selected:
        expected_tokens = len(tokenizer.encode(row["target"], add_special_tokens=False).ids)
        prediction = greedy_generate(
            model,
            tokenizer,
            row["prompt"],
            eos_id,
            device,
            autocast_context,
            min(128, expected_tokens + 20),
        )
        expected_norm, prediction_norm = normalized(row["target"]), normalized(prediction)
        exact = prediction_norm == expected_norm
        similarity = SequenceMatcher(None, prediction_norm, expected_norm).ratio()
        family = row["task_family"]
        totals[family]["examples"] += 1
        totals[family]["exact"] += int(exact)
        totals[family]["similarity_sum"] += similarity
        if len([sample for sample in samples if sample["task_family"] == family]) < 2:
            samples.append(
                {
                    "task_family": family,
                    "example_id": row["example_id"],
                    "expected": row["target"].strip(),
                    "prediction": prediction.strip(),
                    "exact": exact,
                    "similarity": similarity,
                }
            )
    families, quality = {}, 0.0
    for family, values in sorted(totals.items()):
        exact_rate = values["exact"] / values["examples"]
        mean_similarity = values["similarity_sum"] / values["examples"]
        family_quality = exact_rate if family in EXACT_FAMILIES else mean_similarity
        families[family] = {
            "examples": values["examples"],
            "exact_rate": exact_rate,
            "mean_similarity": mean_similarity,
            "selection_quality": family_quality,
        }
        quality += TASK_WEIGHTS[family] * family_quality
    return {"quality_score": quality, "families": families, "samples": samples}


def checkpoint_payload(
    model,
    transformer_config,
    config,
    step,
    best_quality,
    baseline,
    optimizer=None,
    scaler=None,
    rng=None,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(transformer_config),
        "sft_config": asdict(config),
        "step": step,
        "best_generation_quality": best_quality,
        "baseline_validation": baseline,
        "selection_metric": "weighted_greedy_generation_quality",
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
    parser.add_argument("--stop-step", type=int, default=250)
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
    validation_rows = validate_dataset(
        args.dataset_dir / "validation.jsonl", report, "validation"
    )
    groups = defaultdict(list)
    for row in train_rows:
        groups[row["task_family"]].append(row)
    if set(groups) != set(TASK_WEIGHTS):
        raise RuntimeError(f"familles inattendues : {sorted(groups)}")
    if not math.isclose(sum(TASK_WEIGHTS.values()), 1.0):
        raise RuntimeError("poids de tâches invalides")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    autocast_context = lambda: torch.autocast("cuda", dtype=amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    start_step, best_quality, baseline = 0, -math.inf, None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
        scaler.load_state_dict(resume["scaler_state_dict"])
        rng.bit_generator.state = resume["numpy_rng_state"]
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])
        start_step = int(resume["step"])
        best_quality = float(resume["best_generation_quality"])
        baseline = resume["baseline_validation"]
        print(f"Reprise SFT v0.3 exacte depuis l’étape {start_step:,} ✅")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = args.output_dir / "best.pt", args.output_dir / "latest.pt"
    families, probabilities = zip(*TASK_WEIGHTS.items())
    started = time.monotonic()
    print(f"Base : étape {base['step']:,} | paramètres : {sum(p.numel() for p in model.parameters()):,}")
    print(f"SFT v0.3 train/validation : {len(train_rows):,}/{len(validation_rows):,}")

    if start_step == 0:
        baseline_teacher = evaluate(
            model, validation_rows, tokenizer, pad_id, eos_id, device, autocast_context
        )
        baseline_raw = evaluate_raw_language(
            model, raw_validation_tokens, device, autocast_context
        )
        baseline_generation = evaluate_generation(
            model,
            validation_rows,
            tokenizer,
            eos_id,
            device,
            autocast_context,
            config.generation_examples_per_family,
        )
        baseline = {
            "teacher_forced": baseline_teacher,
            "general_language": baseline_raw,
            "generation": baseline_generation,
        }
        best_quality = baseline_generation["quality_score"]
        torch.save(
            checkpoint_payload(
                model, transformer_config, config, 0, best_quality, baseline
            ),
            best_path,
        )
        print(
            f"Référence avant SFT | génération {best_quality:.4f} | "
            f"langue {baseline_raw['loss_nats']:.4f}",
            flush=True,
        )

    last_teacher = last_raw = last_generation = None
    for step in range(start_step + 1, args.stop_step + 1):
        model.train()
        lr = learning_rate(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses, modes = [], []
        for _ in range(config.gradient_accumulation):
            if rng.random() < config.raw_language_probability:
                x, y = raw_batch(
                    raw_tokens, config.batch_size, transformer_config.block_size, rng, device
                )
                mode = "raw"
            else:
                mode = str(rng.choice(families, p=probabilities))
                selected = groups[mode]
                indices = rng.integers(0, len(selected), size=config.batch_size).tolist()
                x, y, _ = supervised_batch(
                    selected,
                    indices,
                    tokenizer,
                    pad_id,
                    eos_id,
                    transformer_config.block_size,
                    device,
                )
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

        if step % config.evaluation_interval == 0:
            last_teacher = evaluate(
                model, validation_rows, tokenizer, pad_id, eos_id, device, autocast_context
            )
            last_raw = evaluate_raw_language(
                model, raw_validation_tokens, device, autocast_context
            )
            last_generation = evaluate_generation(
                model,
                validation_rows,
                tokenizer,
                eos_id,
                device,
                autocast_context,
                config.generation_examples_per_family,
            )
            quality = last_generation["quality_score"]
            print(
                f"étape {step:4d}/{config.max_steps} | train {np.mean(losses):.4f} | "
                f"génération {quality:.4f} | langue {last_raw['loss_nats']:.4f} | "
                f"lr {lr:.2e} | {time.monotonic()-started:.1f}s",
                flush=True,
            )
            if quality > best_quality:
                best_quality = quality
                torch.save(
                    checkpoint_payload(
                        model, transformer_config, config, step, best_quality, baseline
                    ),
                    best_path,
                )
        elif step % config.log_interval == 0:
            print(
                f"étape {step:4d}/{config.max_steps} | train {np.mean(losses):.4f} | "
                f"modes {','.join(modes)} | lr {lr:.2e}",
                flush=True,
            )
        if step % config.checkpoint_interval == 0 or step == args.stop_step:
            torch.save(
                checkpoint_payload(
                    model,
                    transformer_config,
                    config,
                    step,
                    best_quality,
                    baseline,
                    optimizer,
                    scaler,
                    rng,
                ),
                latest_path,
            )

    if last_generation is None:
        last_teacher = evaluate(
            model, validation_rows, tokenizer, pad_id, eos_id, device, autocast_context
        )
        last_raw = evaluate_raw_language(model, raw_validation_tokens, device, autocast_context)
        last_generation = evaluate_generation(
            model,
            validation_rows,
            tokenizer,
            eos_id,
            device,
            autocast_context,
            config.generation_examples_per_family,
        )

    progress = {
        "model_id": config.model_id,
        "base_checkpoint_step": int(base["step"]),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "current_step": args.stop_step,
        "target_step": config.max_steps,
        "best_generation_quality": best_quality,
        "baseline_validation": baseline,
        "last_teacher_forced_validation": last_teacher,
        "last_general_language_validation": last_raw,
        "last_generation_validation": last_generation,
        "task_sampling_weights": TASK_WEIGHTS,
        "raw_language_probability": config.raw_language_probability,
        "latest_checkpoint_sha256": sha256(latest_path),
        "sealed_test_opened": False,
        "elapsed_seconds_this_run": time.monotonic() - started,
    }
    (args.output_dir / "progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    print("Palier SFT v0.3 terminé ; tests finaux toujours scellés ✅")


if __name__ == "__main__":
    main()
