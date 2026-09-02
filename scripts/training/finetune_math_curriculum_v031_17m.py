#!/usr/bin/env python3
"""Pilote du curriculum mathématique progressif v0.3.1 pour le modèle 17M."""
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
from finetune_instruction_sft_v03_17m import greedy_generate


@dataclass
class CurriculumConfig:
    model_id: str = "microivoire_transformer_v0.4_17m_math_v0.3.1_stage1"
    max_steps: int = 1_000
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 1e-5
    minimum_learning_rate: float = 1e-6
    warmup_steps: int = 100
    weight_decay: float = 0.05
    gradient_clip: float = 1.0
    raw_language_probability: float = 0.35
    evaluation_interval: int = 250
    checkpoint_interval: int = 250
    log_interval: int = 50
    generation_examples_per_family: int = 100
    maximum_general_language_loss_increase: float = 0.02
    seed: int = 20260902


TASK_WEIGHTS = {
    "addition_easy": 0.70,
    "multiplication_table": 0.30,
}
PASS_THRESHOLDS = {
    "addition_easy": 0.25,
    "multiplication_table": 0.15,
}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def learning_rate(step: int, config: CurriculumConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = min(
        1.0,
        (step - config.warmup_steps) / (config.max_steps - config.warmup_steps),
    )
    return config.minimum_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        config.learning_rate - config.minimum_learning_rate
    )


def fixed_rows(rows: list[dict], family: str, limit: int) -> list[dict]:
    selected = [row for row in rows if row["task_family"] == family]
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"math-v031-generation:{row['example_id']}".encode("utf-8")
        ).digest()
    )
    return selected[:limit]


@torch.inference_mode()
def evaluate_generation(
    model,
    rows: list[dict],
    tokenizer,
    eos_id: int,
    device,
    autocast_context,
    limit: int,
) -> dict:
    families, samples = {}, []
    for family in TASK_WEIGHTS:
        selected = fixed_rows(rows, family, limit)
        exact, similarity_sum = 0, 0.0
        for row in selected:
            target_length = len(
                tokenizer.encode(row["target"], add_special_tokens=False).ids
            )
            prediction = greedy_generate(
                model,
                tokenizer,
                row["prompt"],
                eos_id,
                device,
                autocast_context,
                min(32, target_length + 8),
            )
            expected_norm = normalized(row["target"])
            prediction_norm = normalized(prediction)
            is_exact = prediction_norm == expected_norm
            similarity = SequenceMatcher(
                None, prediction_norm, expected_norm
            ).ratio()
            exact += int(is_exact)
            similarity_sum += similarity
            if len([item for item in samples if item["task_family"] == family]) < 3:
                samples.append(
                    {
                        "task_family": family,
                        "example_id": row["example_id"],
                        "expected": row["target"].strip(),
                        "prediction": prediction.strip(),
                        "exact": is_exact,
                    }
                )
        families[family] = {
            "examples": len(selected),
            "correct": exact,
            "exact_rate": exact / len(selected),
            "mean_similarity": similarity_sum / len(selected),
        }
    score = sum(
        TASK_WEIGHTS[family] * values["exact_rate"]
        for family, values in families.items()
    )
    return {"quality_score": score, "families": families, "samples": samples}


def checkpoint_payload(
    model,
    transformer_config,
    config,
    step: int,
    best_quality: float,
    baseline: dict,
    dataset_provenance: dict,
    base_checkpoint_sha256: str,
    optimizer=None,
    scaler=None,
    rng=None,
) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(transformer_config),
        "curriculum_config": asdict(config),
        "step": step,
        "best_generation_quality": best_quality,
        "baseline_validation": baseline,
        "dataset_provenance": dataset_provenance,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "selection_metric": "weighted_exact_rate_with_language_guard",
        "sealed_test_opened": False,
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-step", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args, config = arguments(), CurriculumConfig()
    if not 1 <= args.stop_step <= config.max_steps:
        raise ValueError("stop-step invalide")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    report = json.loads((args.dataset_dir / "report.json").read_text(encoding="utf-8"))
    if report["dataset_id"] != "math_curriculum_v0.3.1_stage1":
        raise RuntimeError("dataset inattendu")
    if report["status"] != "stage1_train_validation_only_test_not_created":
        raise RuntimeError("statut du dataset inattendu")
    train_rows = validate_dataset(args.dataset_dir / "train.jsonl", report, "train")
    validation_rows = validate_dataset(
        args.dataset_dir / "validation.jsonl", report, "validation"
    )
    dataset_provenance = {
        "dataset_id": report["dataset_id"],
        "report_sha256": sha256(args.dataset_dir / "report.json"),
        "train_sha256": report["splits"]["train"]["sha256"],
        "validation_sha256": report["splits"]["validation"]["sha256"],
    }
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
    base_checkpoint_sha256 = sha256(args.base_checkpoint)
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
        if resume.get("sealed_test_opened") is not False:
            raise RuntimeError("checkpoint de reprise non conforme")
        if resume.get("dataset_provenance") != dataset_provenance:
            raise RuntimeError("dataset différent de celui du checkpoint de reprise")
        if resume.get("base_checkpoint_sha256") != base_checkpoint_sha256:
            raise RuntimeError("checkpoint parent différent de la reprise")
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
        print(f"Reprise exacte depuis l'étape {start_step:,} ✅", flush=True)
    if args.stop_step <= start_step:
        raise ValueError("stop-step doit être supérieur à l'étape reprise")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    latest_path = args.output_dir / "latest.pt"
    progress_path = args.output_dir / "progress.json"
    if not args.resume and any(
        path.exists() for path in (best_path, latest_path, progress_path)
    ):
        raise FileExistsError(
            "Une expérience existe déjà dans output-dir ; utilise --resume ou un nouveau dossier"
        )
    families, probabilities = zip(*TASK_WEIGHTS.items())
    started = time.monotonic()
    print(
        f"Base : étape {base['step']:,} | paramètres : "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}",
        flush=True,
    )
    print(
        f"Curriculum stage 1 train/validation : {len(train_rows):,}/{len(validation_rows):,}",
        flush=True,
    )

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
                model,
                transformer_config,
                config,
                0,
                best_quality,
                baseline,
                dataset_provenance,
                base_checkpoint_sha256,
            ),
            best_path,
        )
        print(
            f"Référence | exact pondéré {best_quality:.3%} | "
            f"langue {baseline_raw['loss_nats']:.4f}",
            flush=True,
        )

    last_teacher = last_raw = last_generation = None
    best_step = int(torch.load(best_path, map_location="cpu", weights_only=False)["step"])
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
                    raw_tokens,
                    config.batch_size,
                    transformer_config.block_size,
                    rng,
                    device,
                )
                mode = "raw_language"
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
                scaled_loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss non finie à l'étape {step}")
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach()))
            modes.append(mode)
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"gradient non fini à l'étape {step}")
        scaler.step(optimizer)
        scaler.update()

        if step % config.evaluation_interval == 0:
            last_teacher = evaluate(
                model,
                validation_rows,
                tokenizer,
                pad_id,
                eos_id,
                device,
                autocast_context,
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
            language_limit = baseline["general_language"]["loss_nats"] * (
                1 + config.maximum_general_language_loss_increase
            )
            language_guard_passed = last_raw["loss_nats"] <= language_limit
            print(
                f"étape {step:4d}/{config.max_steps} | train {np.mean(losses):.4f} | "
                f"exact {quality:.3%} | langue {last_raw['loss_nats']:.4f} | "
                f"garde {'OK' if language_guard_passed else 'NON'} | lr {lr:.2e} | "
                f"{time.monotonic()-started:.1f}s",
                flush=True,
            )
            for family, values in last_generation["families"].items():
                print(
                    f"  {family}: {values['correct']}/{values['examples']} "
                    f"({values['exact_rate']:.1%})",
                    flush=True,
                )
            if language_guard_passed and quality > best_quality:
                best_quality, best_step = quality, step
                torch.save(
                    checkpoint_payload(
                        model,
                        transformer_config,
                        config,
                        step,
                        best_quality,
                        baseline,
                        dataset_provenance,
                        base_checkpoint_sha256,
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
                    dataset_provenance,
                    base_checkpoint_sha256,
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
    pass_by_family = {
        family: last_generation["families"][family]["exact_rate"] >= threshold
        for family, threshold in PASS_THRESHOLDS.items()
    }
    language_limit = baseline["general_language"]["loss_nats"] * (
        1 + config.maximum_general_language_loss_increase
    )
    language_guard_passed = last_raw["loss_nats"] <= language_limit
    pilot_passed = all(pass_by_family.values()) and language_guard_passed
    progress = {
        "experiment_id": "math_curriculum_v0.3.1_stage1",
        "model_id": config.model_id,
        "base_checkpoint_step": int(base["step"]),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "dataset_provenance": dataset_provenance,
        "current_step": args.stop_step,
        "target_step": config.max_steps,
        "best_step": best_step,
        "best_generation_quality": best_quality,
        "baseline_validation": baseline,
        "last_teacher_forced_validation": last_teacher,
        "last_general_language_validation": last_raw,
        "last_generation_validation": last_generation,
        "pass_thresholds": PASS_THRESHOLDS,
        "pass_by_family": pass_by_family,
        "language_guard_passed": language_guard_passed,
        "pilot_passed": pilot_passed,
        "task_sampling_weights": TASK_WEIGHTS,
        "raw_language_probability": config.raw_language_probability,
        "latest_checkpoint_sha256": sha256(latest_path),
        "sealed_test_opened": False,
        "elapsed_seconds_this_run": time.monotonic() - started,
    }
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    print(
        "Palier stage 1 terminé ; "
        f"décision automatique : {'PASS' if pilot_passed else 'CONTINUER_OU_REVOIR'} ✅",
        flush=True,
    )
    print("Test scellé toujours fermé ✅", flush=True)


if __name__ == "__main__":
    main()
