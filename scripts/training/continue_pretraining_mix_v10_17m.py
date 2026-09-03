#!/usr/bin/env python3
"""Continuation prudente du 17M sur le mélange v1.0, avec gardes par domaine."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


MIXTURE_ID = "ivoireslm_pretraining_mix_v1.0.0_pilot"
MODEL_ID = "microivoire_transformer_v1.0_17m_cpt_pilot"
WEIGHTS = {
    "base_v09": 0.60,
    "wikipedia_fr": 0.15,
    "wikipedia_en": 0.05,
    "mathematics_reasoning": 0.10,
    "code_agents": 0.05,
    "cybersecurity_defensive": 0.05,
}
MAX_STEPS = 1000
LEARNING_RATE = 2e-5
MINIMUM_LEARNING_RATE = 2e-6
WARMUP_STEPS = 50
EVALUATION_INTERVAL = 250
CHECKPOINT_INTERVAL = 250
SEED = 20260902
MAXIMUM_BASE_VALIDATION_INCREASE = 0.03


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location("ivoireslm_model_v04", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def learning_rate(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = min(1.0, (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS))
    return MINIMUM_LEARNING_RATE + 0.5 * (LEARNING_RATE - MINIMUM_LEARNING_RATE) * (
        1.0 + math.cos(math.pi * progress)
    )


def evaluate_all(model, config, module, validation, device, autocast_context) -> dict:
    metrics = {}
    for name, tokens in validation.items():
        metrics[name] = module.evaluate_complete(model, tokens, config, device, autocast_context)
        print(
            f"validation/{name}: loss {metrics[name]['loss_nats']:.4f} | "
            f"ppl {metrics[name]['perplexity']:.3f} | tokens {metrics[name]['evaluated_tokens']:,}",
            flush=True,
        )
    weighted_loss = sum(WEIGHTS[name] * metrics[name]["loss_nats"] for name in WEIGHTS)
    return {"domains": metrics, "weighted_loss": weighted_loss, "weighted_perplexity": math.exp(weighted_loss)}


def payload(model, parent_config, step, best_score, baseline, latest_validation, rng, optimizer=None, scaler=None):
    value = {
        "model_state_dict": model.state_dict(),
        "parent_config": parent_config,
        "cpt_config": {
            "model_id": MODEL_ID, "mixture_id": MIXTURE_ID, "weights": WEIGHTS,
            "max_steps": MAX_STEPS, "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MINIMUM_LEARNING_RATE, "warmup_steps": WARMUP_STEPS,
            "seed": SEED, "maximum_base_validation_increase": MAXIMUM_BASE_VALIDATION_INCREASE,
        },
        "step": step, "best_weighted_loss": best_score, "baseline": baseline,
        "last_validation": latest_validation, "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(), "test_opened": False,
    }
    if torch.cuda.is_available():
        value["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        value["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        value["scaler_state_dict"] = scaler.state_dict()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--supplement-data-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-step", type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.stop_step <= MAX_STEPS:
        raise ValueError("stop-step invalide")
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-12:
        raise AssertionError("les poids ne totalisent pas 1")
    if (args.supplement_data_dir / "test.uint16.bin").exists():
        raise RuntimeError("un test inattendu existe dans le supplément")

    module = load_module(args.model_script)
    parent = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    parent_config = parent["config"]
    config = module.TrainingConfig(**parent_config)
    base_report = json.loads((args.base_data_dir / "report.json").read_text())
    supplement_report = json.loads((args.supplement_data_dir / "report.json").read_text())
    if base_report["tokenizer_sha256"] != supplement_report["tokenizer_sha256"]:
        raise ValueError("tokenizers base et supplément différents")
    for name, digest in {
        "tokenizer.json": base_report["tokenizer_sha256"],
        "train.uint16.bin": base_report["splits"]["train"]["token_sha256"],
        "validation.uint16.bin": base_report["splits"]["validation"]["token_sha256"],
    }.items():
        if sha256_file(args.base_data_dir / name) != digest:
            raise ValueError(f"SHA256 base invalide: {name}")
    for split, buckets in supplement_report["buckets"].items():
        for bucket, row in buckets.items():
            if sha256_file(args.supplement_data_dir / row["file"]) != row["sha256"]:
                raise ValueError(f"SHA256 supplément invalide: {split}/{bucket}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA requis")
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    autocast_context = lambda: torch.autocast("cuda", dtype=amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    train = {"base_v09": module.load_tokens(args.base_data_dir, "train")}
    validation = {"base_v09": module.load_tokens(args.base_data_dir, "validation")}
    for bucket in WEIGHTS:
        if bucket == "base_v09":
            continue
        train[bucket] = np.memmap(args.supplement_data_dir / f"train.{bucket}.uint16.bin", mode="r", dtype=np.uint16)
        validation[bucket] = np.memmap(args.supplement_data_dir / f"validation.{bucket}.uint16.bin", mode="r", dtype=np.uint16)

    model = module.MicroIvoireTransformer17M(config).to(device)
    model.load_state_dict(parent["model_state_dict"])
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != 17_129_280:
        raise AssertionError(parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.05)
    rng = np.random.default_rng(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = args.output_dir / "latest.pt", args.output_dir / "best.pt"

    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume["cpt_config"]["mixture_id"] != MIXTURE_ID:
            raise ValueError("expérience de reprise différente")
        model.load_state_dict(resume["model_state_dict"]); optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume.get("scaler_state_dict", {}))
        start_step = int(resume["step"]); best_score = float(resume["best_weighted_loss"])
        baseline = resume["baseline"]; latest_validation = resume["last_validation"]
        rng.bit_generator.state = resume["numpy_rng_state"]
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng_state_all"]])
        print(f"Reprise exacte depuis l'étape {start_step} ✅", flush=True)
    else:
        if any(args.output_dir.iterdir()):
            raise FileExistsError("le dossier de sortie doit être vide pour une nouvelle expérience")
        print("Évaluation de référence avant continuation…", flush=True)
        baseline = evaluate_all(model, config, module, validation, device, autocast_context)
        latest_validation = baseline
        best_score = float(baseline["weighted_loss"])
        start_step = 0
        torch.save(payload(model, parent_config, 0, best_score, baseline, baseline, rng), best_path)
    if start_step >= args.stop_step:
        raise ValueError("stop-step déjà atteint")

    names = list(WEIGHTS); probabilities = np.asarray([WEIGHTS[name] for name in names])
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for step in range(start_step + 1, args.stop_step + 1):
        model.train(); lr = learning_rate(step - 1)
        for group in optimizer.param_groups: group["lr"] = lr
        losses, used = [], Counter()
        for _ in range(config.gradient_accumulation):
            chosen = str(rng.choice(names, p=probabilities)); used[chosen] += 1
            x, y = module.random_batch(train[chosen], config, rng, device)
            with autocast_context():
                _, loss = model(x, y); scaled = loss / config.gradient_accumulation
            if not torch.isfinite(loss): raise FloatingPointError(step)
            scaler.scale(scaled).backward(); losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm): raise FloatingPointError("gradient non fini")
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"étape {step:4d}/{MAX_STEPS} | train {sum(losses)/len(losses):.4f} | lr {lr:.2e}", flush=True)
        if step % EVALUATION_INTERVAL == 0:
            latest_validation = evaluate_all(model, config, module, validation, device, autocast_context)
            base_increase = latest_validation["domains"]["base_v09"]["loss_nats"] - baseline["domains"]["base_v09"]["loss_nats"]
            guard = base_increase <= MAXIMUM_BASE_VALIDATION_INCREASE
            improved = latest_validation["weighted_loss"] < best_score and guard
            print(f"score pondéré {latest_validation['weighted_loss']:.4f} | garde base {guard} | amélioration {improved}", flush=True)
            if improved:
                best_score = float(latest_validation["weighted_loss"])
                torch.save(payload(model, parent_config, step, best_score, baseline, latest_validation, rng), best_path)
        if step % CHECKPOINT_INTERVAL == 0 or step == args.stop_step:
            torch.save(payload(model, parent_config, step, best_score, baseline, latest_validation, rng, optimizer, scaler), latest_path)

    progress = {
        "model_id": MODEL_ID, "mixture_id": MIXTURE_ID, "parameters": parameters,
        "step": args.stop_step, "best_weighted_loss": best_score, "baseline": baseline,
        "last_validation": latest_validation,
        "base_guard_passed": latest_validation["domains"]["base_v09"]["loss_nats"] - baseline["domains"]["base_v09"]["loss_nats"] <= MAXIMUM_BASE_VALIDATION_INCREASE,
        "test_opened": False, "latest_sha256": sha256_file(latest_path),
        "best_sha256": sha256_file(best_path), "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    print("\nPalier de continuation sauvegardé ✅")
    print("Test toujours scellé ✅")


if __name__ == "__main__":
    main()
