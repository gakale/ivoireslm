#!/usr/bin/env python3
"""Lance le premier palier stage 4 depuis le CPT v1.0, jamais depuis stage 3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path("/content/drive/MyDrive/IvoireSLM/v1.2/datasets/assistant_curriculum_v1_3_stage4")
OUTPUT_DIR = Path("/content/drive/MyDrive/IvoireSLM/v1.2/checkpoints_17m_assistant_v1_3_stage4")
EXPECTED_PARENT_SHA = "3dbd076f7df86fedad4f4b47a657f042dd156de05d8b37d9c46319811fde7072"
EXPECTED_FEEDBACK_SHA = "6f97b00a90bf5f5121cfb9cddb6c6237bdeeee788f6b8baca7c8da672d0e674f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"définis {variable}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    import torch

    print("1/6 — GPU et artefacts", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    parent = required("IVOIRESLM_STAGE4_PARENT")
    data_dir = required("IVOIRESLM_BPE_DIR")
    feedback = required("IVOIRESLM_HUMAN_FEEDBACK")
    if sha256(parent) != EXPECTED_PARENT_SHA:
        raise RuntimeError("le parent doit être le best.pt CPT v1.0 étape 1000")
    if sha256(feedback) != EXPECTED_FEEDBACK_SHA:
        raise RuntimeError("le benchmark humain a changé ; fige une nouvelle version avant de continuer")
    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    if int(parent_payload.get("step", -1)) != 1000:
        raise RuntimeError("étape du checkpoint parent incorrecte")
    print("GPU :", torch.cuda.get_device_name(0), flush=True)
    print("Parent CPT étape 1000 intact ✅", flush=True)
    print("Benchmark humain 66 questions intact ✅", flush=True)

    print("\n2/6 — Construction du curriculum hors benchmark", flush=True)
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        report = json.loads((DATASET_DIR / "report.json").read_text(encoding="utf-8"))
    else:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, "-u", str(PROJECT / "scripts/data/build_assistant_curriculum_v1_3.py"),
            "--output-dir", str(DATASET_DIR), "--held-out-feedback", str(feedback),
        ], check=True)
        report = json.loads((DATASET_DIR / "report.json").read_text(encoding="utf-8"))
    if report["audit"]["human_holdout_questions"] != 66:
        raise RuntimeError("le curriculum n’a pas été construit contre les 66 questions figées")
    for split in ("train", "validation"):
        if sha256(DATASET_DIR / f"{split}.jsonl") != report["splits"][split]["sha256"]:
            raise RuntimeError(f"empreinte invalide : {split}")
    if (DATASET_DIR / "test.jsonl").exists():
        raise RuntimeError("split test inattendu")
    print("Curriculum vérifié, sans contamination ✅", flush=True)

    print("\n3/6 — Protection de l’expérience", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any((OUTPUT_DIR / name).exists() for name in ("best.pt", "latest.pt", "progress.json")):
        raise FileExistsError("le stage 4 existe déjà ; ne pas l’écraser")

    print("\n4/6 — Pilote prudent 0 → 125", flush=True)
    subprocess.run([
        sys.executable, "-u", str(PROJECT / "scripts/training/finetune_assistant_v1_3_stage4_17m.py"),
        "--dataset-dir", str(DATASET_DIR), "--data-dir", str(data_dir),
        "--base-checkpoint", str(parent),
        "--model-script", str(PROJECT / "scripts/training/train_transformer_ci_v04_17m.py"),
        "--output-dir", str(OUTPUT_DIR), "--stop-step", "125",
    ], check=True)

    print("\n5/6 — Gardes anti-collapse", flush=True)
    progress = json.loads((OUTPUT_DIR / "progress.json").read_text(encoding="utf-8"))
    generation = progress["last_generation_validation"]
    language = progress["last_general_language_validation"]["loss_nats"]
    baseline_language = progress["baseline_validation"]["general_language"]["loss_nats"]
    language_delta = language - baseline_language
    improved = generation["quality_score"] > progress["baseline_validation"]["generation"]["quality_score"]
    collapse_guard = bool(generation.get("collapse_guard_passed"))
    language_guard = language_delta <= 0.01
    candidate = improved and collapse_guard and language_guard
    print("Qualité améliorée :", improved)
    print("Garde diversité :", collapse_guard)
    print("Évolution loss générale :", round(language_delta, 6))
    print("Garde langue :", language_guard)
    print("Candidat pour comparaison humaine :", candidate)

    print("\n6/6 — Intégrité", flush=True)
    if sha256(OUTPUT_DIR / "latest.pt") != progress["latest_checkpoint_sha256"]:
        raise RuntimeError("latest.pt corrompu")
    decision = {
        "stage": "assistant_v1_3_stage4_anti_collapse",
        "current_step": 125,
        "parent_sha256": EXPECTED_PARENT_SHA,
        "human_feedback_sha256": EXPECTED_FEEDBACK_SHA,
        "training_questions_from_human_benchmark": 0,
        "quality_improved": improved,
        "collapse_guard_passed": collapse_guard,
        "general_language_loss_delta": language_delta,
        "general_language_guard_passed": language_guard,
        "candidate_for_blind_human_comparison": candidate,
        "sealed_test_opened": False,
    }
    (OUTPUT_DIR / "PILOT_DECISION_STEP125.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PALIER STAGE 4 — 0 → 125 SAUVEGARDÉ ✅")
    print("Benchmark humain non entraîné ; test final toujours scellé ✅")


if __name__ == "__main__":
    main()
