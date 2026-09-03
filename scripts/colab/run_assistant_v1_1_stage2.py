#!/usr/bin/env python3
"""Construit et lance le stage 2 assistant depuis le checkpoint qualifié à 500."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.1/datasets/assistant_curriculum_v1_1_stage2"
)
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.1/checkpoints_17m_assistant_v1_1_stage2"
)
EXPECTED_PARENT_SHA = "fb5d53b7940037caee15b1a38b1b8ee1c057f418b3cbdbd66292930b3873ec6a"
EXPECTED_DATA = {
    "train.jsonl": "f42af03d1a0df1773f491f0db0d6b5bbad869326c93c63ca84e534303c4d886a",
    "validation.jsonl": "3b175e5a2fb99f1a431d8ed0116840a9bf464612121126096092e1b7c3ca3b4d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_path(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"définis {variable} avec le chemin exact dans Drive")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    import torch

    print("1/5 — GPU et chemins", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    parent = required_path("IVOIRESLM_ASSISTANT_STAGE2_PARENT")
    data_dir = required_path("IVOIRESLM_BPE_DIR")
    print("GPU :", torch.cuda.get_device_name(0), flush=True)
    print("Parent stage 1 :", parent, flush=True)
    print("BPE :", data_dir, flush=True)

    print("\n2/5 — Intégrité du checkpoint 500", flush=True)
    actual_parent_sha = sha256(parent)
    print("SHA256 :", actual_parent_sha, flush=True)
    if actual_parent_sha != EXPECTED_PARENT_SHA:
        raise RuntimeError("checkpoint parent inattendu : utilise le best.pt du palier 500")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 500:
        raise RuntimeError("le checkpoint parent n'est pas à l'étape 500")

    print("\n3/5 — Curriculum stage 2", flush=True)
    if not DATASET_DIR.exists():
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(PROJECT / "scripts/data/build_assistant_curriculum_v1_1.py"),
                "--output-dir",
                str(DATASET_DIR),
            ],
            check=True,
        )
    for name, expected in EXPECTED_DATA.items():
        actual = sha256(DATASET_DIR / name)
        print(name, actual, "✅" if actual == expected else "❌", flush=True)
        if actual != expected:
            raise RuntimeError(f"dataset non conforme : {name}")
    if (DATASET_DIR / "test.jsonl").exists():
        raise RuntimeError("split test inattendu")

    print("\n4/5 — Nouvelle expérience protégée", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected = [OUTPUT_DIR / name for name in ("best.pt", "latest.pt", "progress.json")]
    if any(path.exists() for path in protected):
        raise FileExistsError("le stage 2 existe déjà ; ne l'écrase pas")
    print("Sortie :", OUTPUT_DIR, flush=True)

    print("\n5/5 — Stage 2 : étape locale 0 → 250", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(PROJECT / "scripts/training/finetune_assistant_v1_1_stage2_17m.py"),
            "--dataset-dir",
            str(DATASET_DIR),
            "--data-dir",
            str(data_dir),
            "--base-checkpoint",
            str(parent),
            "--model-script",
            str(PROJECT / "scripts/training/train_transformer_ci_v04_17m.py"),
            "--output-dir",
            str(OUTPUT_DIR),
            "--stop-step",
            "250",
        ],
        check=True,
    )
    progress = json.loads((OUTPUT_DIR / "progress.json").read_text(encoding="utf-8"))
    if progress["current_step"] != 250 or progress["sealed_test_opened"] is not False:
        raise RuntimeError("état final inattendu")
    if sha256(OUTPUT_DIR / "latest.pt") != progress["latest_checkpoint_sha256"]:
        raise RuntimeError("checkpoint latest corrompu")
    print("\nSTAGE 2 ASSISTANT 0 → 250 SAUVEGARDÉ ET REPRODUCTIBLE ✅")
    print("Test toujours scellé ✅")


if __name__ == "__main__":
    main()
