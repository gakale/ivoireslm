#!/usr/bin/env python3
"""Lance le stage 3 assistant depuis le meilleur checkpoint du stage 2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path("/content/drive/MyDrive/IvoireSLM/v1.1/datasets/assistant_curriculum_v1_2_stage3")
OUTPUT_DIR = Path("/content/drive/MyDrive/IvoireSLM/v1.1/checkpoints_17m_assistant_v1_2_stage3")
EXPECTED_PARENT_SHA = "0c81357ac03a2d7ff104d47737d9f55e2c45e2654e3bd7d3bb3ca9da729213d7"
EXPECTED_DATA = {
    "train.jsonl": "f831160ccb23d169310e7c649c588aff3ff11b51c2b3f5b45a4777f016b6937f",
    "validation.jsonl": "4138c67ec2ef83aa561f6b08152fb63d093349c37186a2cb99382e7baf4ed8a1",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required(variable):
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"définis {variable}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    import torch

    print("1/5 — GPU et parent stage 2", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    parent = required("IVOIRESLM_ASSISTANT_STAGE3_PARENT")
    data_dir = required("IVOIRESLM_BPE_DIR")
    actual = sha256(parent)
    print("GPU :", torch.cuda.get_device_name(0), flush=True)
    print("SHA256 parent :", actual, flush=True)
    if actual != EXPECTED_PARENT_SHA:
        raise RuntimeError("utilise le best.pt exact du stage 2 à l'étape 500")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 500:
        raise RuntimeError("étape du parent incorrecte")

    print("\n2/5 — Construction de la validation neuve", flush=True)
    if not DATASET_DIR.exists():
        subprocess.run([sys.executable, "-u", str(PROJECT / "scripts/data/build_assistant_curriculum_v1_2.py"), "--output-dir", str(DATASET_DIR)], check=True)
    for name, expected in EXPECTED_DATA.items():
        actual = sha256(DATASET_DIR / name)
        print(name, actual, "✅" if actual == expected else "❌", flush=True)
        if actual != expected:
            raise RuntimeError(f"dataset incorrect : {name}")
    if (DATASET_DIR / "test.jsonl").exists():
        raise RuntimeError("split test inattendu")

    print("\n3/5 — Nouvelle expérience protégée", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any((OUTPUT_DIR / name).exists() for name in ("best.pt", "latest.pt", "progress.json")):
        raise FileExistsError("le stage 3 existe déjà ; ne pas l'écraser")

    print("\n4/5 — Entraînement stage 3 : 0 → 250", flush=True)
    subprocess.run([
        sys.executable, "-u", str(PROJECT / "scripts/training/finetune_assistant_v1_2_stage3_17m.py"),
        "--dataset-dir", str(DATASET_DIR), "--data-dir", str(data_dir),
        "--base-checkpoint", str(parent),
        "--model-script", str(PROJECT / "scripts/training/train_transformer_ci_v04_17m.py"),
        "--output-dir", str(OUTPUT_DIR), "--stop-step", "250",
    ], check=True)

    print("\n5/5 — Intégrité", flush=True)
    progress = json.loads((OUTPUT_DIR / "progress.json").read_text())
    if progress["current_step"] != 250 or progress["sealed_test_opened"] is not False:
        raise RuntimeError("état final inattendu")
    if sha256(OUTPUT_DIR / "latest.pt") != progress["latest_checkpoint_sha256"]:
        raise RuntimeError("checkpoint corrompu")
    print("STAGE 3 — 0 → 250 SAUVEGARDÉ ET REPRODUCTIBLE ✅")
    print("Test toujours scellé ✅")


if __name__ == "__main__":
    main()
