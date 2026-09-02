#!/usr/bin/env python3
"""Construit puis lance le pilote Colab du micro-assistant 17M."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.1/datasets/assistant_curriculum_v1_17m"
)
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.1/checkpoints_17m_assistant_v1"
)
EXPECTED_PARENT_SHA = "3dbd076f7df86fedad4f4b47a657f042dd156de05d8b37d9c46319811fde7072"
EXPECTED_DATA = {
    "train.jsonl": "55627bd6791c9517825ee2e3483f4d43dfdee1629b9b3c31dd1a4d11e531f5aa",
    "validation.jsonl": "931e92b923703af677e66b3972092f20bf0ce8c93c57150603e29d8a286508bc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_path(environment_name: str) -> Path:
    raw = os.environ.get(environment_name)
    if not raw:
        raise RuntimeError(
            f"définis {environment_name} avec le chemin exact dans Google Drive"
        )
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    import torch

    print("1/5 — GPU et chemins", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    parent = required_path("IVOIRESLM_ASSISTANT_PARENT")
    data_dir = required_path("IVOIRESLM_BPE_DIR")
    print("GPU :", torch.cuda.get_device_name(0), flush=True)
    print("Parent :", parent, flush=True)
    print("BPE :", data_dir, flush=True)

    print("\n2/5 — Intégrité du parent CPT", flush=True)
    parent_sha = sha256(parent)
    print("SHA256 :", parent_sha, flush=True)
    if parent_sha != EXPECTED_PARENT_SHA:
        raise RuntimeError("ce fichier n'est pas le meilleur checkpoint CPT v1.0 attendu")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload or not (payload.get("config") or payload.get("parent_config")):
        raise RuntimeError("schéma du checkpoint parent invalide")

    print("\n3/5 — Curriculum contrôlé", flush=True)
    builder = PROJECT / "scripts/data/build_assistant_curriculum_v1.py"
    if not DATASET_DIR.exists():
        subprocess.run(
            [sys.executable, "-u", str(builder), "--output-dir", str(DATASET_DIR)],
            check=True,
        )
    for name, expected in EXPECTED_DATA.items():
        actual = sha256(DATASET_DIR / name)
        print(name, actual, "✅" if actual == expected else "❌", flush=True)
        if actual != expected:
            raise RuntimeError(f"dataset non conforme : {name}")
    if (DATASET_DIR / "test.jsonl").exists():
        raise RuntimeError("split test inattendu")

    print("\n4/5 — Nouvelle branche expérimentale", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected = [OUTPUT_DIR / name for name in ("best.pt", "latest.pt", "progress.json")]
    if any(path.exists() for path in protected):
        raise FileExistsError(
            "un pilote existe déjà ; utilise sa reprise au lieu de l'écraser"
        )
    print("Sortie :", OUTPUT_DIR, flush=True)

    print("\n5/5 — Pilote assistant 0 → 250", flush=True)
    trainer = PROJECT / "scripts/training/finetune_assistant_v1_17m.py"
    model_script = PROJECT / "scripts/training/train_transformer_ci_v04_17m.py"
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(trainer),
            "--dataset-dir",
            str(DATASET_DIR),
            "--data-dir",
            str(data_dir),
            "--base-checkpoint",
            str(parent),
            "--model-script",
            str(model_script),
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
    print("\nPilote 17M assistant étape 250 sauvegardé et reproductible ✅")
    print("Test toujours scellé ✅")


if __name__ == "__main__":
    main()
