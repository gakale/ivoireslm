#!/usr/bin/env python3
"""Restaure les dépendances et lance l'assistant 17M outillé dans Colab."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile


PROJECT = Path(__file__).resolve().parents[2]
DRIVE = Path("/content/drive/MyDrive/IvoireSLM")
BASE_ARCHIVE = DRIVE / "v0.4/artifacts/ivoireslm_colab_v0.4_17m_bundle.tar.gz"
EXPECTED_BASE_SHA = "fc7b93b8e796ff8657c70513b49a586cfceca23778aa6038e4f9ee3e4af2cba3"
BPE_ROOT = Path("/content/ivoireslm_v04_base_restore")
BPE_DIR = BPE_ROOT / "bpe_v0.4"
CHECKPOINT = DRIVE / "v1.4/checkpoints_17m_cpt_v111_pilot/best.pt"
EXPECTED_CHECKPOINT_SHA = "f97f0077cbbae78723284cf59f5f0193192a94b71e4d68bc47f86876dcb3d9e5"
FEEDBACK = DRIVE / "v1.7/human_feedback/tool_assistant_v1_feedback.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"chemin d’archive non sûr : {member.name}")
        try:
            bundle.extractall(destination, filter="data")
        except TypeError:
            bundle.extractall(destination)


def main():
    print("1/4 — Vérification du checkpoint CPT 500", flush=True)
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("checkpoint CPT 500 absent ou modifié")
    print("Checkpoint intact ✅", flush=True)

    print("\n2/4 — Tokenizer BPE", flush=True)
    tokenizer = BPE_DIR / "tokenizer.json"
    if not tokenizer.is_file():
        if not BASE_ARCHIVE.is_file() or sha256(BASE_ARCHIVE) != EXPECTED_BASE_SHA:
            raise RuntimeError("archive BPE absente ou modifiée")
        safe_extract(BASE_ARCHIVE, BPE_ROOT)
    for name in ("tokenizer.json", "train.uint16.bin", "validation.uint16.bin"):
        if not (BPE_DIR / name).is_file():
            raise FileNotFoundError(BPE_DIR / name)
    print("BPE disponible ✅", flush=True)

    print("\n3/4 — Interface et retours humains", flush=True)
    try:
        import gradio  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gradio>=5,<7"], check=True
        )
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    print("Retours :", FEEDBACK, flush=True)

    print("\n4/4 — Lancement", flush=True)
    print("Ouvre le lien public gradio.live qui va apparaître.", flush=True)
    subprocess.run([
        sys.executable, "-u",
        str(PROJECT / "scripts/inference/gradio_tool_assistant_v1.py"),
        "--checkpoint", str(CHECKPOINT),
        "--data-dir", str(BPE_DIR),
        "--model-script", str(PROJECT / "scripts/training/train_transformer_ci_v04_17m.py"),
        "--feedback-file", str(FEEDBACK),
        "--share",
        "--server-port", "7860",
    ], check=True)


if __name__ == "__main__":
    main()
