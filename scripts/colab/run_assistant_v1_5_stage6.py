#!/usr/bin/env python3
"""Construit et lance le pilote Stage 6 depuis le CPT v1.1.1 étape 500."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import torch


PROJECT = Path(__file__).resolve().parents[2]
DRIVE = Path("/content/drive/MyDrive/IvoireSLM")
BASE_ARCHIVE = DRIVE / "v0.4/artifacts/ivoireslm_colab_v0.4_17m_bundle.tar.gz"
EXPECTED_BASE_ARCHIVE_SHA = "fc7b93b8e796ff8657c70513b49a586cfceca23778aa6038e4f9ee3e4af2cba3"
BPE_ROOT = Path("/content/ivoireslm_v04_base_restore")
BPE_DIR = BPE_ROOT / "bpe_v0.4"
PARENT = DRIVE / "v1.4/checkpoints_17m_cpt_v111_pilot/best.pt"
EXPECTED_PARENT_SHA = "f97f0077cbbae78723284cf59f5f0193192a94b71e4d68bc47f86876dcb3d9e5"
FEEDBACK = DRIVE / "v1.1/human_feedback/assistant_stage3_step250_feedback.jsonl"
EXPECTED_FEEDBACK_SHA = "6f97b00a90bf5f5121cfb9cddb6c6237bdeeee788f6b8baca7c8da672d0e674f"
DATASET_DIR = DRIVE / "v1.5/datasets/assistant_curriculum_v1_5_stage6"
OUTPUT_DIR = DRIVE / "v1.5/checkpoints_17m_assistant_stage6"


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


def verify_bpe() -> None:
    report_path = BPE_DIR / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {"tokenizer.json": report["tokenizer_sha256"]}
    for split in ("train", "validation"):
        expected[f"{split}.uint16.bin"] = report["splits"][split]["token_sha256"]
    for name, expected_sha in expected.items():
        path = BPE_DIR / name
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"BPE incomplet ou corrompu : {name}")


def main() -> None:
    print("1/7 — GPU et parent CPT", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    print("GPU :", torch.cuda.get_device_name(0), flush=True)
    if not PARENT.is_file() or sha256(PARENT) != EXPECTED_PARENT_SHA:
        raise RuntimeError("checkpoint CPT 500 absent ou modifié")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    if int(parent.get("step", -1)) != 500:
        raise RuntimeError("le parent doit être le CPT étape 500")
    if not FEEDBACK.is_file() or sha256(FEEDBACK) != EXPECTED_FEEDBACK_SHA:
        raise RuntimeError("retours humains absents ou modifiés")
    print("CPT 500 et retours humains intacts ✅", flush=True)

    print("\n2/7 — Tokenizer et corpus de garde", flush=True)
    try:
        verify_bpe()
    except (FileNotFoundError, RuntimeError):
        if not BASE_ARCHIVE.is_file() or sha256(BASE_ARCHIVE) != EXPECTED_BASE_ARCHIVE_SHA:
            raise RuntimeError("archive BPE officielle absente ou corrompue")
        safe_extract(BASE_ARCHIVE, BPE_ROOT)
        verify_bpe()
    print("BPE vérifié ✅", flush=True)

    print("\n3/7 — Curriculum supervisé v1.5", flush=True)
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        report = json.loads((DATASET_DIR / "report.json").read_text(encoding="utf-8"))
    else:
        DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, "-u",
            str(PROJECT / "scripts/data/build_assistant_curriculum_v1_5.py"),
            "--output-dir", str(DATASET_DIR),
            "--human-feedback", str(FEEDBACK),
        ], check=True)
        report = json.loads((DATASET_DIR / "report.json").read_text(encoding="utf-8"))
    if report["test_created"] is not False:
        raise RuntimeError("split test inattendu")
    for split in ("train", "validation"):
        if sha256(DATASET_DIR / f"{split}.jsonl") != report["splits"][split]["sha256"]:
            raise RuntimeError(f"dataset modifié : {split}")
    if sha256(DATASET_DIR / "human_holdout.jsonl") != report["human_holdout"]["sha256"]:
        raise RuntimeError("contrôle humain modifié")
    print("Curriculum et contrôle humain vérifiés ✅", flush=True)

    print("\n4/7 — Nouvelle branche protégée", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any((OUTPUT_DIR / name).exists() for name in ("best.pt", "latest.pt", "progress.json")):
        raise FileExistsError("le Stage 6 existe déjà ; refus de l’écraser")
    print("Sortie :", OUTPUT_DIR, flush=True)

    print("\n5/7 — Pilote Stage 6 : 0 → 125", flush=True)
    subprocess.run([
        sys.executable, "-u",
        str(PROJECT / "scripts/training/finetune_assistant_v1_5_stage6_17m.py"),
        "--dataset-dir", str(DATASET_DIR),
        "--data-dir", str(BPE_DIR),
        "--base-checkpoint", str(PARENT),
        "--model-script", str(PROJECT / "scripts/training/train_transformer_ci_v04_17m.py"),
        "--output-dir", str(OUTPUT_DIR),
        "--stop-step", "125",
    ], check=True)

    print("\n6/7 — Décision du pilote", flush=True)
    progress = json.loads((OUTPUT_DIR / "progress.json").read_text(encoding="utf-8"))
    baseline = progress["baseline_validation"]
    current = progress["last_generation_validation"]
    quality_delta = current["quality_score"] - baseline["generation"]["quality_score"]
    language_delta = (
        progress["last_general_language_validation"]["loss_nats"]
        - baseline["general_language"]["loss_nats"]
    )
    guards = {
        "quality_improved": quality_delta > 0,
        "direct_answer_guard_passed": bool(current["direct_answer_guard_passed"]),
        "collapse_guard_passed": bool(current["collapse_guard_passed"]),
        "general_language_guard_passed": language_delta <= 0.01,
    }
    candidate = all(guards.values())
    print("Évolution qualité :", round(quality_delta, 4))
    print("Échos/fragments :", f"{current['echo_fragment_rate'] * 100:.1f}%")
    print("Diversité :", f"{current['diversity']['predicted_unique_rate'] * 100:.1f}%")
    print("Évolution loss générale :", round(language_delta, 6))
    for family, values in current["families"].items():
        print(
            family,
            "| réussite", f"{values['semantic_success_rate'] * 100:.1f}%",
            "| exact", f"{values['exact_rate'] * 100:.1f}%",
            "| similarité", round(values["mean_similarity"], 3),
        )
    print("Candidat pour contrôle humain :", candidate)

    print("\n7/7 — Intégrité", flush=True)
    latest_sha = sha256(OUTPUT_DIR / "latest.pt")
    if latest_sha != progress["latest_checkpoint_sha256"]:
        raise RuntimeError("latest.pt corrompu")
    decision = {
        "stage": "assistant_v1.5_stage6_cpt500_bootstrap",
        "current_step": 125,
        "parent_sha256": EXPECTED_PARENT_SHA,
        "feedback_sha256": EXPECTED_FEEDBACK_SHA,
        "human_corrections_training": report["audit"]["feedback"]["training"],
        "human_corrections_holdout": report["audit"]["feedback"]["holdout"],
        "quality_delta": quality_delta,
        "general_language_loss_delta": language_delta,
        **guards,
        "candidate_for_human_holdout": candidate,
        "sealed_test_opened": False,
    }
    (OUTPUT_DIR / "PILOT_DECISION_STEP125.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SHA256 latest :", latest_sha)
    print("STAGE 6 — 0 → 125 SAUVEGARDÉ ✅")
    print("Test final toujours scellé ✅")


if __name__ == "__main__":
    main()
