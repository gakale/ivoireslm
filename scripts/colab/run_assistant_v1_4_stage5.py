#!/usr/bin/env python3
"""Restaure les dépendances et lance le pilote Stage 5 dans Colab."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import torch


PROJECT = Path(__file__).resolve().parents[2]
BASE_ARCHIVE = Path(
    "/content/drive/MyDrive/IvoireSLM/v0.4/artifacts/"
    "ivoireslm_colab_v0.4_17m_bundle.tar.gz"
)
EXPECTED_BASE_ARCHIVE_SHA = "fc7b93b8e796ff8657c70513b49a586cfceca23778aa6038e4f9ee3e4af2cba3"
BASE_RESTORE_ROOT = Path("/content/ivoireslm_v04_base_restore")
BPE_DIR = BASE_RESTORE_ROOT / "bpe_v0.4"
PARENT = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.2/"
    "checkpoints_17m_assistant_v1_3_stage4/best.pt"
)
EXPECTED_PARENT_SHA = "ccb26306e027f2b99e3f5ce20e1cf08ad89ead96c0877645c7c0292e5213dbf8"
FEEDBACK = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.1/human_feedback/"
    "assistant_stage3_step250_feedback.jsonl"
)
EXPECTED_FEEDBACK_SHA = "6f97b00a90bf5f5121cfb9cddb6c6237bdeeee788f6b8baca7c8da672d0e674f"
DATASET_DIR = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.3/datasets/"
    "assistant_curriculum_v1_4_stage5"
)
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.3/"
    "checkpoints_17m_assistant_v1_4_stage5"
)


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
    print("1/7 — GPU et preuves", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : active un T4")
    print("GPU :", torch.cuda.get_device_name(0), flush=True)
    if not PARENT.is_file() or sha256(PARENT) != EXPECTED_PARENT_SHA:
        raise RuntimeError("checkpoint parent Stage 4 incorrect")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 125:
        raise RuntimeError("le parent doit être le Stage 4 étape 125")
    if not FEEDBACK.is_file() or sha256(FEEDBACK) != EXPECTED_FEEDBACK_SHA:
        raise RuntimeError("benchmark de développement absent ou modifié")
    print("Parent Stage 4 étape 125 intact ✅", flush=True)
    print("66 questions de développement intactes ✅", flush=True)

    print("\n2/7 — Restauration automatique du BPE", flush=True)
    try:
        verify_bpe()
    except (FileNotFoundError, RuntimeError):
        if not BASE_ARCHIVE.is_file() or sha256(BASE_ARCHIVE) != EXPECTED_BASE_ARCHIVE_SHA:
            raise RuntimeError("archive BPE officielle absente ou corrompue")
        safe_extract(BASE_ARCHIVE, BASE_RESTORE_ROOT)
        verify_bpe()
    print("Tokenizer et corpus général vérifiés ✅", flush=True)

    print("\n3/7 — Construction du curriculum Stage 5", flush=True)
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        report = json.loads((DATASET_DIR / "report.json").read_text(encoding="utf-8"))
    else:
        DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(PROJECT / "scripts/data/build_assistant_curriculum_v1_4.py"),
                "--output-dir",
                str(DATASET_DIR),
                "--development-feedback",
                str(FEEDBACK),
            ],
            check=True,
        )
        report = json.loads((DATASET_DIR / "report.json").read_text(encoding="utf-8"))
    if report["audit"]["development_questions"] != 66:
        raise RuntimeError("les 66 questions n’ont pas été exclues correctement")
    for split in ("train", "validation"):
        if sha256(DATASET_DIR / f"{split}.jsonl") != report["splits"][split]["sha256"]:
            raise RuntimeError(f"empreinte dataset invalide : {split}")
    if (DATASET_DIR / "test.jsonl").exists():
        raise RuntimeError("split test inattendu")
    print("Réponses humaines jamais chargées comme cibles ✅", flush=True)
    print("Questions exactes du benchmark exclues ✅", flush=True)

    print("\n4/7 — Protection de la nouvelle expérience", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any((OUTPUT_DIR / name).exists() for name in ("best.pt", "latest.pt", "progress.json")):
        raise FileExistsError("le Stage 5 existe déjà ; refus de l’écraser")
    print("Nouveau dossier :", OUTPUT_DIR, flush=True)

    print("\n5/7 — Pilote Stage 5 : 0 → 125", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(PROJECT / "scripts/training/finetune_assistant_v1_4_stage5_17m.py"),
            "--dataset-dir",
            str(DATASET_DIR),
            "--data-dir",
            str(BPE_DIR),
            "--base-checkpoint",
            str(PARENT),
            "--model-script",
            str(PROJECT / "scripts/training/train_transformer_ci_v04_17m.py"),
            "--output-dir",
            str(OUTPUT_DIR),
            "--stop-step",
            "125",
        ],
        check=True,
    )

    print("\n6/7 — Décision automatique", flush=True)
    progress = json.loads((OUTPUT_DIR / "progress.json").read_text(encoding="utf-8"))
    baseline = progress["baseline_validation"]
    generation = progress["last_generation_validation"]
    initial_quality = baseline["generation"]["quality_score"]
    current_quality = generation["quality_score"]
    initial_language = baseline["general_language"]["loss_nats"]
    current_language = progress["last_general_language_validation"]["loss_nats"]
    language_delta = current_language - initial_language
    quality_improved = current_quality > initial_quality
    direct_guard = bool(generation.get("direct_answer_guard_passed"))
    collapse_guard = bool(generation.get("collapse_guard_passed"))
    language_guard = language_delta <= 0.01
    candidate = quality_improved and direct_guard and collapse_guard and language_guard
    print("Qualité initiale :", round(initial_quality, 4))
    print("Qualité actuelle :", round(current_quality, 4))
    print("Échos/fragments :", f"{generation['echo_fragment_rate'] * 100:.1f}%")
    print("Réponses vides ou d’un mot :", f"{generation['empty_or_one_word_rate'] * 100:.1f}%")
    print("Garde réponse directe :", direct_guard)
    print("Garde diversité :", collapse_guard)
    print("Évolution loss générale :", round(language_delta, 6))
    print("Candidat pour comparaison de développement :", candidate)

    print("\n7/7 — Intégrité et scellement", flush=True)
    latest_sha = sha256(OUTPUT_DIR / "latest.pt")
    if latest_sha != progress["latest_checkpoint_sha256"]:
        raise RuntimeError("latest.pt corrompu")
    decision = {
        "stage": "assistant_v1_4_stage5_direct_answer",
        "current_step": 125,
        "parent_sha256": EXPECTED_PARENT_SHA,
        "development_feedback_sha256": EXPECTED_FEEDBACK_SHA,
        "human_answers_used_for_training": 0,
        "quality_improved": quality_improved,
        "direct_answer_guard_passed": direct_guard,
        "collapse_guard_passed": collapse_guard,
        "general_language_loss_delta": language_delta,
        "general_language_guard_passed": language_guard,
        "candidate_for_development_comparison": candidate,
        "sealed_test_opened": False,
    }
    (OUTPUT_DIR / "PILOT_DECISION_STEP125.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SHA256 latest :", latest_sha)
    print("STAGE 5 — 0 → 125 SAUVEGARDÉ ✅")
    print("Test final toujours scellé ✅")


if __name__ == "__main__":
    main()

