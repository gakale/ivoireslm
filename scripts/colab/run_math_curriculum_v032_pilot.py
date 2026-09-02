#!/usr/bin/env python3
"""Installe et lance le pilote Colab du curriculum mathématique v0.3.2."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import torch


ARCHIVE = Path(
    "/content/drive/MyDrive/IvoireSLM/v0.4/artifacts/"
    "ivoireslm_math_curriculum_v032_tables_code.tar.gz"
)
EXPECTED_ARCHIVE_SHA = (
    "5fb6802f686f22960d65afbfb52f5d563540ca0449b2a4390c753c93cc77d076"
)
CODE_ROOT = Path("/content/ivoireslm_math_curriculum_v032_tables")
BASE_ARCHIVE = Path(
    "/content/drive/MyDrive/IvoireSLM/v0.4/artifacts/"
    "ivoireslm_colab_v0.4_17m_bundle.tar.gz"
)
EXPECTED_BASE_ARCHIVE_SHA = (
    "fc7b93b8e796ff8657c70513b49a586cfceca23778aa6038e4f9ee3e4af2cba3"
)
BASE_RESTORE_ROOT = Path("/content/ivoireslm_v04_base_restore")
DATASET_DIR = Path(
    "/content/drive/MyDrive/IvoireSLM/v0.4/datasets/"
    "math_curriculum_v0.3.2_tables"
)
BPE_DIR = BASE_RESTORE_ROOT / "bpe_v0.4"
PARENT = Path(
    "/content/drive/MyDrive/IvoireSLM/v0.4/"
    "checkpoints_17m_math_v031_stage1/best.pt"
)
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/IvoireSLM/v0.4/"
    "checkpoints_17m_math_v032_tables"
)
EXPECTED_DATA = {
    "train.jsonl": "de29e35ea49dbda717ebf5b8d32539437de34b13e780a49159317d6a24cd9f0f",
    "validation.jsonl": "19b61ff7e3b899bf33ce4e5c1c3b75c932e28b8cb90f6524c4659624295d197a",
    "diagnostic_generalization.jsonl": "96efa91c83278d2a17836d5fe0a66c2c6790759445095b1ca2eb2a83807c9a48",
    "report.json": "83ccd3a8561500827de592ae48f774ee19e6f1cebaddf654ddaaffa8e92e250b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"chemin d'archive non sûr : {member.name}")
        bundle.extractall(destination)


def main() -> None:
    print("1/6 — GPU et archive", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent")
    print("GPU :", torch.cuda.get_device_name(0), flush=True)
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    archive_sha = sha256_file(ARCHIVE)
    print("Archive SHA256 :", archive_sha, flush=True)
    if archive_sha != EXPECTED_ARCHIVE_SHA:
        raise RuntimeError("archive v0.3.2 non conforme")

    print("\n2/6 — Extraction protégée", flush=True)
    if not CODE_ROOT.exists():
        safe_extract(ARCHIVE, CODE_ROOT)
    manifest_path = CODE_ROOT / "docs/MATH_CURRICULUM_V032_CODE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = CODE_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"fichier de code non conforme : {relative}")
        print(relative, "✅", flush=True)
    builder = CODE_ROOT / "scripts/data/build_math_curriculum_v032.py"
    trainer = CODE_ROOT / "scripts/training/finetune_math_curriculum_v032_17m.py"
    model_script = CODE_ROOT / "scripts/training/train_transformer_ci_v04_17m.py"
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(builder), str(trainer)], check=True
    )

    print("\n3/6 — Dataset", flush=True)
    if not DATASET_DIR.exists():
        subprocess.run(
            [sys.executable, "-u", str(builder), "--output-dir", str(DATASET_DIR)],
            check=True,
        )
    for name, expected in EXPECTED_DATA.items():
        path = DATASET_DIR / name
        actual = sha256_file(path)
        print(name, actual, "✅" if actual == expected else "❌", flush=True)
        if actual != expected:
            raise RuntimeError(f"dataset non conforme : {name}")
    if (DATASET_DIR / "test.jsonl").exists():
        raise RuntimeError("split test inattendu")
    report = json.loads((DATASET_DIR / "report.json").read_text(encoding="utf-8"))
    print("Train :", report["splits"]["train"]["examples"], flush=True)
    print("Validation rappel :", report["splits"]["validation"]["examples"], flush=True)
    print(
        "Diagnostic inédit :",
        report["splits"]["diagnostic_generalization"]["examples"],
        flush=True,
    )
    print("Test absent ✅", flush=True)

    print("\n4/6 — Parent scientifique", flush=True)
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    parent_sha = sha256_file(PARENT)
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    print("SHA256 parent :", parent_sha, flush=True)
    print("Étape parent :", parent["step"], flush=True)
    print("Score parent :", parent["best_generation_quality"], flush=True)
    if int(parent["step"]) != 500:
        raise RuntimeError("le parent n'est pas le meilleur checkpoint étape 500")
    if parent.get("sealed_test_opened") is not False:
        raise RuntimeError("parent non conforme")
    if abs(float(parent["best_generation_quality"]) - 0.09333333333333332) >= 1e-12:
        raise RuntimeError("score du parent inattendu")
    print("Meilleur checkpoint v0.3.1 étape 500 confirmé ✅", flush=True)

    print("\n5/6 — Restauration BPE et protection de la branche", flush=True)
    required_bpe = (
        BPE_DIR / "tokenizer.json",
        BPE_DIR / "train.uint16.bin",
        BPE_DIR / "validation.uint16.bin",
    )
    if not all(path.is_file() for path in required_bpe):
        if BASE_RESTORE_ROOT.exists():
            raise FileExistsError(
                "restauration BPE partielle détectée ; aucun fichier ne sera écrasé : "
                f"{BASE_RESTORE_ROOT}"
            )
        if not BASE_ARCHIVE.is_file():
            raise FileNotFoundError(BASE_ARCHIVE)
        base_sha = sha256_file(BASE_ARCHIVE)
        print("Archive BPE SHA256 :", base_sha, flush=True)
        if base_sha != EXPECTED_BASE_ARCHIVE_SHA:
            raise RuntimeError("archive BPE v0.4 non conforme")
        safe_extract(BASE_ARCHIVE, BASE_RESTORE_ROOT)
    for path in required_bpe:
        if not path.is_file():
            raise FileNotFoundError(path)
        print(path.name, "✅", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected = [OUTPUT_DIR / name for name in ("best.pt", "latest.pt", "progress.json")]
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError("branche déjà présente, aucun écrasement : " + ", ".join(existing))
    print("Nouvelle sortie :", OUTPUT_DIR, flush=True)

    print("\n6/6 — Pilote tables 0 → 250", flush=True)
    print(
        "La référence inclut le diagnostic inédit et peut prendre plusieurs minutes.\n",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(trainer),
            "--dataset-dir",
            str(DATASET_DIR),
            "--data-dir",
            str(BPE_DIR),
            "--base-checkpoint",
            str(PARENT),
            "--model-script",
            str(model_script),
            "--output-dir",
            str(OUTPUT_DIR),
            "--stop-step",
            "250",
        ],
        check=True,
    )

    progress_path = OUTPUT_DIR / "progress.json"
    latest_path = OUTPUT_DIR / "latest.pt"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress["current_step"] != 250 or progress["sealed_test_opened"] is not False:
        raise RuntimeError("progression finale non conforme")
    if progress["diagnostic_generalization_used_for_selection"] is not False:
        raise RuntimeError("le diagnostic a influencé la sélection")
    if sha256_file(latest_path) != progress["latest_checkpoint_sha256"]:
        raise RuntimeError("checkpoint latest corrompu")

    print("\nRÉSULTAT SYNTHÉTIQUE", flush=True)
    print("Étape :", progress["current_step"], flush=True)
    print("Meilleur checkpoint :", progress["best_step"], flush=True)
    print("Score pondéré :", f"{progress['best_generation_quality']:.2%}", flush=True)
    for family, values in progress["last_table_recall_validation"]["families"].items():
        print(
            f"{family}: {values['correct']}/{values['examples']} "
            f"({values['exact_rate']:.1%})",
            flush=True,
        )
    diagnostic = progress["last_unseen_generalization_diagnostic"]["families"][
        "multiplication_unseen_generalization"
    ]
    print(
        f"Diagnostic inédit 13–20 : {diagnostic['correct']}/"
        f"{diagnostic['examples']} ({diagnostic['exact_rate']:.1%})",
        flush=True,
    )
    print("Garde langue :", progress["language_guard_passed"], flush=True)
    print("Pilote réussi :", progress["pilot_passed"], flush=True)
    print("SHA256 latest :", progress["latest_checkpoint_sha256"], flush=True)
    print("Test ouvert :", progress["sealed_test_opened"], flush=True)
    print("\nV0.3.2 TABLES — PALIER 0 → 250 TERMINÉ ✅", flush=True)
    print("TEST TOUJOURS SCELLÉ ✅", flush=True)


if __name__ == "__main__":
    main()
