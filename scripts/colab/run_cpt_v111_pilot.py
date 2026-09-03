#!/usr/bin/env python3
"""Tokenise le corpus v1.1.1 puis lance uniquement le pilote CPT 0 -> 125."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
DRIVE = Path("/content/drive/MyDrive/IvoireSLM")
CORPUS_ROOT = DRIVE / "v1.4/corpus_v1.1"
DATASET = CORPUS_ROOT / "datasets/ivoireslm_corpus_supplement_v1.1.1"
AUDIT = CORPUS_ROOT / "audits/corpus_mix_v1.1.1.json"
TOKENIZED = CORPUS_ROOT / "tokenized/ivoireslm_corpus_supplement_v1.1.1_bpe_v0.4"
BASE_DATA = Path("/content/ivoireslm_v04_base_restore/bpe_v0.4")
BASE_CHECKPOINT = DRIVE / "v0.4/checkpoints_17m/best.pt"
OUTPUT = DRIVE / "v1.4/checkpoints_17m_cpt_v111_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-u", *arguments], check=True)


print("1/5 — Vérification de l'audit")
for path in (DATASET / "report.json", AUDIT, BASE_CHECKPOINT):
    if not path.is_file():
        raise FileNotFoundError(path)
audit = json.loads(AUDIT.read_text(encoding="utf-8"))
if audit.get("dataset_id") != "ivoireslm_corpus_supplement_v1.1.1":
    raise RuntimeError("audit d'un autre dataset")
if audit.get("cpt_training_authorized") is not True:
    raise RuntimeError(f"gardes CPT non satisfaites : {audit.get('failed_checks')}")
if audit.get("test_opened") is not False:
    raise RuntimeError("le test ne doit pas être ouvert")
print("Audit CPT validé ✅")
print("Supervision conversationnelle autorisée :", audit.get("assistant_sft_authorized"))

print("\n2/5 — Vérification du BPE v0.4")
for path in (
    BASE_DATA / "tokenizer.json",
    BASE_DATA / "train.uint16.bin",
    BASE_DATA / "validation.uint16.bin",
    BASE_DATA / "report.json",
):
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} absent. Restaure d'abord le bundle v0.4 dans "
            "/content/ivoireslm_v04_base_restore."
        )
print("BPE et corpus de référence présents ✅")
print("Checkpoint parent SHA256 :", sha256(BASE_CHECKPOINT))

print("\n3/5 — Tokenisation du supplément")
if not TOKENIZED.exists():
    TOKENIZED.parent.mkdir(parents=True, exist_ok=True)
    run(
        str(PROJECT / "scripts/tokenizer/tokenize_corpus_supplement_v111.py"),
        "--supplement-root", str(DATASET),
        "--tokenizer", str(BASE_DATA / "tokenizer.json"),
        "--output-root", str(TOKENIZED),
    )
else:
    print("Supplément déjà tokenisé : réutilisation ✅")
tokenized_report = json.loads((TOKENIZED / "report.json").read_text(encoding="utf-8"))
if tokenized_report.get("source_dataset_id") != "ivoireslm_corpus_supplement_v1.1.1":
    raise RuntimeError("supplément tokenisé incompatible")
if tokenized_report.get("test_created") is not False:
    raise RuntimeError("split test inattendu")

print("\n4/5 — Protection de l'expérience")
OUTPUT.mkdir(parents=True, exist_ok=True)
if any(OUTPUT.iterdir()):
    raise FileExistsError(
        f"{OUTPUT} contient déjà une expérience. Ne l'écrase pas et utilise "
        "une cellule de reprise dédiée."
    )
print("Nouveau dossier expérimental :", OUTPUT)

print("\n5/5 — Pilote CPT 0 → 125")
print("L'évaluation initiale peut rester silencieuse plusieurs minutes.", flush=True)
run(
    str(PROJECT / "scripts/training/continue_pretraining_mix_v111_17m.py"),
    "--base-data-dir", str(BASE_DATA),
    "--supplement-data-dir", str(TOKENIZED),
    "--base-checkpoint", str(BASE_CHECKPOINT),
    "--model-script", str(PROJECT / "scripts/training/train_transformer_ci_v04_17m.py"),
    "--output-dir", str(OUTPUT),
    "--stop-step", "125",
)
print("\nPILOTE CPT V1.1.1 TERMINÉ ✅")
print("Aucune supervision lancée. Test final toujours scellé ✅")
