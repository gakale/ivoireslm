#!/usr/bin/env python3
"""Complète le corpus v1.1 avec les données ouvertes ivoiriennes, sans entraînement."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path("/content/drive/MyDrive/IvoireSLM/v1.4/corpus_v1.1")
WIKIPEDIA = ROOT / "sources/wikipedia_fr_natural_v0.3"
OASST = ROOT / "sources/openassistant_fr_v0.1"
OPEN_DATA = ROOT / "sources/data_gouv_ci_open_v0.1.1"
IVOIRIAN_CONVERSATIONS = ROOT / "sources/ivoirian_conversations_consented_v0.1.jsonl"
IVOIRIAN_LANGUAGES = ROOT / "sources/ivoirian_languages_verified_v0.1"
DATASET = ROOT / "datasets/ivoireslm_corpus_supplement_v1.1.1"
AUDIT = ROOT / "audits/corpus_mix_v1.1.1.json"


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-u", *arguments], check=True)


print("1/4 — Collecte des données publiques ivoiriennes attribuées")
if not OPEN_DATA.exists():
    run(
        str(PROJECT / "scripts/data/snapshot_datagouvci_open_v011.py"),
        "--output-dir", str(OPEN_DATA),
    )
else:
    print("Snapshot data.gouv.ci déjà présent ✅")

print("\n2/4 — Construction du supplément v1.1.1")
if not DATASET.exists():
    command = [
        str(PROJECT / "scripts/data/build_natural_french_supplement_v11.py"),
        "--wikipedia-dir", str(WIKIPEDIA),
        "--oasst-dir", str(OASST),
        "--ivoirian-open-data-dir", str(OPEN_DATA),
        "--dataset-id", "ivoireslm_corpus_supplement_v1.1.1",
        "--output-dir", str(DATASET),
    ]
    if IVOIRIAN_CONVERSATIONS.is_file():
        command.extend(["--ivoirian-conversations", str(IVOIRIAN_CONVERSATIONS)])
    if (IVOIRIAN_LANGUAGES / "documents.jsonl").is_file():
        command.extend(["--ivoirian-languages-dir", str(IVOIRIAN_LANGUAGES)])
    run(*command)
else:
    print("Dataset v1.1.1 déjà présent ✅")

print("\n3/4 — Audit CPT et SFT séparés")
AUDIT.parent.mkdir(parents=True, exist_ok=True)
if not AUDIT.exists():
    run(
        str(PROJECT / "scripts/data/audit_corpus_mix_v11.py"),
        "--report", str(DATASET / "report.json"),
        "--output", str(AUDIT),
    )
else:
    print("Audit v1.1.1 déjà présent ✅")

print("\n4/4 — Décision")
result = json.loads(AUDIT.read_text(encoding="utf-8"))
print("Continuation de préentraînement autorisée :", result["cpt_training_authorized"])
print("Nouvelle supervision autorisée :", result["assistant_sft_authorized"])
print("Gardes CPT manquantes :", result["failed_checks"])
print("Gardes SFT manquantes :", result["failed_sft_checks"])
print("Aucun entraînement lancé. Test final toujours scellé ✅")
