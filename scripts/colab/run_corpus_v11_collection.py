#!/usr/bin/env python3
"""Collecte Colab v1.1 : français naturel et conversations ouvertes, sans entraînement."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path("/content/drive/MyDrive/IvoireSLM/v1.4/corpus_v1.1")
WIKIPEDIA = ROOT / "sources/wikipedia_fr_natural_v0.3"
OASST = ROOT / "sources/openassistant_fr_v0.1"
IVOIRIAN = ROOT / "sources/ivoirian_conversations_consented_v0.1.jsonl"
DATASET = ROOT / "datasets/ivoireslm_natural_french_supplement_v1.1.0"
AUDIT = ROOT / "audits/corpus_mix_v1.1.0.json"


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-u", *arguments], check=True)


print("1/4 — Collecte du français encyclopédique attribué")
if not WIKIPEDIA.exists():
    run(str(PROJECT / "scripts/data/snapshot_wikipedia_fr_natural_v03.py"),
        "--output-dir", str(WIKIPEDIA), "--minimum-characters", "30000000")
else:
    print("Snapshot Wikipédia déjà présent ✅")

print("\n2/4 — Collecte des conversations humaines françaises ouvertes")
if not OASST.exists():
    run(str(PROJECT / "scripts/data/snapshot_openassistant_fr_v01.py"), "--output-dir", str(OASST))
else:
    print("Snapshot OASST1 français déjà présent ✅")

print("\n3/4 — Construction du candidat")
if not DATASET.exists():
    command = [
        str(PROJECT / "scripts/data/build_natural_french_supplement_v11.py"),
        "--wikipedia-dir", str(WIKIPEDIA), "--oasst-dir", str(OASST),
        "--output-dir", str(DATASET),
    ]
    if IVOIRIAN.is_file():
        command.extend(["--ivoirian-conversations", str(IVOIRIAN)])
    run(*command)
else:
    print("Dataset candidat déjà présent ✅")

print("\n4/4 — Audit bloquant avant entraînement")
AUDIT.parent.mkdir(parents=True, exist_ok=True)
if not AUDIT.exists():
    run(str(PROJECT / "scripts/data/audit_corpus_mix_v11.py"),
        "--report", str(DATASET / "report.json"), "--output", str(AUDIT))
else:
    print("Audit déjà présent ✅")
result = json.loads(AUDIT.read_text(encoding="utf-8"))
print("\nCollecte v1.1 terminée ✅")
print("Entraînement autorisé :", result["training_authorized"])
print("Gardes manquantes :", result["failed_checks"])
print("Aucun entraînement n'a été lancé. Test final toujours scellé ✅")
