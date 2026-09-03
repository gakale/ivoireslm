#!/usr/bin/env python3
"""Cellule Python autonome pour lancer le constructeur v1.0 depuis Colab."""

import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/content/drive/MyDrive/IvoireSLM — Corpus STEM & Cybersécurité")
OUTPUT = Path("/content/drive/MyDrive/IvoireSLM/v1.0/datasets/ivoireslm_corpus_supplement_v1.0.0")
SCRIPT = PROJECT / "scripts/data/build_corpus_supplement_v10.py"

for path in (SOURCE_ROOT, SCRIPT):
    if not path.exists():
        raise FileNotFoundError(path)

print("Construction du supplément v1.0…")
print("Sources :", SOURCE_ROOT)
print("Sortie  :", OUTPUT)
subprocess.run(
    [sys.executable, "-u", str(SCRIPT), "--source-root", str(SOURCE_ROOT), "--output-dir", str(OUTPUT)],
    check=True,
)
print("\nConstruction terminée ✅")
