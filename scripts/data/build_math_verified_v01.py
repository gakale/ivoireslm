#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.math_verified import FAMILIES, generate_exercise, split_for_exercise, verify_exercise


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
OUTPUT_ROOT = STORAGE_ROOT / "derived/math_verified_v0.1"
EXERCISES_PER_FAMILY = 800
SEED = 20260822


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: [] for split in ("train", "validation", "test")}
    for family in FAMILIES:
        for index in range(EXERCISES_PER_FAMILY):
            record = generate_exercise(family, index, seed=SEED)
            if not verify_exercise(record):
                raise RuntimeError(f"échec de vérification : {record['exercise_id']}")
            split = split_for_exercise(record["exercise_id"])
            rows_by_split[split].append({**record, "split": split})

    artifacts = {}
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: row["exercise_id"])
        text_path = OUTPUT_ROOT / f"{split}.txt"
        jsonl_path = OUTPUT_ROOT / f"{split}.jsonl"
        text_path.write_text("\n".join(row["text"].rstrip() for row in rows) + "\n", encoding="utf-8")
        with jsonl_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        artifacts[split] = {
            "exercises": len(rows),
            "characters": text_path.stat().st_size,
            "families": dict(sorted(Counter(row["family"] for row in rows).items())),
            "text_path": str(text_path),
            "text_sha256": sha256(text_path),
            "jsonl_path": str(jsonl_path),
            "jsonl_sha256": sha256(jsonl_path),
        }

    report = {
        "dataset_id": "math_verified_v0.1",
        "generation": "deterministic_local_rule_based",
        "seed": SEED,
        "answer_verification": "100_percent_programmatic",
        "license": "CC0-1.0",
        "families": list(FAMILIES),
        "exercises_per_family": EXERCISES_PER_FAMILY,
        "total_exercises": sum(item["exercises"] for item in artifacts.values()),
        "splits": artifacts,
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

