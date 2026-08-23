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

from data.math_sft import SFT_FAMILIES, generate_sft_exercise, verify_sft_exercise


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
OUTPUT_ROOT = STORAGE_ROOT / "derived/math_sft_v0.1"
FORBIDDEN_BENCHMARKS = (
    STORAGE_ROOT / "benchmarks/math_reasoning_v0.1/benchmark.jsonl",
    STORAGE_ROOT / "benchmarks/math_reasoning_final_v0.2/benchmark.jsonl",
)
EXAMPLES_PER_FAMILY = 5_000
SEED = 20260826


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forbidden_problems() -> set[str]:
    problems = set()
    for path in FORBIDDEN_BENCHMARKS:
        if not path.is_file():
            raise FileNotFoundError(f"benchmark à geler avant le SFT : {path}")
        with path.open(encoding="utf-8") as stream:
            problems.update(json.loads(line)["problem"] for line in stream if line.strip())
    return problems


def validation_split(identifier: str) -> bool:
    value = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "big") % 100
    return value < 4


def main() -> None:
    forbidden = forbidden_problems()
    seen, rows = set(), []
    attempts = 0
    for family in SFT_FAMILIES:
        accepted, index = 0, 0
        while accepted < EXAMPLES_PER_FAMILY:
            attempts += 1
            record = generate_sft_exercise(family, index, seed=SEED)
            index += 1
            if not verify_sft_exercise(record):
                raise RuntimeError(record["example_id"])
            if record["problem"] in forbidden or record["problem"] in seen:
                continue
            seen.add(record["problem"])
            record["split"] = "validation" if validation_split(record["example_id"]) else "train"
            rows.append(record)
            accepted += 1
            if index > EXAMPLES_PER_FAMILY * 200:
                raise RuntimeError(f"pas assez d'exemples uniques pour {family}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for split in ("train", "validation"):
        split_rows = sorted((row for row in rows if row["split"] == split), key=lambda row: row["example_id"])
        path = OUTPUT_ROOT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        lengths = [len(row["text"]) for row in split_rows]
        artifacts[split] = {
            "examples": len(split_rows),
            "families": dict(sorted(Counter(row["family"] for row in split_rows).items())),
            "difficulties": dict(sorted(Counter(str(row["difficulty"]) for row in split_rows).items())),
            "characters": sum(lengths),
            "maximum_example_characters": max(lengths),
            "examples_over_256_characters": sum(length > 256 for length in lengths),
            "path": str(path),
            "sha256": sha256(path),
        }

    report = {
        "dataset_id": "math_sft_v0.1",
        "status": "training_and_validation_only",
        "seed": SEED,
        "license": "CC0-1.0",
        "generation": "deterministic_local_rule_based",
        "total_examples": len(rows),
        "candidate_attempts": attempts,
        "answers_programmatically_verified": len(rows),
        "duplicate_problems": 0,
        "overlap_with_development_or_final_benchmarks": 0,
        "target_loss_only": True,
        "splits": artifacts,
    }
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
