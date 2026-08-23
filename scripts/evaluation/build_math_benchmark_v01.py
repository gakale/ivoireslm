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

from evaluation.math_benchmark import BENCHMARK_FAMILIES, generate_benchmark_exercise, verify_benchmark_exercise


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
TRAINING_MATH_ROOT = STORAGE_ROOT / "derived/math_verified_v0.1"
OUTPUT_ROOT = STORAGE_ROOT / "benchmarks/math_reasoning_v0.1"
EXERCISES_PER_FAMILY = 100
SEED = 20260824


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_problems() -> set[str]:
    problems = set()
    for split in ("train", "validation", "test"):
        path = TRAINING_MATH_ROOT / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"split mathématique source manquant : {path}")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                problems.add(json.loads(line)["problem"])
    return problems


def main() -> None:
    seen_training = training_problems()
    rows = []
    seen_benchmark = set()
    candidate_attempts = 0
    for family in BENCHMARK_FAMILIES:
        accepted_for_family = 0
        index = 0
        while accepted_for_family < EXERCISES_PER_FAMILY:
            candidate_attempts += 1
            record = generate_benchmark_exercise(family, index, seed=SEED)
            index += 1
            if not verify_benchmark_exercise(record):
                raise RuntimeError(f"réponse non vérifiée : {record['benchmark_id']}")
            if record["problem"] in seen_training:
                continue
            if record["problem"] in seen_benchmark:
                continue
            seen_benchmark.add(record["problem"])
            rows.append(record)
            accepted_for_family += 1
            if index > EXERCISES_PER_FAMILY * 100:
                raise RuntimeError(f"impossible de produire assez de questions uniques pour {family}")

    rows.sort(key=lambda row: row["benchmark_id"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    benchmark_path = OUTPUT_ROOT / "benchmark.jsonl"
    with benchmark_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "benchmark_id": "math_reasoning_v0.1",
        "status": "frozen_before_first_model_evaluation",
        "generation": "deterministic_local_rule_based",
        "seed": SEED,
        "license": "CC0-1.0",
        "total_exercises": len(rows),
        "candidate_attempts": candidate_attempts,
        "families": dict(sorted(Counter(row["family"] for row in rows).items())),
        "answers_programmatically_verified": len(rows),
        "exact_problem_overlap_with_math_verified_v0.1": 0,
        "internal_duplicate_problems": 0,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256(benchmark_path),
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
