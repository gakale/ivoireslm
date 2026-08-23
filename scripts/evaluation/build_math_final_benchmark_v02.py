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
OUTPUT_ROOT = STORAGE_ROOT / "benchmarks/math_reasoning_final_v0.2"
DEV_BENCHMARK = STORAGE_ROOT / "benchmarks/math_reasoning_v0.1/benchmark.jsonl"
TRAINING_MATH_ROOT = STORAGE_ROOT / "derived/math_verified_v0.1"
SEED = 20260825
EXERCISES_PER_FAMILY = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_forbidden_problems() -> set[str]:
    paths = [DEV_BENCHMARK]
    paths.extend(TRAINING_MATH_ROOT / f"{split}.jsonl" for split in ("train", "validation", "test"))
    problems = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as stream:
            problems.update(json.loads(line)["problem"] for line in stream if line.strip())
    return problems


def main() -> None:
    forbidden = load_forbidden_problems()
    seen, rows = set(), []
    attempts = 0
    for family in BENCHMARK_FAMILIES:
        accepted, index = 0, 0
        while accepted < EXERCISES_PER_FAMILY:
            attempts += 1
            record = generate_benchmark_exercise(family, index, seed=SEED)
            index += 1
            if not verify_benchmark_exercise(record):
                raise RuntimeError(record["benchmark_id"])
            if record["problem"] in forbidden or record["problem"] in seen:
                continue
            record["benchmark_id"] = f"final_v02_{family}_{accepted:06d}"
            seen.add(record["problem"])
            rows.append(record)
            accepted += 1
            if index > EXERCISES_PER_FAMILY * 100:
                raise RuntimeError(f"pas assez de questions uniques pour {family}")

    rows.sort(key=lambda row: row["benchmark_id"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    benchmark_path = OUTPUT_ROOT / "benchmark.jsonl"
    with benchmark_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "benchmark_id": "math_reasoning_final_v0.2",
        "status": "sealed_before_math_sft_v0.1_creation",
        "seed": SEED,
        "total_exercises": len(rows),
        "candidate_attempts": attempts,
        "families": dict(sorted(Counter(row["family"] for row in rows).items())),
        "answers_programmatically_verified": len(rows),
        "overlap_with_pretraining_or_development": 0,
        "benchmark_sha256": sha256(benchmark_path),
        "opening_policy": "evaluate_once after SFT checkpoint selection",
    }
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
