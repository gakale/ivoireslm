#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
SOURCE_PATH = STORAGE_ROOT / "derived/math_sft_v0.1/validation.jsonl"
OUTPUT_ROOT = STORAGE_ROOT / "benchmarks/math_sft_validation_diagnostic_v0.1"
PER_FAMILY_AND_DIFFICULTY = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_from(row: dict) -> dict:
    family, values = row["family"], row["verification"]
    if family in {"addition", "multiplication", "percentage"}:
        return {"kind": "integer", "value": values["result"]}
    if family == "linear_equation":
        return {"kind": "integer", "value": values["root"]}
    if family == "quadratic_factorization":
        return {"kind": "roots", "values": values["roots"]}
    if family == "fraction_reduction":
        return {"kind": "fraction", "numerator": values["reduced_numerator"], "denominator": values["reduced_denominator"]}
    if family == "rectangle":
        return {"kind": "rectangle", "area": values["area"], "perimeter": values["perimeter"]}
    if family == "arithmetic_sequence":
        return {"kind": "integer", "value": values["result"]}
    if family == "ivorian_market_change":
        return {"kind": "integer", "value": values["change"]}
    if family == "ivorian_cooperative_sharing":
        return {"kind": "sharing", "quotient": values["quotient"], "remainder": values["remainder"]}
    raise ValueError(family)


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    with SOURCE_PATH.open(encoding="utf-8") as stream:
        source = [json.loads(line) for line in stream if line.strip()]
    cells = defaultdict(list)
    for row in source:
        cells[(row["family"], row["difficulty"])].append(row)
    selected = []
    families = sorted({family for family, _ in cells})
    for family in families:
        family_selected = []
        selected_ids = set()
        for difficulty in range(1, 5):
            rows = sorted(cells[(family, difficulty)], key=lambda row: row["example_id"])
            for row in rows[:PER_FAMILY_AND_DIFFICULTY]:
                family_selected.append(row)
                selected_ids.add(row["example_id"])
        if len(family_selected) < 100:
            remaining = sorted(
                (row for row in source if row["family"] == family and row["example_id"] not in selected_ids),
                key=lambda row: row["example_id"],
            )
            family_selected.extend(remaining[: 100 - len(family_selected)])
        if len(family_selected) != 100:
            raise RuntimeError(f"famille insuffisante {family}: {len(family_selected)}")
        selected.extend(family_selected)

    benchmark = []
    for row in selected:
        benchmark.append(
            {
                "benchmark_id": f"validation_diagnostic_{row['example_id']}",
                "family": row["family"],
                "difficulty": row["difficulty"],
                "problem": row["problem"],
                "prompt": row["prompt"],
                "reference_answer": row["answer"],
                "expected": expected_from(row),
            }
        )
    benchmark.sort(key=lambda row: row["benchmark_id"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    benchmark_path = OUTPUT_ROOT / "benchmark.jsonl"
    with benchmark_path.open("w", encoding="utf-8") as stream:
        for row in benchmark:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "benchmark_id": "math_sft_validation_diagnostic_v0.1",
        "status": "diagnostic_only_not_final_test",
        "source": str(SOURCE_PATH),
        "selection": "up to 25 stable IDs per family and difficulty, then stable family fill to 100",
        "total": len(benchmark),
        "families": dict(sorted(Counter(row["family"] for row in benchmark).items())),
        "difficulties": dict(sorted(Counter(str(row["difficulty"]) for row in benchmark).items())),
        "benchmark_sha256": sha256(benchmark_path),
    }
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
