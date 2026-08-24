#!/usr/bin/env python3
"""Évalue le solveur déterministe sur un benchmark de développement JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from evaluation.math_benchmark import score_completion
from tools.math_solver import UnsupportedMathProblem, solve_math_problem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarized(counts_by_key: dict[str, Counter]) -> dict:
    result = {}
    for key, counts in sorted(counts_by_key.items()):
        total = counts["total"]
        result[key] = {
            **dict(counts),
            "format_rate": counts["formatted"] / total,
            "exact_accuracy": counts["correct"] / total,
        }
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    # Vérifier le nom avant toute ouverture : le benchmark final reste aveugle.
    if "final" in args.benchmark.parent.name.lower():
        raise RuntimeError("le benchmark final gelé ne doit pas être ouvert pendant le développement")
    benchmark = read_jsonl(args.benchmark)
    if args.limit is not None:
        benchmark = benchmark[: args.limit]
    if not benchmark:
        raise ValueError("benchmark vide")
    started = time.monotonic()
    predictions = []
    family_counts: dict[str, Counter] = defaultdict(Counter)
    difficulty_counts: dict[str, Counter] = defaultdict(Counter)
    for record in benchmark:
        error = None
        try:
            solution = solve_math_problem(record["problem"])
            completion = solution.completion
            tool_family = solution.family
        except UnsupportedMathProblem as exc:
            completion = ""
            tool_family = None
            error = str(exc)
        score = score_completion(record, completion)
        row = {
            "benchmark_id": record["benchmark_id"],
            "family": record["family"],
            "difficulty": record.get("difficulty"),
            "problem": record["problem"],
            "tool_family": tool_family,
            "completion": completion,
            "error": error,
            **score,
        }
        predictions.append(row)
        family_counts[record["family"]]["total"] += 1
        family_counts[record["family"]]["formatted"] += int(score["formatted"])
        family_counts[record["family"]]["correct"] += int(score["correct"])
        if record.get("difficulty") is not None:
            key = str(record["difficulty"])
            difficulty_counts[key]["total"] += 1
            difficulty_counts[key]["formatted"] += int(score["formatted"])
            difficulty_counts[key]["correct"] += int(score["correct"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    total = len(predictions)
    formatted = sum(row["formatted"] for row in predictions)
    correct = sum(row["correct"] for row in predictions)
    errors = sum(row["error"] is not None for row in predictions)
    report = {
        "evaluation_id": f"deterministic_math_tool_v0.1_{args.benchmark.parent.name}",
        "system_id": "deterministic_math_tool_v0.1",
        "benchmark_sha256": sha256(args.benchmark),
        "input_contract": "problem_text_only",
        "total": total,
        "formatted_answers": formatted,
        "correct_answers": correct,
        "solver_errors": errors,
        "format_rate": formatted / total,
        "exact_accuracy": correct / total,
        "families": summarized(family_counts),
        "difficulties": summarized(difficulty_counts),
        "predictions_sha256": sha256(predictions_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if correct != total:
        raise RuntimeError(f"le solveur a échoué sur {total - correct}/{total} exercices")
    print("Évaluation du moteur mathématique terminée : 100 % exact ✅")


if __name__ == "__main__":
    main()
