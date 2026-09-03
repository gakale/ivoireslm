#!/usr/bin/env python3
"""Compare plusieurs checkpoints sur un benchmark humain figé, sans entraînement."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inference.transformer17m_runtime import Transformer17MRuntime


CANNED_MARKERS = (
    "je suis ivoireslm",
    "mes connaissances sont limitées",
    "je demande une précision",
    "je ne dois pas inventer de source",
    "bonjour comment puis je t aider",
    "a proclamé son indépendance le 7 août 1960",
)


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœ']+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("format attendu : nom=/chemin/checkpoint.pt")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("nom de checkpoint vide")
    return name.strip(), Path(raw_path)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--checkpoint", type=checkpoint_argument, action="append", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_question = {}
    for row in rows:
        question_key = normalize(row["question"])
        if not question_key:
            continue
        by_question[question_key] = row
    benchmark = []
    for index, row in enumerate(by_question.values(), start=1):
        expected = row.get("human_correction")
        if not expected and row.get("rating") == "Correcte":
            expected = row.get("model_response")
        benchmark.append({
            "benchmark_id": f"human-{index:04d}",
            "question": row["question"].strip(),
            "category": row.get("category", "non_classee"),
            "human_expected_answer": expected,
            "original_rating": row.get("rating"),
        })
    if not benchmark:
        raise RuntimeError("benchmark humain vide")
    return benchmark


def summarize(predictions: list[dict]) -> dict:
    normalized_outputs = [normalize(item["prediction"]) for item in predictions]
    counts = Counter(normalized_outputs)
    exact_eligible = [item for item in predictions if item["human_expected_answer"]]
    exact = sum(
        normalize(item["prediction"]) == normalize(item["human_expected_answer"])
        for item in exact_eligible
    )
    canned = sum(
        any(marker in output for marker in CANNED_MARKERS)
        for output in normalized_outputs
    )
    return {
        "examples": len(predictions),
        "unique_outputs": len(counts),
        "unique_output_rate": len(counts) / len(predictions),
        "most_common_output_count": counts.most_common(1)[0][1],
        "most_common_output_rate": counts.most_common(1)[0][1] / len(predictions),
        "canned_marker_count": canned,
        "canned_marker_rate": canned / len(predictions),
        "stopped_on_eos_rate": sum(item["stopped_on_eos"] for item in predictions) / len(predictions),
        "mean_generated_tokens": sum(item["generated_tokens"] for item in predictions) / len(predictions),
        "exact_eligible": len(exact_eligible),
        "exact_matches_to_human_answer": exact,
        "exact_match_rate": exact / len(exact_eligible) if exact_eligible else None,
        "warning": "exact match is diagnostic only; final correctness requires human review",
    }


def main():
    args = arguments()
    if len({name for name, _ in args.checkpoint}) != len(args.checkpoint):
        raise RuntimeError("noms de checkpoints dupliqués")
    benchmark = load_benchmark(args.feedback)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / "human_benchmark_frozen.jsonl"
    frozen_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in benchmark),
        encoding="utf-8",
    )
    report = {
        "evaluation_id": "ivoireslm_17m_human_checkpoint_comparison_v1",
        "source_feedback_sha256": sha256(args.feedback),
        "frozen_benchmark_sha256": sha256(frozen_path),
        "raw_feedback_rows": sum(1 for line in args.feedback.read_text(encoding="utf-8").splitlines() if line),
        "unique_questions": len(benchmark),
        "training_performed": False,
        "checkpoints": {},
    }
    for name, checkpoint in args.checkpoint:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"\nChargement {name} : {checkpoint}", flush=True)
        runtime = Transformer17MRuntime(checkpoint, args.data_dir, args.model_script)
        predictions = []
        for index, item in enumerate(benchmark, start=1):
            generated = runtime.generate(item["question"], max_new_tokens=args.max_new_tokens)
            predictions.append({
                **item,
                "checkpoint_name": name,
                "checkpoint_step": runtime.checkpoint_step,
                "model_id": runtime.model_id,
                "prediction": generated.text,
                "generated_tokens": generated.generated_tokens,
                "stopped_on_eos": generated.stopped_on_eos,
            })
            print(f"{name} {index:02d}/{len(benchmark)}", flush=True)
        prediction_path = args.output_dir / f"predictions_{name}.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions),
            encoding="utf-8",
        )
        report["checkpoints"][name] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_step": runtime.checkpoint_step,
            "model_id": runtime.model_id,
            "predictions_sha256": sha256(prediction_path),
            **summarize(predictions),
        }
        del runtime
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    report_path = args.output_dir / "checkpoint_comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    print("\nComparaison terminée ; aucun entraînement effectué ✅")


if __name__ == "__main__":
    main()
