#!/usr/bin/env python3
"""Évalue un checkpoint IvoireSLM sur le benchmark mathématique gelé."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from evaluation.math_benchmark import score_completion


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_module(path: Path):
    specification = importlib.util.spec_from_file_location("ivoireslm_transformer_v02", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"impossible d'importer le script modèle : {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_tokenizer(path: Path) -> tuple[list[str], dict[str, int]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    id_to_token = record["id_to_token"]
    return id_to_token, {token: index for index, token in enumerate(id_to_token)}


@torch.inference_mode()
def greedy_completion(
    model,
    prompt: str,
    id_to_token: list[str],
    token_to_id: dict[str, int],
    device: torch.device,
    max_new_tokens: int,
) -> str:
    unknown = token_to_id["<UNK>"]
    ids = [token_to_id.get(character, unknown) for character in prompt]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    completion = ""
    response_seen = False
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.block_size :]
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(context)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
        piece = id_to_token[int(next_token.item())]
        if piece not in {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}:
            completion += piece
        if "Réponse :" in completion:
            response_seen = True
        if response_seen and completion.endswith("\n"):
            break
    return completion


def upload(path: Path, gcs_output: str | None, remote_name: str | None = None) -> None:
    if not gcs_output:
        return
    name = remote_name or path.name
    destination = f"{gcs_output.rstrip('/')}/{name}"
    subprocess.run(["gcloud", "storage", "cp", str(path), destination], check=True)


def write_predictions(path: Path, predictions: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/math_evaluation_v01"))
    parser.add_argument("--gcs-output")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume-predictions", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent : cette évaluation doit être lancée sur le T4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = read_jsonl(args.benchmark)
    if args.limit:
        benchmark = benchmark[: args.limit]
    benchmark_by_id = {row["benchmark_id"]: row for row in benchmark}
    if len(benchmark_by_id) != len(benchmark):
        raise ValueError("identifiants du benchmark non uniques")

    training_module = load_training_module(args.model_script)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = training_module.TrainingConfig(**checkpoint["config"])
    model = training_module.MicroIvoireTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    model.to(device).eval()
    id_to_token, token_to_id = read_tokenizer(args.data_dir / "tokenizer.json")
    if len(id_to_token) != config.vocab_size:
        raise ValueError("tokenizer incompatible avec le checkpoint")

    predictions = []
    completed_ids = set()
    if args.resume_predictions and args.resume_predictions.is_file():
        predictions = read_jsonl(args.resume_predictions)
        completed_ids = {row["benchmark_id"] for row in predictions}
        unknown_ids = completed_ids - benchmark_by_id.keys()
        if unknown_ids:
            raise ValueError(f"prédictions étrangères au benchmark : {sorted(unknown_ids)[:3]}")
        print(f"Reprise de {len(predictions):,} prédictions existantes ✅", flush=True)

    partial_path = args.output_dir / "predictions.partial.jsonl"
    started = time.monotonic()
    for position, record in enumerate(benchmark, start=1):
        if record["benchmark_id"] in completed_ids:
            continue
        completion = greedy_completion(
            model,
            record["prompt"],
            id_to_token,
            token_to_id,
            device,
            args.max_new_tokens,
        )
        score = score_completion(record, completion)
        predictions.append(
            {
                "benchmark_id": record["benchmark_id"],
                "family": record["family"],
                "difficulty": record.get("difficulty"),
                "prompt": record["prompt"],
                "reference_answer": record["reference_answer"],
                "completion": completion,
                **score,
            }
        )
        if len(predictions) % 25 == 0:
            correct = sum(row["correct"] for row in predictions)
            print(
                f"{len(predictions):4d}/{len(benchmark)} | exact {correct / len(predictions):.2%} | "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
        if len(predictions) % 100 == 0:
            write_predictions(partial_path, predictions)
            upload(partial_path, args.gcs_output, "predictions.partial.jsonl")

    predictions.sort(key=lambda row: row["benchmark_id"])
    predictions_path = args.output_dir / "predictions.jsonl"
    write_predictions(predictions_path, predictions)

    family_counts = defaultdict(Counter)
    difficulty_counts = defaultdict(Counter)
    for row in predictions:
        family_counts[row["family"]]["total"] += 1
        family_counts[row["family"]]["formatted"] += int(row["formatted"])
        family_counts[row["family"]]["correct"] += int(row["correct"])
        if row.get("difficulty") is not None:
            key = str(row["difficulty"])
            difficulty_counts[key]["total"] += 1
            difficulty_counts[key]["formatted"] += int(row["formatted"])
            difficulty_counts[key]["correct"] += int(row["correct"])
    families = {}
    for family, counts in sorted(family_counts.items()):
        families[family] = {
            **dict(counts),
            "format_rate": counts["formatted"] / counts["total"],
            "exact_accuracy": counts["correct"] / counts["total"],
        }
    difficulties = {}
    for difficulty, counts in sorted(difficulty_counts.items()):
        difficulties[difficulty] = {
            **dict(counts),
            "format_rate": counts["formatted"] / counts["total"],
            "exact_accuracy": counts["correct"] / counts["total"],
        }

    total = len(predictions)
    formatted = sum(row["formatted"] for row in predictions)
    correct = sum(row["correct"] for row in predictions)
    report = {
        "evaluation_id": f"{config.model_id}_{args.benchmark.parent.name}",
        "model_id": config.model_id,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": sha256(args.checkpoint),
        "benchmark_sha256": sha256(args.benchmark),
        "decoding": "deterministic_greedy",
        "max_new_tokens": args.max_new_tokens,
        "total": total,
        "formatted_answers": formatted,
        "correct_answers": correct,
        "format_rate": formatted / total,
        "exact_accuracy": correct / total,
        "families": families,
        "difficulties": difficulties,
        "predictions_sha256": sha256(predictions_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    upload(predictions_path, args.gcs_output)
    upload(report_path, args.gcs_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Évaluation mathématique terminée et sauvegardée ✅")


if __name__ == "__main__":
    main()
