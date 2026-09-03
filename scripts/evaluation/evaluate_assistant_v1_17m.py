#!/usr/bin/env python3
"""Évalue les générations du micro-assistant 17M avant tout test public."""

from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inference.transformer17m_runtime import Transformer17MRuntime


THRESHOLDS = {
    "assistant_core": 0.80,
    "uncertainty_refusal": 0.80,
    "ivoire_grounded": 0.70,
    "reading_comprehension": 0.70,
    "instruction_following": 0.70,
}
EXACT_FAMILIES = {
    "ivoire_grounded",
    "reading_comprehension",
    "instruction_following",
}


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœ']+", " ", text).strip()


def non_repetitive(text: str) -> bool:
    words = normalize(text).split()
    if len(words) < 8:
        return True
    trigrams = list(zip(words, words[1:], words[2:]))
    return len(set(trigrams)) / len(trigrams) >= 0.70


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-acceptable", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def evaluate_general_language(runtime, data_dir: Path, *, windows: int = 128) -> float:
    """Mesure la loss du checkpoint évalué, pas celle d'un autre palier."""
    import numpy as np

    tokens = np.memmap(data_dir / "validation.uint16.bin", mode="r", dtype=np.uint16)
    block_size = runtime.model.config.block_size
    maximum_start = len(tokens) - block_size - 1
    if maximum_start < 0:
        raise RuntimeError("validation générale trop courte")
    starts = np.linspace(0, maximum_start, min(windows, maximum_start + 1), dtype=int)
    losses = []
    runtime.model.eval()
    with runtime.torch.inference_mode():
        for start in starts:
            inputs = runtime.torch.tensor(
                np.asarray(tokens[start : start + block_size], dtype=np.int64),
                dtype=runtime.torch.long,
                device=runtime.device,
            ).unsqueeze(0)
            targets = runtime.torch.tensor(
                np.asarray(tokens[start + 1 : start + block_size + 1], dtype=np.int64),
                dtype=runtime.torch.long,
                device=runtime.device,
            ).unsqueeze(0)
            _, loss = runtime.model(inputs, targets)
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def main() -> None:
    args = arguments()
    report = json.loads((args.dataset_dir / "report.json").read_text(encoding="utf-8"))
    validation_path = args.dataset_dir / "validation.jsonl"
    if sha256(validation_path) != report["splits"]["validation"]["sha256"]:
        raise RuntimeError("empreinte de validation invalide")
    rows = [json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != report["splits"]["validation"]["examples"]:
        raise RuntimeError("taille de validation invalide")
    runtime = Transformer17MRuntime(args.checkpoint, args.data_dir, args.model_script)
    totals = defaultdict(lambda: {"examples": 0, "acceptable": 0, "exact": 0, "similarity": 0.0})
    predictions, clean_outputs = [], 0
    for index, row in enumerate(rows, start=1):
        generation = runtime.generate(row["prompt"], max_new_tokens=args.max_new_tokens)
        expected, predicted = normalize(row["target"]), normalize(generation.text)
        exact = expected == predicted
        similarity = SequenceMatcher(None, expected, predicted).ratio()
        acceptable = exact if row["task_family"] in EXACT_FAMILIES else similarity >= 0.70
        clean = bool(predicted) and generation.stopped_on_eos and non_repetitive(generation.text)
        family = row["task_family"]
        totals[family]["examples"] += 1
        totals[family]["acceptable"] += int(acceptable)
        totals[family]["exact"] += int(exact)
        totals[family]["similarity"] += similarity
        clean_outputs += int(clean)
        predictions.append({
            "example_id": row["example_id"],
            "task_family": family,
            "expected": row["target"].strip(),
            "prediction": generation.text,
            "exact": exact,
            "similarity": similarity,
            "acceptable": acceptable,
            "stopped_on_eos": generation.stopped_on_eos,
            "non_repetitive": non_repetitive(generation.text),
        })
        print(f"{index:02d}/{len(rows)} {family} {'✅' if acceptable else '❌'}", flush=True)
    families = {}
    for family, values in sorted(totals.items()):
        count = values["examples"]
        families[family] = {
            "examples": count,
            "acceptable": values["acceptable"],
            "acceptable_rate": values["acceptable"] / count,
            "exact_rate": values["exact"] / count,
            "mean_similarity": values["similarity"] / count,
            "threshold": THRESHOLDS[family],
            "passed": values["acceptable"] / count >= THRESHOLDS[family],
        }
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    baseline_loss = float(progress["baseline_validation"]["general_language"]["loss_nats"])
    current_loss = evaluate_general_language(runtime, args.data_dir)
    language_delta = current_loss - baseline_loss
    clean_rate = clean_outputs / len(rows)
    automatic_passed = (
        clean_rate >= 0.90
        and language_delta <= 0.03
        and all(values["passed"] for values in families.values())
    )
    human_passed = args.human_acceptable is not None and args.human_acceptable >= 24
    result = {
        "evaluation_id": "microivoire_transformer_v1.1_17m_assistant_qualification_v1",
        "model_id": runtime.model_id,
        "checkpoint_step": runtime.checkpoint_step,
        "checkpoint_sha256": sha256(args.checkpoint),
        "validation_sha256": sha256(validation_path),
        "decoding": "deterministic_greedy",
        "families": families,
        "clean_terminated_nonrepetitive_rate": clean_rate,
        "clean_output_threshold": 0.90,
        "general_language_baseline_loss": baseline_loss,
        "general_language_current_loss": current_loss,
        "general_language_loss_delta": language_delta,
        "general_language_guard_passed": language_delta <= 0.03,
        "automatic_gates_passed": automatic_passed,
        "ready_for_human_test": automatic_passed,
        "human_acceptable_out_of_30": args.human_acceptable,
        "human_gate_passed": human_passed,
        "publication_qualified": automatic_passed and human_passed,
        "sealed_test_opened": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "assistant_qualification_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    result["predictions_sha256"] = sha256(predictions_path)
    result_path = args.output_dir / "assistant_qualification.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Qualification automatique terminée ; test humain requis avant publication ✅")


if __name__ == "__main__":
    main()
