#!/usr/bin/env python3
"""Construit le curriculum train/validation du micro-assistant 17M v1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.assistant_curriculum_v1 import (
    CORE,
    CORE_VALIDATION,
    FACTS,
    TRANSFORM_SENTENCES,
    UNCERTAINTY,
    UNCERTAINTY_VALIDATION,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(example_id: str, family: str, subfamily: str, question: str, answer: str, split: str, **metadata) -> dict:
    return {
        "example_id": example_id,
        "task_family": family,
        "task_subfamily": subfamily,
        "prompt": f"Utilisateur : {question}\nAssistant :",
        "target": f" {answer.strip()}\n",
        "split": split,
        "license": "CC0-1.0",
        "source": "ivoireslm_curated_assistant_v1",
        **metadata,
    }


def build_rows(split: str) -> list[dict]:
    if split not in {"train", "validation"}:
        raise KeyError(split)
    rows = []
    core = CORE if split == "train" else CORE_VALIDATION
    uncertainty = UNCERTAINTY if split == "train" else UNCERTAINTY_VALIDATION
    prefixes = ("", "Réponds brièvement : ", "J’ai une question : ") if split == "train" else ("",)
    for index, (question, answer, intent) in enumerate(core):
        for variant, prefix in enumerate(prefixes):
            rows.append(row(f"{split}_core_{index}_{variant}", "assistant_core", intent, prefix + question, answer, split))
    for index, (question, answer) in enumerate(uncertainty):
        for variant, prefix in enumerate(prefixes):
            rows.append(row(f"{split}_uncertainty_{index}_{variant}", "uncertainty_refusal", "safe_uncertainty", prefix + question, answer, split))
    question_key = "train_questions" if split == "train" else "validation_questions"
    for fact in FACTS:
        for index, question in enumerate(fact[question_key]):
            for variant, prefix in enumerate(prefixes):
                rows.append(row(f"{split}_fact_{fact['id']}_{index}_{variant}", "ivoire_grounded", fact["id"], prefix + question, fact["answer"], split, source_url=fact["source_url"], source="official_fact_reworded_cc0"))
                context_question = f"{prefix}Contexte : {fact['answer']} Question : {question}"
                rows.append(row(f"{split}_reading_{fact['id']}_{index}_{variant}", "reading_comprehension", fact["id"], context_question, fact["answer"], split, source_url=fact["source_url"], source="official_fact_reworded_cc0"))
    selected_sentences = TRANSFORM_SENTENCES[:4] if split == "train" else TRANSFORM_SENTENCES[4:]
    for index, text in enumerate(selected_sentences):
        transformations = (
            ("uppercase", f"Écris en majuscules : {text}", text.upper()),
            ("lowercase", f"Écris en minuscules : {text}", text.lower()),
            ("copy", f"Recopie exactement : {text}", text),
        )
        for name, question, answer in transformations:
            for variant, prefix in enumerate(prefixes):
                rows.append(row(f"{split}_instruction_{index}_{name}_{variant}", "instruction_following", name, prefix + question, answer, split))
    return rows


def normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def validate(rows_by_split: dict[str, list[dict]]) -> None:
    expected = {"assistant_core", "uncertainty_refusal", "ivoire_grounded", "reading_comprehension", "instruction_following"}
    prompts = {}
    identifiers = set()
    for split, rows in rows_by_split.items():
        if {item["task_family"] for item in rows} != expected:
            raise RuntimeError(f"familles incomplètes dans {split}")
        prompts[split] = set()
        for item in rows:
            if item["example_id"] in identifiers or item["split"] != split:
                raise RuntimeError(item["example_id"])
            identifiers.add(item["example_id"])
            prompts[split].add(normalized_prompt(item["prompt"]))
    overlap = prompts["train"] & prompts["validation"]
    if overlap:
        raise RuntimeError(f"fuite exacte train/validation : {len(overlap)}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for item in rows:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("le dossier de sortie doit être vide")
    output.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: build_rows(split) for split in ("train", "validation")}
    validate(rows_by_split)
    for split, rows in rows_by_split.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    report = {
        "dataset_id": "assistant_curriculum_v1_17m",
        "status": "train_validation_only_test_not_created",
        "seed": 20260902,
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        path = output / f"{split}.jsonl"
        report["splits"][split] = {
            "examples": len(rows),
            "task_families": dict(sorted(Counter(item["task_family"] for item in rows).items())),
            "sha256": sha256(path),
        }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Curriculum assistant 17M v1 construit ; aucun test créé ✅")


if __name__ == "__main__":
    main()
