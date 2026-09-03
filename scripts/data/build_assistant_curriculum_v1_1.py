#!/usr/bin/env python3
"""Construit le stage 2 diversifié du curriculum micro-assistant 17M."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.assistant_curriculum_v1 import FACTS
from data.assistant_curriculum_v1_1 import (
    CITIES, GREETING_ANSWER, GREETING_TRAIN, GREETING_VALIDATION,
    IDENTITY_ANSWER, IDENTITY_TRAIN, IDENTITY_VALIDATION,
    LIMITS_ANSWER, LIMITS_TRAIN, LIMITS_VALIDATION, NAMES, PREDICATES,
    SUBJECTS, UNCERTAINTY_GROUPS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_row(identifier, family, subfamily, prompt, target, split, **extra):
    return {
        "example_id": identifier,
        "task_family": family,
        "task_subfamily": subfamily,
        "prompt": f"Utilisateur : {prompt}\nAssistant :",
        "target": f" {target.strip()}\n",
        "split": split,
        "license": "CC0-1.0",
        "source": "ivoireslm_curated_assistant_v1_1",
        **extra,
    }


def core_rows(split: str) -> list[dict]:
    groups = (
        ("identity", IDENTITY_TRAIN if split == "train" else IDENTITY_VALIDATION, IDENTITY_ANSWER),
        ("greeting", GREETING_TRAIN if split == "train" else GREETING_VALIDATION, GREETING_ANSWER),
        ("limits", LIMITS_TRAIN if split == "train" else LIMITS_VALIDATION, LIMITS_ANSWER),
    )
    rows = []
    prefixes = ("", "Réponds brièvement : ", "J’ai une question : ") if split == "train" else ("",)
    for intent, prompts, answer in groups:
        for index, prompt in enumerate(prompts):
            for variant, prefix in enumerate(prefixes):
                rows.append(make_row(f"{split}_core_{intent}_{index}_{variant}", "assistant_core", intent, prefix + prompt, answer, split))
    return rows


def uncertainty_rows(split: str) -> list[dict]:
    rows = []
    prefixes = ("", "Réponds avec prudence : ", "Sans rien inventer : ") if split == "train" else ("",)
    for group_index, (answer, train_prompts, validation_prompts) in enumerate(UNCERTAINTY_GROUPS):
        prompts = train_prompts if split == "train" else validation_prompts
        for index, prompt in enumerate(prompts):
            for variant, prefix in enumerate(prefixes):
                rows.append(make_row(f"{split}_uncertainty_{group_index}_{index}_{variant}", "uncertainty_refusal", f"rule_{group_index}", prefix + prompt, answer, split))
    return rows


def instruction_and_reading_rows(split: str) -> list[dict]:
    sentences = [f"{subject} {predicate}" for subject in SUBJECTS for predicate in PREDICATES]
    selected = sentences[:90] if split == "train" else sentences[90:]
    rows = []
    for index, sentence in enumerate(selected):
        for operation, prompt, target in (
            ("copy", f"Recopie exactement : {sentence}", sentence),
            ("uppercase", f"Écris en majuscules : {sentence}", sentence.upper()),
            ("lowercase", f"Écris en minuscules : {sentence}", sentence.lower()),
        ):
            rows.append(make_row(f"{split}_instruction_{index}_{operation}", "instruction_following", operation, prompt, target, split))
    pairs = [(name, city) for name in NAMES for city in CITIES]
    selected_pairs = pairs[:48] if split == "train" else pairs[48:]
    for index, (name, city) in enumerate(selected_pairs):
        context = f"{name} habite à {city}."
        prompt = f"Contexte : {context} Question : Où habite {name} ?"
        rows.append(make_row(f"{split}_reading_{index}", "reading_comprehension", "location", prompt, context, split))
    return rows


def fact_rows(split: str) -> list[dict]:
    rows = []
    key = "train_questions" if split == "train" else "validation_questions"
    prefixes = ("", "Réponds en une phrase : ", "Question sur la Côte d’Ivoire : ") if split == "train" else ("",)
    for fact in FACTS:
        for index, question in enumerate(fact[key]):
            for variant, prefix in enumerate(prefixes):
                rows.append(make_row(f"{split}_fact_{fact['id']}_{index}_{variant}", "ivoire_grounded", fact["id"], prefix + question, fact["answer"], split, source_url=fact["source_url"], source="official_fact_reworded_cc0"))
    return rows


def build(split: str) -> list[dict]:
    return core_rows(split) + uncertainty_rows(split) + instruction_and_reading_rows(split) + fact_rows(split)


def normalize(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def validate(splits: dict[str, list[dict]]) -> None:
    expected = {"assistant_core", "uncertainty_refusal", "ivoire_grounded", "reading_comprehension", "instruction_following"}
    seen_ids = set()
    prompt_sets = {}
    for split, rows in splits.items():
        if {row["task_family"] for row in rows} != expected:
            raise RuntimeError(f"familles incomplètes : {split}")
        prompt_sets[split] = set()
        for row in rows:
            if row["example_id"] in seen_ids or row["split"] != split:
                raise RuntimeError(row["example_id"])
            seen_ids.add(row["example_id"])
            normalized = normalize(row["prompt"])
            if normalized in prompt_sets[split]:
                raise RuntimeError(f"doublon interne : {row['example_id']}")
            prompt_sets[split].add(normalized)
    if prompt_sets["train"] & prompt_sets["validation"]:
        raise RuntimeError("fuite train/validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("le dossier de sortie doit être vide")
    output.mkdir(parents=True, exist_ok=True)
    splits = {name: build(name) for name in ("train", "validation")}
    validate(splits)
    report = {"dataset_id": "assistant_curriculum_v1_1_stage2", "status": "train_validation_only_test_not_created", "parent_stage": "assistant_v1_step500", "splits": {}}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        report["splits"][split] = {"examples": len(rows), "task_families": dict(sorted(Counter(row["task_family"] for row in rows).items())), "sha256": sha256(path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Curriculum assistant v1.1 stage 2 construit ; aucun test créé ✅")


if __name__ == "__main__":
    main()
