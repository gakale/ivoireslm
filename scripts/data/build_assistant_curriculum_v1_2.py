#!/usr/bin/env python3
"""Construit le stage 3 contrastif du curriculum micro-assistant 17M."""

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
from data.assistant_curriculum_v1_1 import CITIES, NAMES, PREDICATES, SUBJECTS
from data.assistant_curriculum_v1_2 import CORE_GROUPS, UNCERTAINTY_GROUPS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(identifier, family, subfamily, prompt, target, split, **extra):
    return {
        "example_id": identifier,
        "task_family": family,
        "task_subfamily": subfamily,
        "prompt": f"Utilisateur : {prompt}\nAssistant :",
        "target": f" {target.strip()}\n",
        "split": split,
        "license": "CC0-1.0",
        "source": "ivoireslm_curated_assistant_v1_2",
        **extra,
    }


def core_rows(split):
    prefixes = ("", "Réponds précisément : ") if split == "train" else ("",)
    rows = []
    for intent, group in CORE_GROUPS.items():
        for index, prompt in enumerate(group[split]):
            for variant, prefix in enumerate(prefixes):
                rows.append(row(f"{split}_core_{intent}_{index}_{variant}", "assistant_core", intent, prefix + prompt, group["answer"], split))
    return rows


def uncertainty_rows(split):
    prefixes = ("", "Réponds avec prudence : ") if split == "train" else ("",)
    rows = []
    for group_index, group in enumerate(UNCERTAINTY_GROUPS):
        for index, prompt in enumerate(group[split]):
            for variant, prefix in enumerate(prefixes):
                rows.append(row(f"{split}_uncertainty_{group_index}_{index}_{variant}", "uncertainty_refusal", f"rule_{group_index}", prefix + prompt, group["answer"], split))
    return rows


def instruction_rows(split):
    rows = []
    for subject_index, subject in enumerate(SUBJECTS):
        for predicate_index, predicate in enumerate(PREDICATES):
            sentence = f"{subject} {predicate}"
            assigned = "validation" if (subject_index * 7 + predicate_index * 11) % 5 == 0 else "train"
            if assigned != split:
                continue
            operations = (
                ("copy", ("Recopie exactement :", "Répète sans modifier :"), sentence),
                ("lowercase", ("Écris en minuscules :", "Transforme en lettres minuscules :"), sentence.lower()),
                ("uppercase", ("Écris en majuscules :", "Transforme en lettres majuscules :", "Mets toute cette phrase en majuscules :"), sentence.upper()),
            )
            for operation, templates, target in operations:
                selected = templates if split == "train" else templates[:1]
                for variant, template in enumerate(selected):
                    rows.append(row(f"{split}_instruction_{subject_index}_{predicate_index}_{operation}_{variant}", "instruction_following", operation, f"{template} {sentence}", target, split))
    return rows


def reading_rows(split):
    rows = []
    for name_index, name in enumerate(NAMES):
        for city_index, city in enumerate(CITIES):
            assigned = "validation" if (name_index * 3 + city_index * 2) % 5 == 0 else "train"
            if assigned != split:
                continue
            context = f"{name} habite à {city}."
            rows.append(row(f"{split}_reading_{name_index}_{city_index}", "reading_comprehension", "location", f"Contexte : {context} Question : Où habite {name} ?", context, split))
    return rows


def fact_rows(split):
    rows = []
    key = "train_questions" if split == "train" else "validation_questions"
    prefixes = ("", "Réponds en une phrase : ") if split == "train" else ("",)
    for fact in FACTS:
        for index, question in enumerate(fact[key]):
            for variant, prefix in enumerate(prefixes):
                rows.append(row(f"{split}_fact_{fact['id']}_{index}_{variant}", "ivoire_grounded", fact["id"], prefix + question, fact["answer"], split, source_url=fact["source_url"], source="official_fact_reworded_cc0"))
    return rows


def build(split):
    return core_rows(split) + uncertainty_rows(split) + instruction_rows(split) + reading_rows(split) + fact_rows(split)


def normalized(value):
    return " ".join(value.casefold().split())


def validate(splits):
    expected = {"assistant_core", "uncertainty_refusal", "ivoire_grounded", "reading_comprehension", "instruction_following"}
    prompts = {}
    identifiers = set()
    for split, rows in splits.items():
        if {item["task_family"] for item in rows} != expected:
            raise RuntimeError(f"familles incomplètes : {split}")
        prompts[split] = set()
        for item in rows:
            if item["example_id"] in identifiers:
                raise RuntimeError(f"identifiant dupliqué : {item['example_id']}")
            identifiers.add(item["example_id"])
            prompt = normalized(item["prompt"])
            if prompt in prompts[split]:
                raise RuntimeError(f"prompt dupliqué : {item['example_id']}")
            prompts[split].add(prompt)
    if prompts["train"] & prompts["validation"]:
        raise RuntimeError("fuite train/validation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("le dossier de sortie doit être vide")
    output.mkdir(parents=True, exist_ok=True)
    splits = {name: build(name) for name in ("train", "validation")}
    validate(splits)
    report = {"dataset_id": "assistant_curriculum_v1_2_stage3", "status": "train_validation_only_test_not_created", "parent_stage": "assistant_v1_1_stage2_step500", "splits": {}}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows), encoding="utf-8")
        report["splits"][split] = {"examples": len(rows), "task_families": dict(sorted(Counter(item["task_family"] for item in rows).items())), "sha256": sha256(path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Curriculum assistant v1.2 stage 3 construit ; aucun test créé ✅")


if __name__ == "__main__":
    main()
