#!/usr/bin/env python3
"""Construit le curriculum v0.3.2 consacré aux tables de multiplication.

Objectif scientifique : séparer deux capacités différentes.

* rappel : apprendre explicitement les tables de 2 à 12, comme un humain les
  mémorise, avec des formulations variées et des indices de calcul ;
* généralisation : mesurer sans l'entraîner la multiplication de 13 à 20.

Le diagnostic de généralisation n'est ni un split d'entraînement ni un test
scellé. Aucun fichier ``test.jsonl`` n'est créé.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


DATASET_ID = "math_curriculum_v0.3.2_tables"
SCHEMA_VERSION = "ivoireslm.math-curriculum.v032.tables"
SEED = 20260902


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(
    *,
    split: str,
    family: str,
    index: int,
    prompt: str,
    target: int,
    left: int,
    right: int,
    operation: str,
    template_id: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "example_id": f"v032_{split}_{family}_{index:07d}",
        "task_family": family,
        "difficulty": 1 if max(left, right) <= 12 else 2,
        "prompt": prompt,
        "target": f" {target}\n",
        "verification": {
            "operation": operation,
            "left": left,
            "right": right,
            "result": target,
            "template_id": template_id,
        },
        "source": "deterministic_verified_arithmetic_v032",
        "license": "CC0-1.0",
        "split": split,
    }


def repeated_addition(left: int, right: int) -> str:
    return " + ".join([str(left)] * right)


def decomposition_hint(left: int, right: int) -> str:
    if right < 10:
        delta = 10 - right
        return f"{left} × {right} = {left} × 10 − {left} × {delta}"
    if right == 10:
        return f"{left} × 10"
    delta = right - 10
    return f"{left} × {right} = {left} × 10 + {left} × {delta}"


def build_rows() -> dict[str, list[dict]]:
    rows = {"train": [], "validation": [], "diagnostic_generalization": []}
    train_index = validation_index = diagnostic_index = 0

    # Révision d'addition : mêmes faits, formulations distinctes entre train et
    # validation. C'est une mesure de rappel, explicitement déclarée comme telle.
    for left in range(21):
        for right in range(21):
            result = left + right
            train_prompts = (
                ("add_train_calcul", f"Calcul : {left} + {right}\nRéponse exacte :"),
                ("add_train_somme", f"Donne uniquement la somme de {left} et {right} :"),
            )
            for template_id, prompt in train_prompts:
                rows["train"].append(
                    row(
                        split="train",
                        family="addition_review",
                        index=train_index,
                        prompt=prompt,
                        target=result,
                        left=left,
                        right=right,
                        operation="addition",
                        template_id=template_id,
                    )
                )
                train_index += 1
            rows["validation"].append(
                row(
                    split="validation",
                    family="addition_review",
                    index=validation_index,
                    prompt=f"Combien font {left} plus {right} ? Réponds par le nombre :",
                    target=result,
                    left=left,
                    right=right,
                    operation="addition",
                    template_id="add_validation_combien",
                )
            )
            validation_index += 1

    # Tables 2–12 : tous les faits sont enseignés. Les prompts de validation
    # sont nouveaux afin de mesurer le rappel sous une formulation différente.
    for left in range(2, 13):
        for right in range(2, 13):
            result = left * right
            direct_prompts = (
                ("mul_direct_calcul", f"Calcul : {left} × {right}\nRéponse exacte :"),
                ("mul_direct_produit", f"Donne uniquement le produit de {left} par {right} :"),
                ("mul_direct_table", f"Dans la table de {left}, combien vaut {left} × {right} ? Réponse :"),
                ("mul_direct_fois", f"Calcule {left} fois {right}. Nombre exact :"),
            )
            for template_id, prompt in direct_prompts:
                rows["train"].append(
                    row(
                        split="train",
                        family="multiplication_table_direct",
                        index=train_index,
                        prompt=prompt,
                        target=result,
                        left=left,
                        right=right,
                        operation="multiplication",
                        template_id=template_id,
                    )
                )
                train_index += 1

            repeated = repeated_addition(left, right)
            for template_id, prompt in (
                (
                    "mul_repeat_explicit",
                    f"Addition répétée : {left} × {right} = {repeated}. Résultat exact :",
                ),
                (
                    "mul_repeat_guided",
                    f"Calcule {left} × {right} en additionnant {left}, {right} fois. Réponse :",
                ),
            ):
                rows["train"].append(
                    row(
                        split="train",
                        family="multiplication_repeated_addition",
                        index=train_index,
                        prompt=prompt,
                        target=result,
                        left=left,
                        right=right,
                        operation="multiplication",
                        template_id=template_id,
                    )
                )
                train_index += 1

            hint = decomposition_hint(left, right)
            for template_id, prompt in (
                (
                    "mul_decompose_ten",
                    f"Calcule {left} × {right}. Indice : {hint}. Réponse exacte :",
                ),
                (
                    "mul_decompose_verify",
                    f"Utilise la décomposition « {hint} » puis donne seulement le résultat :",
                ),
            ):
                rows["train"].append(
                    row(
                        split="train",
                        family="multiplication_decomposition",
                        index=train_index,
                        prompt=prompt,
                        target=result,
                        left=left,
                        right=right,
                        operation="multiplication",
                        template_id=template_id,
                    )
                )
                train_index += 1

            rows["validation"].append(
                row(
                    split="validation",
                    family="multiplication_table_recall",
                    index=validation_index,
                    prompt=f"Sans explication, combien font {left} multiplié par {right} ?",
                    target=result,
                    left=left,
                    right=right,
                    operation="multiplication",
                    template_id="mul_validation_recall",
                )
            )
            validation_index += 1

    # Diagnostic hors tables : jamais utilisé pour le gradient ou la sélection.
    for left in range(13, 21):
        for right in range(13, 21):
            rows["diagnostic_generalization"].append(
                row(
                    split="diagnostic_generalization",
                    family="multiplication_unseen_generalization",
                    index=diagnostic_index,
                    prompt=f"Calcul inédit : {left} × {right}\nRéponse exacte :",
                    target=left * right,
                    left=left,
                    right=right,
                    operation="multiplication",
                    template_id="mul_diagnostic_unseen_13_20",
                )
            )
            diagnostic_index += 1

    for split, split_rows in rows.items():
        random.Random(f"{SEED}:{split}").shuffle(split_rows)
    return rows


def normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def validate(rows_by_split: dict[str, list[dict]]) -> None:
    expected = {
        "train": {
            "addition_review",
            "multiplication_table_direct",
            "multiplication_repeated_addition",
            "multiplication_decomposition",
        },
        "validation": {"addition_review", "multiplication_table_recall"},
        "diagnostic_generalization": {"multiplication_unseen_generalization"},
    }
    identifiers, prompts = set(), {}
    facts = {}
    for split, split_rows in rows_by_split.items():
        families = {item["task_family"] for item in split_rows}
        if families != expected[split]:
            raise RuntimeError(f"familles {split} invalides : {sorted(families)}")
        prompts[split], facts[split] = set(), set()
        for item in split_rows:
            if item["split"] != split or item["example_id"] in identifiers:
                raise RuntimeError(item["example_id"])
            identifiers.add(item["example_id"])
            values = item["verification"]
            result = (
                values["left"] + values["right"]
                if values["operation"] == "addition"
                else values["left"] * values["right"]
            )
            if result != values["result"] or item["target"].strip() != str(result):
                raise RuntimeError(f"résultat invalide : {item['example_id']}")
            prompt_key = normalized_prompt(item["prompt"])
            if prompt_key in prompts[split]:
                raise RuntimeError(f"prompt dupliqué : {item['example_id']}")
            prompts[split].add(prompt_key)
            facts[split].add(
                (values["operation"], values["left"], values["right"])
            )
    if prompts["train"] & prompts["validation"]:
        raise RuntimeError("fuite exacte de prompts train/validation")
    if prompts["train"] & prompts["diagnostic_generalization"]:
        raise RuntimeError("fuite de prompts vers le diagnostic")
    train_multiplications = {fact for fact in facts["train"] if fact[0] == "multiplication"}
    diagnostic_facts = facts["diagnostic_generalization"]
    if train_multiplications & diagnostic_facts:
        raise RuntimeError("fait de diagnostic vu pendant l'entraînement")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for item in rows:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Le dossier existe déjà ; aucune donnée ne sera écrasée : {args.output_dir}"
        )
    rows_by_split = build_rows()
    validate(rows_by_split)
    args.output_dir.mkdir(parents=True)
    for split, split_rows in rows_by_split.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", split_rows)
    report = {
        "dataset_id": DATASET_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "tables_train_validation_and_unseen_diagnostic_no_test",
        "seed": SEED,
        "scientific_goal": {
            "primary": "memorization_and_recall_of_multiplication_tables_2_to_12",
            "secondary": "preserve_easy_addition_and_general_language",
            "diagnostic_only": "unseen_multiplication_13_to_20",
        },
        "known_design_choice": (
            "multiplication facts 2-12 intentionally occur in train and validation; "
            "validation uses unseen prompt templates and measures table recall, not "
            "algorithmic generalization"
        ),
        "selection_thresholds": {
            "multiplication_table_recall_exact_rate": 0.60,
            "addition_review_exact_rate": 0.20,
            "maximum_general_language_loss_increase": 0.02,
        },
        "splits": {},
        "test_created": False,
    }
    for split, split_rows in rows_by_split.items():
        path = args.output_dir / f"{split}.jsonl"
        report["splits"][split] = {
            "examples": len(split_rows),
            "families": dict(Counter(item["task_family"] for item in split_rows)),
            "sha256": sha256(path),
            "used_for_model_selection": split == "validation",
            "used_for_gradient": split == "train",
        }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Curriculum mathématique v0.3.2 tables construit ✅")
    print("Diagnostic 13–20 isolé du gradient et de la sélection ✅")
    print("Aucun split test créé ✅")


if __name__ == "__main__":
    main()
