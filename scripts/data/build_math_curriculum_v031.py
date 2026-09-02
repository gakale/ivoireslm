#!/usr/bin/env python3
"""Construit le premier palier mathématique progressif du Transformer 17M.

Cette branche isole deux compétences accessibles au petit modèle :

* additions dont les deux termes sont compris entre 0 et 20 ;
* tables de multiplication, avec facteurs compris entre 2 et 12.

Les couples d'opérandes sont disjoints entre train et validation. Le test n'est
volontairement pas créé à ce stade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


DATASET_ID = "math_curriculum_v0.3.1_stage1"
SCHEMA_VERSION = "ivoireslm.math-curriculum.v031.stage1"
SEED = 20260902
TRAIN_FRACTION = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_shuffle(values: list[tuple[int, int]], salt: str) -> list[tuple[int, int]]:
    ordered = list(values)
    random.Random(f"{SEED}:{salt}").shuffle(ordered)
    return ordered


def split_commutative_pairs(
    pairs: list[tuple[int, int]], salt: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Sépare les couples canoniques puis ajoute leur ordre inverse au même split."""
    ordered = stable_shuffle(pairs, salt)
    boundary = int(len(ordered) * TRAIN_FRACTION)

    def expand(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
        output = []
        for left, right in values:
            output.append((left, right))
            if left != right:
                output.append((right, left))
        return output

    return expand(ordered[:boundary]), expand(ordered[boundary:])


def addition_row(split: str, index: int, operands: tuple[int, int]) -> dict:
    left, right = operands
    result = left + right
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "example_id": f"stage1_{split}_addition_{index:06d}",
        "task_family": "addition_easy",
        "difficulty": 1,
        "prompt": f"Calcul : {left} + {right}\nRéponse exacte :",
        "target": f" {result}\n",
        "verification": {"left": left, "right": right, "result": result},
        "source": "deterministic_verified_arithmetic_v031",
        "license": "CC0-1.0",
        "split": split,
    }


def multiplication_row(split: str, index: int, operands: tuple[int, int]) -> dict:
    left, right = operands
    result = left * right
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "example_id": f"stage1_{split}_multiplication_{index:06d}",
        "task_family": "multiplication_table",
        "difficulty": 1,
        "prompt": f"Calcul : {left} × {right}\nRéponse exacte :",
        "target": f" {result}\n",
        "verification": {"left": left, "right": right, "result": result},
        "source": "deterministic_verified_arithmetic_v031",
        "license": "CC0-1.0",
        "split": split,
    }


def build_rows() -> dict[str, list[dict]]:
    # Les couples canoniques sont séparés avant que leur ordre inverse soit ajouté.
    # Ainsi, 3+7 et 7+3 (ou 3×7 et 7×3) restent toujours dans le même split.
    addition_pairs = [
        (left, right)
        for left in range(21)
        for right in range(left, 21)
    ]
    multiplication_pairs = [
        (left, right)
        for left in range(2, 13)
        for right in range(left, 13)
    ]
    addition_train, addition_validation = split_commutative_pairs(
        addition_pairs, "addition"
    )
    multiplication_train, multiplication_validation = split_commutative_pairs(
        multiplication_pairs, "multiplication"
    )
    rows = {"train": [], "validation": []}
    for split, additions, multiplications in (
        ("train", addition_train, multiplication_train),
        ("validation", addition_validation, multiplication_validation),
    ):
        rows[split].extend(
            addition_row(split, index, operands)
            for index, operands in enumerate(additions)
        )
        rows[split].extend(
            multiplication_row(split, index, operands)
            for index, operands in enumerate(multiplications)
        )
        rows[split].sort(
            key=lambda row: hashlib.sha256(
                f"{SEED}:{split}:{row['example_id']}".encode("utf-8")
            ).digest()
        )
    return rows


def verify_row(row: dict) -> bool:
    values = row["verification"]
    if row["task_family"] == "addition_easy":
        return values["left"] + values["right"] == values["result"]
    if row["task_family"] == "multiplication_table":
        return values["left"] * values["right"] == values["result"]
    return False


def normalized_prompt(row: dict) -> str:
    return " ".join(row["prompt"].casefold().split())


def operand_key(row: dict) -> tuple[str, int, int]:
    values = row["verification"]
    left, right = sorted((values["left"], values["right"]))
    return row["task_family"], left, right


def ordered_operand_key(row: dict) -> tuple[str, int, int]:
    values = row["verification"]
    return row["task_family"], values["left"], values["right"]


def validate(rows_by_split: dict[str, list[dict]]) -> None:
    expected_families = {"addition_easy", "multiplication_table"}
    identifiers: set[str] = set()
    prompts: dict[str, set[str]] = {}
    operands: dict[str, set[tuple[str, int, int]]] = {}
    ordered_operands: dict[str, set[tuple[str, int, int]]] = {}
    for split, rows in rows_by_split.items():
        if {row["task_family"] for row in rows} != expected_families:
            raise RuntimeError(f"familles {split} invalides")
        prompts[split], operands[split], ordered_operands[split] = set(), set(), set()
        for row in rows:
            if row["split"] != split or row["example_id"] in identifiers:
                raise RuntimeError(row["example_id"])
            if not verify_row(row):
                raise RuntimeError(f"calcul invalide : {row['example_id']}")
            identifiers.add(row["example_id"])
            prompt = normalized_prompt(row)
            key = operand_key(row)
            ordered_key = ordered_operand_key(row)
            if prompt in prompts[split] or ordered_key in ordered_operands[split]:
                raise RuntimeError(f"doublon {split} : {row['example_id']}")
            prompts[split].add(prompt)
            operands[split].add(key)
            ordered_operands[split].add(ordered_key)
    if prompts["train"] & prompts["validation"]:
        raise RuntimeError("fuite de prompts train/validation")
    if operands["train"] & operands["validation"]:
        raise RuntimeError("fuite d'opérandes train/validation")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
    for split, rows in rows_by_split.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    report = {
        "dataset_id": DATASET_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "stage1_train_validation_only_test_not_created",
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "curriculum": {
            "stage": 1,
            "skills": ["addition_easy", "multiplication_table"],
            "excluded": [
                "translation",
                "grounded_wdi",
                "advanced_math",
                "sealed_test",
            ],
            "selection_thresholds": {
                "addition_exact_rate": 0.25,
                "multiplication_exact_rate": 0.15,
                "maximum_general_language_loss_increase": 0.02,
            },
        },
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        path = args.output_dir / f"{split}.jsonl"
        report["splits"][split] = {
            "examples": len(rows),
            "families": dict(Counter(row["task_family"] for row in rows)),
            "sha256": sha256(path),
        }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Curriculum mathématique v0.3.1 stage 1 construit ✅")
    print("Aucun split test créé ✅")


if __name__ == "__main__":
    main()
