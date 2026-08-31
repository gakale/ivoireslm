#!/usr/bin/env python3
"""Construit le curriculum supervisé équilibré du Transformer IvoireSLM 17M."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.instruction_sft import core_rows, grounded_wdi_row


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
OUTPUT_ROOT = STORAGE_ROOT / "derived/instruction_sft_v0.2"
MATH_ROOT = STORAGE_ROOT / "derived/math_sft_v0.1"
PARALLEL_ROOT = STORAGE_ROOT / "derived/ivoirian_parallel_v0.1"
WDI_PATH = STORAGE_ROOT / "derived/structured_factual_v0.2/multidomain/civ_worldbank_wdi_1960_2025_v0.1.txt"
MATH_TRAIN_PER_FAMILY = 2_000
MATH_VALIDATION_PER_FAMILY = 150


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def math_rows(split: str) -> list[dict]:
    source = read_jsonl(MATH_ROOT / f"{split}.jsonl")
    limit = MATH_TRAIN_PER_FAMILY if split == "train" else MATH_VALIDATION_PER_FAMILY
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in source:
        groups[row["family"]].append(row)
    output = []
    for family, rows in sorted(groups.items()):
        for row in sorted(rows, key=lambda item: item["example_id"])[:limit]:
            output.append(
                {
                    "example_id": row["example_id"],
                    "task_family": "math_verified",
                    "task_subfamily": family,
                    "prompt": row["prompt"],
                    "target": row["target"],
                    "license": "CC0-1.0",
                    "source": "math_sft_v0.1",
                    "split": split,
                }
            )
    return output


def translation_rows(split: str) -> list[dict]:
    output = []
    for row in read_jsonl(PARALLEL_ROOT / f"{split}.jsonl"):
        dyula, french = row["texts"]["dyu"].strip(), row["texts"]["fr"].strip()
        if not dyula or not french:
            continue
        identifier = row["record_id"]
        output.extend(
            (
                {
                    "example_id": f"translation_dyu_fr_{identifier}",
                    "task_family": "translation_dyu_fr",
                    "prompt": f"Instruction : Traduis ce texte du dioula vers le français.\nTexte : {dyula}\nRéponse :",
                    "target": f" {french}\n",
                    "license": row["license"],
                    "source": row["source_dataset"],
                    "split": split,
                    "group_id": identifier,
                },
                {
                    "example_id": f"translation_fr_dyu_{identifier}",
                    "task_family": "translation_fr_dyu",
                    "prompt": f"Instruction : Traduis ce texte du français vers le dioula.\nTexte : {french}\nRéponse :",
                    "target": f" {dyula}\n",
                    "license": row["license"],
                    "source": row["source_dataset"],
                    "split": split,
                    "group_id": identifier,
                },
            )
        )
    return output


def wdi_rows() -> list[dict]:
    rows = []
    for index, sentence in enumerate(WDI_PATH.read_text(encoding="utf-8").splitlines()):
        row = grounded_wdi_row(sentence, index)
        if row is not None:
            rows.append(row)
    return rows


def validate(rows_by_split: dict[str, list[dict]]) -> None:
    identifiers, prompts = set(), {}
    for split, rows in rows_by_split.items():
        if not rows:
            raise RuntimeError(f"split vide : {split}")
        split_prompts = set()
        for row in rows:
            required = {"example_id", "task_family", "prompt", "target", "license", "source", "split"}
            if required - row.keys():
                raise RuntimeError(f"champs manquants : {row}")
            if row["split"] != split or not row["prompt"].strip() or not row["target"].strip():
                raise RuntimeError(row["example_id"])
            if row["example_id"] in identifiers:
                raise RuntimeError(f"identifiant dupliqué : {row['example_id']}")
            identifiers.add(row["example_id"])
            split_prompts.add(row["prompt"])
        prompts[split] = split_prompts
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = prompts[left] & prompts[right]
        if overlap:
            raise RuntimeError(f"fuite de prompts {left}/{right} : {len(overlap)}")


def main() -> None:
    for path in (
        MATH_ROOT / "train.jsonl",
        MATH_ROOT / "validation.jsonl",
        PARALLEL_ROOT / "train.jsonl",
        PARALLEL_ROOT / "validation.jsonl",
        PARALLEL_ROOT / "test.jsonl",
        WDI_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows_by_split = {
        split: core_rows(split) + translation_rows(split)
        for split in ("train", "validation", "test")
    }
    rows_by_split["train"].extend(math_rows("train"))
    rows_by_split["validation"].extend(math_rows("validation"))
    for row in wdi_rows():
        rows_by_split[row["split"]].append(row)
    for rows in rows_by_split.values():
        rows.sort(key=lambda row: row["example_id"])
    validate(rows_by_split)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for split, rows in rows_by_split.items():
        path = OUTPUT_ROOT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        artifacts[split] = {
            "examples": len(rows),
            "task_families": dict(sorted(Counter(row["task_family"] for row in rows).items())),
            "characters": sum(len(row["prompt"]) + len(row["target"]) for row in rows),
            "path": str(path),
            "sha256": sha256(path),
        }

    report = {
        "dataset_id": "instruction_sft_v0.2",
        "status": "train_validation_and_sealed_test",
        "target_loss_only": True,
        "raw_language_replay_expected": True,
        "sources": {
            "assistant_core": "curated local CC0 seed",
            "math_verified": "math_sft_v0.1 deterministic and programmatically verified",
            "translation": "uvci/koumankan4dyula CC-BY-SA-4.0",
            "grounded_wdi": "World Bank WDI context-grounded extraction CC-BY-4.0",
        },
        "split_policy": {
            "math": "existing leak-free train/validation; independent math benchmarks remain sealed",
            "translation": "existing component-level train/validation/test",
            "grounded_wdi": "indicator-group hash split 90/5/5",
            "assistant_core": "manually disjoint prompts",
        },
        "splits": artifacts,
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
