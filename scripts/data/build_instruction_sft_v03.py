#!/usr/bin/env python3
"""Construit le curriculum SFT v0.3 orienté qualité de génération."""
from __future__ import annotations

import hashlib
import argparse
import json
import itertools
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.assistant_core_v03 import assistant_core_rows


DEFAULT_STORAGE_ROOT = Path.home() / "ivoireslm-storage"
DEFAULT_V02_ROOT = DEFAULT_STORAGE_ROOT / "derived/instruction_sft_v0.2"
DEFAULT_OUTPUT_ROOT = DEFAULT_STORAGE_ROOT / "derived/instruction_sft_v0.3"
WDI_LIMITS = {"train": 5_000, "validation": 500, "test": 500}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_selection(rows: list[dict], limit: int) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(row["example_id"].encode()).digest(),
    )[:limit]


def exact_math_rows(v02_root: Path, split: str) -> list[dict]:
    source = [
        row
        for row in read_jsonl(v02_root / f"{split}.jsonl")
        if row["task_family"] == "math_verified"
    ]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in source:
        groups[row["task_subfamily"]].append(row)
    output = []
    for family, rows in sorted(groups.items()):
        for row in rows:
            if "Réponse :" not in row["target"] or not row["prompt"].endswith("Méthode :"):
                raise RuntimeError(f"format mathématique inattendu : {row['example_id']}")
            answer = row["target"].rsplit("Réponse :", 1)[1].strip()
            problem_prompt = row["prompt"].removesuffix("Méthode :").rstrip()
            output.append(
                {
                    "example_id": row["example_id"].replace("math_sft", "math_exact_v03"),
                    "task_family": "math_exact",
                    "task_subfamily": family,
                    "prompt": f"{problem_prompt}\nRéponse exacte :",
                    "target": f" {answer}\n",
                    "license": "CC0-1.0",
                    "source": "math_sft_v0.1_programmatically_verified_inherited_v02",
                    "split": split,
                }
            )
    return output


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def filtered_translations(v02_root: Path, split: str) -> tuple[list[dict], int]:
    rows = [
        row
        for row in read_jsonl(v02_root / f"{split}.jsonl")
        if row["task_family"] in {"translation_dyu_fr", "translation_fr_dyu"}
    ]
    targets_by_prompt: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        targets_by_prompt[(row["task_family"], normalized(row["prompt"]))].add(
            normalized(row["target"])
        )
    output = []
    for row in rows:
        prompt, target = row["prompt"], row["target"]
        lengths = (len(prompt), len(target))
        unambiguous = len(
            targets_by_prompt[(row["task_family"], normalized(prompt))]
        ) == 1
        clean = (
            unambiguous
            and 8 <= lengths[0] <= 800
            and 2 <= lengths[1] <= 500
            and "�" not in prompt + target
            and "http://" not in prompt + target
            and "https://" not in prompt + target
        )
        if clean:
            copied = dict(row)
            copied["source"] = f"{row['source']}|v03_quality_filter"
            output.append(copied)
    return output, len(rows) - len(output)


def selected_wdi(v02_root: Path, split: str) -> list[dict]:
    rows = [
        row
        for row in read_jsonl(v02_root / f"{split}.jsonl")
        if row["task_family"] == "grounded_wdi"
    ]
    return stable_selection(rows, min(WDI_LIMITS[split], len(rows)))


def validate(rows_by_split: dict[str, list[dict]]) -> None:
    prompts, identifiers = {}, set()
    for split, rows in rows_by_split.items():
        if not rows:
            raise RuntimeError(f"split vide : {split}")
        prompts[split] = set()
        for row in rows:
            required = {"example_id", "task_family", "prompt", "target", "license", "source", "split"}
            if required - row.keys() or row["split"] != split:
                raise RuntimeError(row.get("example_id", "champs manquants"))
            if row["example_id"] in identifiers:
                raise RuntimeError(f"identifiant dupliqué : {row['example_id']}")
            identifiers.add(row["example_id"])
            prompts[split].add(normalized(row["prompt"]))
    for left, right in itertools.combinations(rows_by_split, 2):
        overlap = prompts[left] & prompts[right]
        if overlap:
            raise RuntimeError(f"fuite de prompts {left}/{right} : {len(overlap)}")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v02-root", type=Path, default=DEFAULT_V02_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    for path in (
        args.v02_root / "train.jsonl",
        args.v02_root / "validation.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows_by_split, translation_dropped = {}, {}
    available_splits = ["train", "validation"]
    if (args.v02_root / "test.jsonl").is_file():
        available_splits.append("test")
    for split in available_splits:
        translations, dropped = filtered_translations(args.v02_root, split)
        translation_dropped[split] = dropped
        rows_by_split[split] = (
            assistant_core_rows(split)
            + selected_wdi(args.v02_root, split)
            + translations
        )
    rows_by_split["train"].extend(exact_math_rows(args.v02_root, "train"))
    rows_by_split["validation"].extend(exact_math_rows(args.v02_root, "validation"))
    for rows in rows_by_split.values():
        rows.sort(key=lambda row: row["example_id"])
    validate(rows_by_split)

    args.output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for split, rows in rows_by_split.items():
        path = args.output_root / f"{split}.jsonl"
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
        "dataset_id": "instruction_sft_v0.3",
        "status": (
            "train_validation_and_sealed_test"
            if "test" in rows_by_split
            else "train_validation_only_test_remains_external_and_sealed"
        ),
        "sealed_test_packaged": "test" in rows_by_split,
        "changes_from_v0.2": {
            "math": "answer-only targets; exact answers remain programmatically verified",
            "wdi": "deterministically downsampled because v0.2 reached exact generation",
            "translation": "ambiguous duplicate and basic corruption filters",
            "assistant_core": "80 disjoint teacher-synthetic prompts; provenance explicitly marked",
        },
        "translation_rows_dropped": translation_dropped,
        "splits": artifacts,
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
