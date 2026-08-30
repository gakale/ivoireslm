#!/usr/bin/env python3
"""Construit des splits parallèles dioula-français-anglais sans fuite exacte."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.ivoirian_languages import (
    assign_components_to_splits,
    deduplicate_parallel_records,
    leakage_components,
    quality_statistics,
)


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
SOURCE_ROOT = STORAGE_ROOT / "snapshots/ivoirian_languages_v0.1/koumankan4dyula_v1.0.0"
OUTPUT_ROOT = STORAGE_ROOT / "derived/ivoirian_parallel_v0.1"
LANGUAGES = ("dyu", "fr", "en")
RATIOS = {"train": 0.9, "validation": 0.05, "test": 0.05}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"ligne JSONL invalide : {line_number}")
        records.append(record)
    return records


def main() -> None:
    source_manifest = SOURCE_ROOT / "records.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"snapshot dioula absent : {source_manifest}")
    original = read_records(source_manifest)
    complete = [row for row in original if all(row.get("texts", {}).get(lang) for lang in LANGUAGES)]
    deduplicated = deduplicate_parallel_records(complete, LANGUAGES)
    components = leakage_components(deduplicated, LANGUAGES)
    splits = assign_components_to_splits(components, RATIOS)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    split_reports = {}
    language_split_texts = {language: {} for language in LANGUAGES}
    for split, rows in splits.items():
        manifest_path = OUTPUT_ROOT / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                derived = {
                    "record_id": row["record_id"],
                    "source_dataset": row["source_dataset"],
                    "source_split": row["source_split"],
                    "source_row_index": row["source_row_index"],
                    "license": row["license"],
                    "texts": {language: row["texts"][language] for language in LANGUAGES},
                    "split": split,
                }
                stream.write(json.dumps(derived, ensure_ascii=False, sort_keys=True) + "\n")
        artifacts = {}
        for language in LANGUAGES:
            texts = [row["texts"][language] for row in rows]
            language_split_texts[language][split] = texts
            path = OUTPUT_ROOT / f"{split}.{language}.txt"
            path.write_text("\n".join(texts) + ("\n" if texts else ""), encoding="utf-8")
            artifacts[language] = {
                "records": len(texts),
                "characters": sum(map(len, texts)),
                "path": str(path),
                "sha256": sha256(path),
            }
        split_reports[split] = {
            "records": len(rows),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "languages": artifacts,
        }

    report = {
        "dataset_id": "ivoirian_parallel_v0.1",
        "source_snapshot": str(SOURCE_ROOT),
        "source_manifest_sha256": sha256(source_manifest),
        "license": "CC-BY-SA-4.0",
        "languages": LANGUAGES,
        "requested_ratios": RATIOS,
        "original_records": len(original),
        "complete_parallel_records": len(complete),
        "exact_parallel_duplicates_removed": len(complete) - len(deduplicated),
        "leakage_components": len(components),
        "splits": split_reports,
        "quality": {
            language: quality_statistics(split_texts)
            for language, split_texts in language_split_texts.items()
        },
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
