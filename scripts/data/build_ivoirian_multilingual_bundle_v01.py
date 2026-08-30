#!/usr/bin/env python3
"""Prépare un paquet multilingue pour le prochain corpus IvoireSLM."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
PARALLEL_ROOT = STORAGE_ROOT / "derived/ivoirian_parallel_v0.1"
BAOULE_ROOT = STORAGE_ROOT / "snapshots/ivoirian_languages_v0.1/baoule_common_voice_v0.1"
OUTPUT_ROOT = STORAGE_ROOT / "derived/ivoirian_multilingual_bundle_v0.1"
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_dyu_fr(texts: dict[str, str]) -> str:
    return f"Dioula : {texts['dyu']}\nFrançais : {texts['fr']}"


def parallel_documents(split: str) -> list[dict]:
    documents = []
    for row in read_jsonl(PARALLEL_ROOT / f"{split}.jsonl"):
        documents.append(
            {
                "document_id": f"dyu_fr_{row['record_id']}",
                "group_id": f"koumankan:{row['record_id']}",
                "source_id": "koumankan4dyula_v1.0.0",
                "source_url": "https://huggingface.co/datasets/uvci/koumankan4dyula",
                "source_split": row["source_split"],
                "source_row_index": row["source_row_index"],
                "language": "dyu-fr",
                "languages": ["dyu", "fr"],
                "domain": "ivorian_multilingual_parallel",
                "content_type": "parallel_translation",
                "country_code": "CIV",
                "rights_status": "open_license",
                "rights_tier": "A_REDISTRIBUTABLE",
                "license": "CC-BY-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": "Koumankan4Dyula, UVCI et partenaires AI4D",
                "split": split,
                "text": render_dyu_fr(row["texts"]),
            }
        )
    return documents


def baoule_documents(split: str) -> list[dict]:
    path = BAOULE_ROOT / f"{split}.bci.txt"
    documents = []
    for index, text in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text = text.strip()
        if not text:
            continue
        identity = hashlib.sha256(f"{split}\0{index}\0{text}".encode("utf-8")).hexdigest()[:24]
        documents.append(
            {
                "document_id": f"bci_{identity}",
                "group_id": f"baoule-common-voice:{split}:{index}",
                "source_id": "baoule_common_voice_v0.1",
                "source_url": "https://huggingface.co/datasets/Klayt/baoule-common-voice",
                "source_split": split,
                "source_row_index": index,
                "language": "bci",
                "languages": ["bci"],
                "domain": "ivorian_language_natural",
                "content_type": "speech_transcription",
                "country_code": "CIV",
                "rights_status": "public_domain_dedication",
                "rights_tier": "A_REDISTRIBUTABLE",
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "attribution": "Baoulé Common Voice",
                "split": split,
                "text": text,
            }
        )
    return documents


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refus d'écraser {OUTPUT_ROOT}")
    required = [PARALLEL_ROOT / f"{split}.jsonl" for split in SPLITS]
    required += [BAOULE_ROOT / f"{split}.bci.txt" for split in SPLITS]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    building = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".building")
    if building.exists():
        raise FileExistsError(building)
    building.mkdir(parents=True)
    split_reports = {}
    language_counts = Counter()
    for split in SPLITS:
        documents = parallel_documents(split) + baoule_documents(split)
        documents.sort(key=lambda row: row["document_id"])
        path = building / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for document in documents:
                stream.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
                language_counts[document["language"]] += 1
        split_reports[split] = {
            "documents": len(documents),
            "characters": sum(len(row["text"]) for row in documents),
            "path": str(OUTPUT_ROOT / path.name),
            "sha256": sha256(path),
        }

    report = {
        "bundle_id": "ivoirian_multilingual_bundle_v0.1",
        "purpose": "future_corpus_only",
        "changes_current_17m_experiment": False,
        "languages": ["bci", "dyu", "fr"],
        "licenses": ["CC0-1.0", "CC-BY-SA-4.0"],
        "language_document_counts": dict(sorted(language_counts.items())),
        "splits": split_reports,
    }
    report_path = building / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256(report_path)
    building.replace(OUTPUT_ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

