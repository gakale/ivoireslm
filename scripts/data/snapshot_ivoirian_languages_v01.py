#!/usr/bin/env python3
"""Crée des snapshots texte baoulé/dioula sans télécharger les fichiers audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.ivoirian_languages import extract_text_fields, quality_statistics, stable_record_id


API_ROOT = "https://datasets-server.huggingface.co"
STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
OUTPUT_ROOT = STORAGE_ROOT / "snapshots/ivoirian_languages_v0.1"
PAGE_SIZE = 100
USER_AGENT = "IvoireSLM/0.1 (research corpus; https://github.com/gakale/ivoireslm)"


@dataclass(frozen=True)
class Source:
    source_id: str
    dataset: str
    config: str
    field_languages: dict[str, str]
    license: str
    license_url: str
    gated: bool = False


SOURCES = {
    "baoule": Source(
        source_id="baoule_common_voice_v0.1",
        dataset="Klayt/baoule-common-voice",
        config="default",
        field_languages={"sentence": "bci"},
        license="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    ),
    "dioula": Source(
        source_id="koumankan4dyula_v1.0.0",
        dataset="uvci/koumankan4dyula",
        config="default",
        field_languages={"dyu": "dyu", "fr": "fr", "en": "en"},
        license="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        gated=True,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_request(endpoint: str, parameters: dict[str, object], token: str | None, retries: int = 6) -> dict:
    url = f"{API_ROOT}/{endpoint}?{urllib.parse.urlencode(parameters)}"
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code in (401, 403):
                raise RuntimeError(
                    "Accès Hugging Face refusé. Accepte les conditions du dataset, puis définis "
                    "HF_TOKEN directement sur la VM."
                ) from error
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2**attempt))
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("boucle de reprise impossible")


def discover_splits(source: Source, token: str | None) -> list[str]:
    response = api_request("splits", {"dataset": source.dataset}, token)
    splits = [
        row["split"]
        for row in response.get("splits", [])
        if row.get("config") == source.config
    ]
    if not splits:
        raise RuntimeError(f"Aucun split trouvé pour {source.dataset}/{source.config}")
    return splits


def fetch_split(source: Source, split: str, token: str | None) -> list[dict]:
    records = []
    offset = 0
    total = None
    while total is None or offset < total:
        response = api_request(
            "rows",
            {
                "dataset": source.dataset,
                "config": source.config,
                "split": split,
                "offset": offset,
                "length": PAGE_SIZE,
            },
            token,
        )
        total = int(response["num_rows_total"])
        rows = response.get("rows", [])
        if not rows and offset < total:
            raise RuntimeError(f"Page vide inattendue à l'offset {offset}/{total}")
        for wrapped in rows:
            row_index = int(wrapped["row_idx"])
            texts = extract_text_fields(wrapped["row"], source.field_languages)
            if not texts:
                continue
            records.append(
                {
                    "record_id": stable_record_id(source.dataset, split, row_index),
                    "source_dataset": source.dataset,
                    "source_split": split,
                    "source_row_index": row_index,
                    "license": source.license,
                    "texts": texts,
                }
            )
        offset += len(rows)
        print(f"{source.source_id}/{split} : {min(offset, total)}/{total}", flush=True)
    return records


def write_snapshot(source: Source, token: str | None) -> dict:
    source_root = OUTPUT_ROOT / source.source_id
    source_root.mkdir(parents=True, exist_ok=True)
    split_reports = {}
    all_records = []
    language_split_texts: dict[str, dict[str, list[str]]] = {}
    for split in discover_splits(source, token):
        records = fetch_split(source, split, token)
        all_records.extend(records)
        languages = sorted({language for record in records for language in record["texts"]})
        artifacts = {}
        for language in languages:
            path = source_root / f"{split}.{language}.txt"
            texts = [record["texts"][language] for record in records if language in record["texts"]]
            language_split_texts.setdefault(language, {})[split] = texts
            path.write_text("\n".join(texts) + ("\n" if texts else ""), encoding="utf-8")
            artifacts[language] = {
                "records": len(texts),
                "characters": sum(len(text) for text in texts),
                "path": str(path),
                "sha256": sha256(path),
            }
        split_reports[split] = {"records": len(records), "languages": artifacts}

    manifest_path = source_root / "records.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "snapshot_id": source.source_id,
        "source_dataset": source.dataset,
        "source_url": f"https://huggingface.co/datasets/{source.dataset}",
        "config": source.config,
        "gated": source.gated,
        "audio_downloaded": False,
        "license": source.license,
        "license_url": source.license_url,
        "field_languages": source.field_languages,
        "records": len(all_records),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "splits": split_reports,
        "quality": {
            language: quality_statistics(split_texts)
            for language, split_texts in sorted(language_split_texts.items())
        },
    }
    report_path = source_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=tuple(SOURCES), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = SOURCES[args.source]
    token = os.environ.get("HF_TOKEN")
    if source.gated and not token:
        raise RuntimeError(
            "Le corpus dioula est soumis à acceptation. Accepte ses conditions sur Hugging Face, "
            "puis définis HF_TOKEN directement sur la VM."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = write_snapshot(source, token)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
