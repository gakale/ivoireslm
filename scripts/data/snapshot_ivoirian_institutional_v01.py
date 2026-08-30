#!/usr/bin/env python3
"""Extrait un snapshot institutionnel ivoirien local, nettoyé et non redistribuable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.corpus import normalize_text, normalized_line_key, sha256_text


STORAGE_ROOT = Path(
    os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage")
)
DEFAULT_OUTPUT = STORAGE_ROOT / "snapshots/ivoirian_institutional_v0.1"
DOCKER_CONTAINER = "ivoiredata-api-1"
MINIMUM_DOCUMENT_CHARACTERS = 500
MAXIMUM_REPLACEMENT_CHARACTER_RATE = 0.001

# Ces sources contiennent des listes nominatives de bénéficiaires ou de candidats.
BLOCKED_SOURCES = {"civ_solidarity_poverty", "civ_cei"}
SENSITIVE_MARKERS = (
    "beneficiaire",
    "bénéficiaire",
    "liste nominative",
    "liste des candidats",
    "répertoire électoral",
    "repertoire electoral",
)
BINARY_SUFFIXES = (".xlsx", ".xls", ".zip", ".rar", ".7z")
BOILERPLATE_RE = re.compile(
    r"^(?:accueil|menu|navigation|lire la suite|partager|imprimer|haut de page|"
    r"politique de confidentialité|mentions légales|tous droits réservés|copyright)$",
    re.IGNORECASE,
)
DOT_LEADER_RE = re.compile(r"\.{6,}\s*\d*\s*$")


CONTAINER_EXPORTER = r'''
import collections
import json
import pathlib
import pyarrow.parquet as pq

root = pathlib.Path("/app/data_lake/domains")
blocked = set(__BLOCKED_SOURCES__)
for source in sorted(path for path in root.glob("*/*") if path.is_dir()):
    manifest_path = source / "manifest.json"
    metadata = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if metadata.get("rights_tier") != "C_PUBLIC_LOCAL_INGEST" or source.name in blocked:
        continue
    rows = []
    for parquet_path in source.rglob("*.parquet"):
        try:
            table = pq.read_table(
                parquet_path,
                columns=[
                    "source_url", "content_sha256", "chunk_index", "text",
                    "metadata_only", "retrieved_at", "language", "document_title",
                    "document_type", "provider", "extraction_status",
                ],
            )
            rows.extend(table.to_pylist())
        except Exception:
            continue
    versions = collections.defaultdict(list)
    for row in rows:
        if not row.get("text") or row.get("metadata_only"):
            continue
        if not str(row.get("language") or "fr").startswith("fr"):
            continue
        versions[(row["source_url"], row["content_sha256"])].append(row)
    latest = {}
    for (url, content_sha256), version_rows in versions.items():
        retrieved_at = max(str(row.get("retrieved_at") or "") for row in version_rows)
        if url not in latest or retrieved_at > latest[url][0]:
            latest[url] = (retrieved_at, content_sha256, version_rows)
    for url, (retrieved_at, content_sha256, version_rows) in sorted(latest.items()):
        ordered = sorted(version_rows, key=lambda row: row.get("chunk_index") or 0)
        first = ordered[0]
        print(json.dumps({
            "source_id": source.name,
            "source_domain": source.parent.name,
            "source_url": url,
            "source_manifest_url": metadata.get("source_url"),
            "content_sha256": content_sha256,
            "retrieved_at": retrieved_at,
            "title": first.get("document_title"),
            "document_type": first.get("document_type"),
            "provider": first.get("provider"),
            "extraction_status": first.get("extraction_status"),
            "text": "\n".join(row.get("text") or "" for row in ordered),
        }, ensure_ascii=False))
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_url(source_id: str, source_url: str) -> str:
    score = int.from_bytes(
        hashlib.sha256(f"institutional-v0.1:{source_id}:{source_url}".encode()).digest()[:4],
        "big",
    ) % 100
    if score < 5:
        return "test"
    if score < 10:
        return "validation"
    return "train"


def rejection_reason(row: dict) -> str | None:
    source_id = str(row.get("source_id") or "")
    url = str(row.get("source_url") or "").casefold()
    title = str(row.get("title") or "").casefold()
    text = str(row.get("text") or "")
    searchable = unicodedata.normalize("NFKD", f"{url} {title}").casefold()
    if source_id in BLOCKED_SOURCES:
        return "blocked_sensitive_source"
    if url.endswith(BINARY_SUFFIXES):
        return "unsupported_binary_document"
    if any(marker in searchable for marker in SENSITIVE_MARKERS):
        return "sensitive_nominative_list"
    if text.startswith("PK!") or "[Content_Types].xml" in text[:1000]:
        return "binary_extraction"
    if text and text.count("�") / len(text) > MAXIMUM_REPLACEMENT_CHARACTER_RATE:
        return "corrupted_extraction"
    return None


def clean_lines(text: str, seen_lines: set[str]) -> tuple[str, int]:
    text = normalize_text(text)
    kept = []
    removed = 0
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        key = normalized_line_key(line)
        if not key:
            continue
        if BOILERPLATE_RE.fullmatch(line) or DOT_LEADER_RE.search(line):
            removed += 1
            continue
        if key in seen_lines:
            removed += 1
            continue
        seen_lines.add(key)
        kept.append(line)
    return "\n".join(kept).strip() + "\n", removed


def export_rows_from_container() -> list[dict]:
    code = CONTAINER_EXPORTER.replace(
        "__BLOCKED_SOURCES__", json.dumps(sorted(BLOCKED_SOURCES))
    )
    process = subprocess.run(
        ["docker", "exec", "-i", DOCKER_CONTAINER, "python3", "-"],
        input=code,
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]


def write_snapshot(output_root: Path, raw_rows: list[dict]) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    seen_lines: set[str] = set()
    rejected = Counter()
    accepted = []
    duplicate_lines_removed = 0
    for row in sorted(
        raw_rows,
        key=lambda item: hashlib.sha256(
            f"{item['source_id']}:{item['source_url']}".encode()
        ).digest(),
    ):
        reason = rejection_reason(row)
        if reason:
            rejected[reason] += 1
            continue
        cleaned, removed = clean_lines(row["text"], seen_lines)
        duplicate_lines_removed += removed
        if len(cleaned) < MINIMUM_DOCUMENT_CHARACTERS:
            rejected["too_short_after_cleaning"] += 1
            continue
        document_key = f"{row['source_id']}:{row['source_url']}"
        document_id = "institutional_" + hashlib.sha256(document_key.encode()).hexdigest()[:20]
        accepted.append(
            {
                **{key: value for key, value in row.items() if key != "text"},
                "document_id": document_id,
                "country_code": "CIV",
                "country_name": "Côte d’Ivoire",
                "language": "fr",
                "domain": "natural_ivoirian_french",
                "split": split_for_url(row["source_id"], row["source_url"]),
                "rights_status": "public_web_local_research_only",
                "rights_tier": "C_PUBLIC_LOCAL_INGEST",
                "redistributable": False,
                "generation_method": "latest_url_version_global_line_deduplication",
                "text": cleaned,
                "text_sha256": sha256_text(cleaned),
                "characters": len(cleaned),
            }
        )

    manifest_path = output_root / "documents.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in accepted:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    splits = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in accepted if row["split"] == split]
        path = output_root / f"{split}.txt"
        path.write_text(
            "".join(f"{row['title']}\n\n{row['text'].rstrip()}\n\n" for row in selected),
            encoding="utf-8",
        )
        splits[split] = {
            "documents": len(selected),
            "characters": sum(row["characters"] for row in selected),
            "path": str(path),
            "sha256": sha256_file(path),
        }
    report = {
        "snapshot_id": "ivoirian_institutional_v0.1",
        "raw_latest_urls": len(raw_rows),
        "accepted_documents": len(accepted),
        "characters": sum(row["characters"] for row in accepted),
        "sources_by_characters": {},
        "rejections": dict(sorted(rejected.items())),
        "duplicate_or_boilerplate_lines_removed": duplicate_lines_removed,
        "rights_tier": "C_PUBLIC_LOCAL_INGEST",
        "redistributable": False,
        "privacy_exclusions": sorted(BLOCKED_SOURCES),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "splits": splits,
    }
    source_characters = Counter()
    for row in accepted:
        source_characters[row["source_id"]] += row["characters"]
    report["sources_by_characters"] = dict(sorted(source_characters.items()))
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refus d’écraser le snapshot existant : {args.output_root}")
    rows = export_rows_from_container()
    report = write_snapshot(args.output_root, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
