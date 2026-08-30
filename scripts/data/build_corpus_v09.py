#!/usr/bin/env python3
"""Assemble IvoireSLM v0.9.0 avec français ivoirien et langues ivoiriennes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.corpus import CONTROL_RE, normalize_text, normalized_line_key, sha256_text, word_count


STORAGE = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
BASE = STORAGE / "corpora/ivoireslm_corpus_v0.8.0.rejected"
WIKIDATA = STORAGE / "snapshots/wikipedia_ci_wikidata_v0.1"
INSTITUTIONAL = STORAGE / "snapshots/ivoirian_institutional_v0.1"
MULTILINGUAL = STORAGE / "derived/ivoirian_multilingual_bundle_v0.1"
OUTPUT = STORAGE / "corpora/ivoireslm_corpus_v0.9.0"
BUILD = OUTPUT.with_name(OUTPUT.name + ".building")
MAX_DOCUMENT_CHARACTERS = 32_000
STRUCTURED_DOMAINS = {
    "agriculture", "demography", "economy", "environment_climate", "multidomain"
}
PRIVACY_BLOCKED_SOURCES = {"civ_cei", "civ_solidarity_poverty"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_document(text: str, maximum_characters: int = MAX_DOCUMENT_CHARACTERS) -> list[str]:
    chunks, current, characters = [], [], 0
    for line in text.splitlines():
        addition = len(line) + 1
        if current and characters + addition > maximum_characters:
            chunks.append("\n".join(current).strip() + "\n")
            current, characters = [], 0
        current.append(line)
        characters += addition
    if current:
        chunks.append("\n".join(current).strip() + "\n")
    return chunks


def baseline_sources() -> list[dict]:
    rows = read_jsonl(BASE / "manifests/documents.jsonl")
    sources = []
    for row in rows:
        path = BASE / "documents" / row["split"] / f"{row['document_id']}.txt"
        source = dict(row)
        source.update(
            {
                "source_path": str(path),
                "priority": 0,
                "parent_candidate_document": True,
            }
        )
        sources.append(source)
    return sources


def wikidata_sources() -> list[dict]:
    sources = []
    for row in read_jsonl(WIKIDATA / "pages.jsonl"):
        sources.append(
            {
                "document_id": f"frwiki_wikidata_{row['page_id']}",
                "source_id": "wikipedia_ci_wikidata_v0.1",
                "group_id": f"frwiki:{row['page_id']}",
                "title": row["title"],
                "country_code": "CIV",
                "country_name": "Côte d’Ivoire",
                "language": "fr",
                "domain": "natural_ivoirian_french",
                "content_type": "natural_encyclopedic_text",
                "rights_status": "open_license",
                "rights_tier": "A_REDISTRIBUTABLE",
                "redistributable": True,
                "license": row["license"],
                "license_url": row["license_url"],
                "attribution": row["attribution"],
                "source_url": row["url"],
                "history_url": row["history_url"],
                "revision_id": row["revision_id"],
                "revision_timestamp": row["revision_timestamp"],
                "split": row["split"],
                "inline_text": row["text"],
                "generation_method": "wikidata_relation_mediawiki_plaintext_extract",
                "priority": 1,
            }
        )
    return sources


def institutional_sources() -> list[dict]:
    sources = []
    for row in read_jsonl(INSTITUTIONAL / "documents.jsonl"):
        source = {key: value for key, value in row.items() if key != "text"}
        source.update(
            {
                "group_id": f"institutional:{row['source_id']}:{row['source_url']}",
                "inline_text": row["text"],
                "content_type": "natural_institutional_text",
                "priority": 2,
            }
        )
        sources.append(source)
    return sources


def multilingual_sources() -> list[dict]:
    sources = []
    for split in ("train", "validation", "test"):
        for row in read_jsonl(MULTILINGUAL / f"{split}.jsonl"):
            source = {key: value for key, value in row.items() if key != "text"}
            source.update(
                {
                    "inline_text": row["text"],
                    "redistributable": True,
                    "generation_method": "source_preserving_multilingual_bundle",
                    "priority": 3,
                }
            )
            sources.append(source)
    return sources


def source_text(source: dict) -> str:
    if "inline_text" in source:
        return normalize_text(source["inline_text"])
    return normalize_text(Path(source["source_path"]).read_text(encoding="utf-8"))


def required_paths() -> tuple[Path, ...]:
    return (
        BASE / "manifests/documents.jsonl",
        WIKIDATA / "pages.jsonl",
        INSTITUTIONAL / "documents.jsonl",
        MULTILINGUAL / "train.jsonl",
        MULTILINGUAL / "validation.jsonl",
        MULTILINGUAL / "test.jsonl",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-characters", type=int, default=150_000_000)
    args = parser.parse_args()
    if OUTPUT.exists() or BUILD.exists():
        raise FileExistsError(f"refus d’écraser une version existante : {OUTPUT}")
    for path in required_paths():
        if not path.is_file():
            raise FileNotFoundError(path)

    documents_dir = BUILD / "documents"
    splits_dir = BUILD / "splits"
    reports_dir = BUILD / "reports"
    manifests_dir = BUILD / "manifests"
    for directory in (documents_dir, splits_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=False)

    sources = baseline_sources()
    additions = wikidata_sources() + institutional_sources() + multilingual_sources()
    additions.sort(key=lambda row: (row["priority"], row["split"], row["document_id"]))
    sources.extend(additions)
    global_lines = {}
    duplicate_lines = []
    output_records = []

    for source in sources:
        kept_lines = []
        for line_number, line in enumerate(source_text(source).splitlines(), 1):
            key = normalized_line_key(line)
            if not key:
                continue
            if key in global_lines:
                duplicate_lines.append(
                    {
                        "document_id": source["document_id"],
                        "line": line_number,
                        "duplicate_of": global_lines[key],
                    }
                )
                continue
            global_lines[key] = {"document_id": source["document_id"], "line": line_number}
            kept_lines.append(line)
        cleaned = "\n".join(kept_lines).strip() + "\n"
        if len(cleaned) < 100:
            continue
        chunks = split_document(cleaned)
        for chunk_index, chunk in enumerate(chunks, 1):
            document_id = source["document_id"]
            if len(chunks) > 1:
                document_id = f"{document_id}__v09_chunk_{chunk_index:04d}"
            destination = documents_dir / source["split"] / f"{document_id}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(chunk, encoding="utf-8")
            excluded = {
                "priority", "inline_text", "source_path", "corpus_path", "text_sha256",
                "characters", "words", "lines", "pipeline_status", "chunk_index", "chunk_count"
            }
            record = {key: value for key, value in source.items() if key not in excluded}
            record.update(
                {
                    "document_id": document_id,
                    "source_document_id": source["document_id"],
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "corpus_path": str(destination).replace(str(BUILD), str(OUTPUT)),
                    "text_sha256": sha256_text(chunk),
                    "characters": len(chunk),
                    "words": word_count(chunk),
                    "lines": len(chunk.splitlines()),
                    "pipeline_status": "accepted_clean_ivoireslm_corpus_v0.9.0",
                }
            )
            output_records.append(record)

    for split in ("train", "validation", "test"):
        rows = [row for row in output_records if row["split"] == split]
        with (splits_dir / f"{split}.txt").open("w", encoding="utf-8") as text_stream, (
            splits_dir / f"{split}.jsonl"
        ).open("w", encoding="utf-8") as jsonl_stream:
            for record in rows:
                actual = Path(record["corpus_path"].replace(str(OUTPUT), str(BUILD)))
                text = actual.read_text(encoding="utf-8")
                text_stream.write(text.rstrip() + "\n\n")
                jsonl_stream.write(
                    json.dumps({**record, "text": text}, ensure_ascii=False, sort_keys=True) + "\n"
                )

    domains, source_characters, methods, rights, languages = (Counter() for _ in range(5))
    split_stats = {}
    for row in output_records:
        characters = row["characters"]
        domains[row["domain"]] += characters
        source_characters[row["source_id"]] += characters
        methods[str(row.get("generation_method"))] += characters
        rights[str(row.get("rights_tier") or "UNKNOWN")] += characters
        languages[str(row.get("language") or "unknown")] += characters
    for split in ("train", "validation", "test"):
        rows = [row for row in output_records if row["split"] == split]
        split_stats[split] = {
            "documents": len(rows),
            "characters": sum(row["characters"] for row in rows),
            "words": sum(row["words"] for row in rows),
            "lines": sum(row["lines"] for row in rows),
        }

    total = sum(domains.values())
    natural = domains["natural_ivoirian_french"] + domains["natural_french_open"]
    natural_ci = domains["natural_ivoirian_french"]
    math = domains["mathematics"]
    dictionary = domains["dictionary_lexicography"]
    structured = sum(domains[domain] for domain in STRUCTURED_DOMAINS)
    group_splits = defaultdict(set)
    for row in output_records:
        group_splits[row["group_id"]].add(row["split"])
    leaks = {group: sorted(values) for group, values in group_splits.items() if len(values) > 1}
    controls = sum(
        bool(
            CONTROL_RE.search(
                Path(row["corpus_path"].replace(str(OUTPUT), str(BUILD))).read_text(encoding="utf-8")
            )
        )
        for row in output_records
    )
    blocked_privacy_records = [
        row["document_id"] for row in output_records if row["source_id"] in PRIVACY_BLOCKED_SOURCES
    ]
    report = {
        "corpus_version": "ivoireslm_corpus_v0.9.0",
        "parent_candidate": "ivoireslm_corpus_v0.8.0.rejected",
        "last_trained_corpus": "ivoireslm_corpus_v0.7.0",
        "documents": len(output_records),
        "characters": total,
        "words": sum(row["words"] for row in output_records),
        "lines": sum(row["lines"] for row in output_records),
        "domains_by_characters": dict(sorted(domains.items())),
        "sources_by_characters": dict(sorted(source_characters.items())),
        "generation_methods_by_characters": dict(sorted(methods.items())),
        "rights_tiers_by_characters": dict(sorted(rights.items())),
        "languages_by_characters": dict(sorted(languages.items())),
        "redistributable_characters": rights["A_REDISTRIBUTABLE"],
        "restricted_local_characters": rights["C_PUBLIC_LOCAL_INGEST"],
        "natural_characters": natural,
        "natural_percent": 100 * natural / total,
        "natural_ivoirian_characters": natural_ci,
        "natural_ivoirian_percent": 100 * natural_ci / total,
        "mathematics_characters": math,
        "mathematics_percent": 100 * math / total,
        "dictionary_percent": 100 * dictionary / total,
        "structured_factual_percent": 100 * structured / total,
        "splits": split_stats,
        "group_leaks_between_splits": leaks,
        "duplicate_lines_removed_from_additions": len(duplicate_lines),
        "control_character_findings": controls,
        "privacy_blocked_records_found": blocked_privacy_records,
        "maximum_document_characters": MAX_DOCUMENT_CHARACTERS,
        "minimum_characters_required": args.minimum_characters,
        "test_frozen": True,
        "distribution_policy": {
            "full_corpus": "local_research_and_training_only",
            "public_export": "exclude_C_PUBLIC_LOCAL_INGEST",
        },
    }
    report["quality_gate_passed"] = all(
        (
            total >= args.minimum_characters,
            len(output_records) >= 20_000,
            report["natural_percent"] >= 70.0,
            report["natural_ivoirian_percent"] >= 8.0,
            7.0 <= report["mathematics_percent"] <= 13.0,
            report["dictionary_percent"] <= 3.0,
            report["structured_factual_percent"] <= 5.0,
            split_stats["validation"]["characters"] >= 3_000_000,
            split_stats["test"]["characters"] >= 3_000_000,
            not leaks,
            controls == 0,
            not blocked_privacy_records,
        )
    )

    write_jsonl(manifests_dir / "documents.jsonl", output_records)
    write_jsonl(reports_dir / "duplicate_lines.jsonl", duplicate_lines)
    (reports_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (BUILD / "DATASET_CARD.md").write_text(
        "# IvoireSLM Corpus v0.9.0\n\n"
        "Corpus de recherche local combinant français ouvert, français institutionnel ivoirien "
        "restreint, baoulé et dioula-français.\n\n"
        f"- Caractères : {total:,}\n"
        f"- Français ivoirien naturel : {report['natural_ivoirian_percent']:.2f} %\n"
        f"- Mathématiques : {report['mathematics_percent']:.2f} %\n"
        f"- Données locales non redistribuables : {report['restricted_local_characters']:,} caractères\n"
        f"- Quality gate : {report['quality_gate_passed']}\n\n"
        "Un export public doit exclure tous les documents C_PUBLIC_LOCAL_INGEST.\n",
        encoding="utf-8",
    )
    hash_lines = []
    for path in sorted(BUILD.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            hash_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(BUILD)}")
    (BUILD / "SHA256SUMS").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["quality_gate_passed"]:
        rejected = OUTPUT.with_name(OUTPUT.name + ".rejected")
        if rejected.exists():
            raise FileExistsError(rejected)
        BUILD.replace(rejected)
        raise RuntimeError(f"quality gate v0.9 échoué ; candidat conservé : {rejected}")
    BUILD.replace(OUTPUT)
    print(f"Corpus validé : {OUTPUT} ✅")


if __name__ == "__main__":
    main()
