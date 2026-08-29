#!/usr/bin/env python3
"""Assemble le corpus diversifié IvoireSLM v0.7.0 sans écraser la v0.6."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.corpus import CONTROL_RE, normalize_text, normalized_line_key, sha256_text, word_count


STORAGE = Path(
    os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage")
)
SOURCE_V06 = STORAGE / "corpora/ivoireslm_corpus_v0.6.0"
SOURCE_V05 = STORAGE / "corpora/ivoireslm_corpus_v0.5.0"
WIKIPEDIA = STORAGE / "snapshots/wikipedia_fr_diverse_v0.1"
OUTPUT = STORAGE / "corpora/ivoireslm_corpus_v0.7.0"
BUILD = OUTPUT.with_name(OUTPUT.name + ".building")

OLD_WIKIPEDIA_PREFIX = "wikipedia_ci_fr_v0.1_"
EXPANDED_MATH_DOCUMENTS = {
    "frwikibooks_mathematics_v0.1",
    "frwikipedia_mathematics_complement_v0.1",
}
STRUCTURED_DOMAINS = {
    "agriculture",
    "demography",
    "economy",
    "environment_climate",
    "multidomain",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def existing_sources() -> list[dict]:
    v06_rows = read_jsonl(SOURCE_V06 / "manifests/documents.jsonl")
    v05_rows = {
        row["document_id"]: row
        for row in read_jsonl(SOURCE_V05 / "manifests/documents.jsonl")
    }
    sources = []
    for row in v06_rows:
        if row["document_id"].startswith(OLD_WIKIPEDIA_PREFIX):
            continue
        source = dict(row)
        source["split"] = "train"
        source["group_id"] = source["document_id"]
        source["seen_in_previous_corpus"] = True
        source["source_path"] = source["corpus_path"]
        if source["document_id"] in EXPANDED_MATH_DOCUMENTS:
            expanded = v05_rows[source["document_id"]]
            source["source_path"] = expanded["corpus_path"]
            source["expanded_from_v05"] = True
        sources.append(source)
    return sources


def wikipedia_sources() -> list[dict]:
    records = read_jsonl(WIKIPEDIA / "pages.jsonl")
    sources = []
    for row in records:
        labels = set(row["category_labels"])
        domain = (
            "natural_ivoirian_french"
            if "cote_ivoire" in labels
            else "natural_french_open"
        )
        sources.append(
            {
                "document_id": f"frwiki_{row['page_id']}_v0.1",
                "source_id": "wikipedia_fr_diverse_v0.1",
                "group_id": f"frwiki:{row['page_id']}",
                "title": row["title"],
                "country_code": "CIV" if "cote_ivoire" in labels else None,
                "country_name": "Côte d’Ivoire" if "cote_ivoire" in labels else None,
                "language": "fr",
                "domain": domain,
                "content_type": "natural_encyclopedic_text",
                "rights_status": "open_license",
                "rights_tier": "A_REDISTRIBUTABLE",
                "license": row["license"],
                "license_url": row["license_url"],
                "attribution": row["attribution"],
                "source_url": row["url"],
                "history_url": row["history_url"],
                "revision_id": row["revision_id"],
                "revision_timestamp": row["revision_timestamp"],
                "category_labels": row["category_labels"],
                "split": row["split"],
                "seen_in_previous_corpus": row.get(
                    "seen_in_previous_snapshot", False
                ),
                "inline_text": row["text"],
                "generation_method": "mediawiki_plaintext_extract_snapshot",
            }
        )
    return sources


def source_text(source: dict) -> str:
    if "inline_text" in source:
        return normalize_text(source["inline_text"])
    return normalize_text(Path(source["source_path"]).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-characters", type=int, default=35_000_000)
    args = parser.parse_args()

    if OUTPUT.exists() or BUILD.exists():
        raise FileExistsError(f"refus d'écraser une version existante : {OUTPUT}")
    required = (
        SOURCE_V06 / "manifests/documents.jsonl",
        SOURCE_V05 / "manifests/documents.jsonl",
        WIKIPEDIA / "pages.jsonl",
        WIKIPEDIA / "report.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    documents_dir = BUILD / "documents"
    splits_dir = BUILD / "splits"
    reports_dir = BUILD / "reports"
    manifests_dir = BUILD / "manifests"
    for directory in (documents_dir, splits_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=False)

    sources = existing_sources() + wikipedia_sources()
    sources.sort(key=lambda row: (row["split"], row["document_id"]))
    global_lines: dict[str, dict] = {}
    duplicate_lines = []
    output_records = []

    for source in sources:
        raw = source_text(source)
        kept_lines = []
        for line_number, line in enumerate(raw.splitlines(), 1):
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
            global_lines[key] = {
                "document_id": source["document_id"],
                "line": line_number,
            }
            kept_lines.append(line)

        cleaned = "\n".join(kept_lines).strip() + "\n"
        if len(cleaned) < 100:
            continue
        destination = (
            documents_dir / source["split"] / f"{source['document_id']}.txt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned, encoding="utf-8")
        record = {
            **{
                key: value
                for key, value in source.items()
                if key
                not in {
                    "inline_text",
                    "corpus_path",
                    "text_sha256",
                    "characters",
                    "words",
                    "lines",
                    "atomic_facts",
                    "pipeline_status",
                }
            },
            "corpus_path": str(destination).replace(str(BUILD), str(OUTPUT)),
            "text_sha256": sha256_text(cleaned),
            "characters": len(cleaned),
            "words": word_count(cleaned),
            "lines": len(kept_lines),
            "pipeline_status": "accepted_clean_ivoireslm_corpus_v0.7.0",
        }
        output_records.append(record)

    for split in ("train", "validation", "test"):
        rows = [row for row in output_records if row["split"] == split]
        with (splits_dir / f"{split}.txt").open(
            "w", encoding="utf-8"
        ) as text_stream, (splits_dir / f"{split}.jsonl").open(
            "w", encoding="utf-8"
        ) as jsonl_stream:
            for record in rows:
                actual = Path(
                    record["corpus_path"].replace(str(OUTPUT), str(BUILD))
                )
                text = actual.read_text(encoding="utf-8")
                text_stream.write(text.rstrip() + "\n\n")
                jsonl_stream.write(
                    json.dumps(
                        {**record, "text": text}, ensure_ascii=False, sort_keys=True
                    )
                    + "\n"
                )

    domains = Counter()
    split_stats = {}
    for record in output_records:
        domains[record["domain"]] += record["characters"]
    for split in ("train", "validation", "test"):
        rows = [row for row in output_records if row["split"] == split]
        split_stats[split] = {
            "documents": len(rows),
            "characters": sum(row["characters"] for row in rows),
            "words": sum(row["words"] for row in rows),
            "lines": sum(row["lines"] for row in rows),
        }

    total = sum(domains.values())
    math_characters = domains["mathematics"]
    natural_characters = (
        domains["natural_ivoirian_french"] + domains["natural_french_open"]
    )
    dictionary_characters = domains["dictionary_lexicography"]
    structured_characters = sum(domains[domain] for domain in STRUCTURED_DOMAINS)
    group_splits = defaultdict(set)
    for row in output_records:
        group_splits[row["group_id"]].add(row["split"])
    leaks = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    previous_in_held_out = [
        row["document_id"]
        for row in output_records
        if row["split"] != "train" and row.get("seen_in_previous_corpus")
    ]
    controls = sum(
        bool(
            CONTROL_RE.search(
                Path(row["corpus_path"].replace(str(OUTPUT), str(BUILD))).read_text(
                    encoding="utf-8"
                )
            )
        )
        for row in output_records
    )

    report = {
        "corpus_version": "ivoireslm_corpus_v0.7.0",
        "parent_version": "ivoireslm_corpus_v0.6.0",
        "documents": len(output_records),
        "characters": total,
        "words": sum(row["words"] for row in output_records),
        "lines": sum(row["lines"] for row in output_records),
        "domains_by_characters": dict(sorted(domains.items())),
        "natural_characters": natural_characters,
        "natural_percent": 100 * natural_characters / total,
        "mathematics_characters": math_characters,
        "mathematics_percent": 100 * math_characters / total,
        "dictionary_percent": 100 * dictionary_characters / total,
        "structured_factual_percent": 100 * structured_characters / total,
        "splits": split_stats,
        "previous_material_in_validation_or_test": previous_in_held_out,
        "group_leaks_between_splits": leaks,
        "duplicate_lines_removed": len(duplicate_lines),
        "control_character_findings": controls,
        "minimum_characters_required": args.minimum_characters,
        "test_frozen": True,
    }
    report["quality_gate_passed"] = all(
        (
            total >= args.minimum_characters,
            report["natural_percent"] >= 40.0,
            7.0 <= report["mathematics_percent"] <= 15.0,
            report["dictionary_percent"] <= 12.0,
            report["structured_factual_percent"] <= 25.0,
            split_stats["validation"]["characters"] >= 500_000,
            split_stats["test"]["characters"] >= 500_000,
            not leaks,
            not previous_in_held_out,
            controls == 0,
        )
    )

    write_jsonl(manifests_dir / "documents.jsonl", output_records)
    write_jsonl(reports_dir / "duplicate_lines.jsonl", duplicate_lines)
    (reports_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (BUILD / "DATASET_CARD.md").write_text(
        "# IvoireSLM Corpus v0.7.0\n\n"
        "Corpus français diversifié pour l'expérience Transformer 17M.\n\n"
        f"- Caractères : {total:,}\n"
        f"- Français naturel : {report['natural_percent']:.2f} %\n"
        f"- Mathématiques : {report['mathematics_percent']:.2f} %\n"
        f"- Quality gate : {report['quality_gate_passed']}\n",
        encoding="utf-8",
    )
    hash_lines = []
    for path in sorted(BUILD.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_lines.append(f"{digest}  {path.relative_to(BUILD)}")
    (BUILD / "SHA256SUMS").write_text(
        "\n".join(hash_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["quality_gate_passed"]:
        rejected = OUTPUT.with_name(OUTPUT.name + ".rejected")
        if rejected.exists():
            raise FileExistsError(rejected)
        BUILD.replace(rejected)
        raise RuntimeError(f"quality gate v0.7 échoué ; candidat conservé : {rejected}")
    BUILD.replace(OUTPUT)
    print(f"Corpus validé : {OUTPUT} ✅")


if __name__ == "__main__":
    main()
