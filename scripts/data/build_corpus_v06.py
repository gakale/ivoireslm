#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.corpus import CONTROL_RE, EMAIL_RE, PHONE_RE, normalize_text, normalized_line_key, sha256_text, word_count


STORAGE = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
SOURCE_ROOT = STORAGE / "corpora/ivoireslm_corpus_v0.5.0"
OUTPUT_ROOT = STORAGE / "corpora/ivoireslm_corpus_v0.6.0"
BUILD_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".building")
MATH_ROOT = STORAGE / "derived/math_verified_v0.1"
WIKIPEDIA_ROOT = STORAGE / "snapshots/wikipedia_ci_fr_v0.1"

NEW_TEST_DOCUMENTS = {
    "civ_agricultural_population_2024_v0.1",
    "civ_ghg_emissions_1990_2020_v0.1",
}
NEW_VALIDATION_DOCUMENTS = {
    "civ_fish_meat_trade_1999_2014_v0.1",
    "civ_rainfall_stations_2022_2023_v0.1",
}

CHARACTER_CAPS = {
    "civ_worldbank_wdi_1960_2025_v0.1": 1_200_000,
    "civ_faostat_production_1961_2024_v0.1": 2_000_000,
    "civ_food_prices_2022_v0.1": 1_200_000,
    "civ_market_prices_2020_2022_v0.1": 1_500_000,
    "frwiktionary_definitions_v0.1": 4_000_000,
    "python_docs_fr_3_14_v0.1": 2_500_000,
    "frwikibooks_mathematics_v0.1": 600_000,
    "frwikipedia_mathematics_complement_v0.1": 300_000,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_score(document_id: str, position: int, text: str) -> bytes:
    return hashlib.sha256(f"{document_id}:{position}:{text}".encode("utf-8")).digest()


def capped_text(text: str, document_id: str, maximum_characters: int) -> tuple[str, dict]:
    if len(text) <= maximum_characters:
        return text, {"method": "unchanged_below_cap", "original_characters": len(text), "kept_characters": len(text)}
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) < 20:
        blocks = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = sorted(
        enumerate(blocks),
        key=lambda item: stable_score(document_id, item[0], item[1]),
    )
    selected = set()
    used = 0
    for position, block in candidates:
        cost = len(block) + 2
        if used + cost > maximum_characters and selected:
            continue
        selected.add(position)
        used += cost
        if used >= maximum_characters:
            break
    kept = "\n\n".join(blocks[position] for position in sorted(selected)) + "\n"
    return kept, {
        "method": "deterministic_hash_block_cap",
        "original_characters": len(text),
        "kept_characters": len(kept),
        "original_blocks": len(blocks),
        "kept_blocks": len(selected),
        "cap": maximum_characters,
    }


def wdi_pattern(line: str) -> str:
    key = re.sub(r"[-+]?\d[\d\s., ]*", "<NUM>", line)
    return re.sub(r"«[^»]+»", "«INDICATEUR»", key)


def rebalance_wdi(text: str, document_id: str, maximum_characters: int) -> tuple[str, dict]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    by_pattern = defaultdict(list)
    for position, line in enumerate(lines):
        by_pattern[wdi_pattern(line)].append((position, line))
    pattern_budget = max(1, maximum_characters // max(1, len(by_pattern)))
    selected = []
    for pattern, rows in sorted(by_pattern.items()):
        used = 0
        for position, line in sorted(rows, key=lambda row: stable_score(document_id, row[0], row[1])):
            if used + len(line) + 1 > pattern_budget and used:
                continue
            selected.append((position, line))
            used += len(line) + 1
            if used >= pattern_budget:
                break
    selected.sort()
    kept = "\n".join(line for _, line in selected) + "\n"
    return kept, {
        "method": "deterministic_equal_budget_per_normalized_template",
        "original_characters": len(text),
        "kept_characters": len(kept),
        "original_lines": len(lines),
        "kept_lines": len(selected),
        "patterns": len(by_pattern),
        "pattern_budget_characters": pattern_budget,
    }


def source_split(document_id: str) -> str:
    if document_id in NEW_TEST_DOCUMENTS:
        return "test"
    if document_id in NEW_VALIDATION_DOCUMENTS:
        return "validation"
    return "train"


def new_record(document_id: str, split: str, domain: str, source_id: str, text_path: Path, **metadata) -> dict:
    text = normalize_text(text_path.read_text(encoding="utf-8"))
    return {
        "document_id": document_id,
        "source_id": source_id,
        "group_id": document_id,
        "title": metadata.pop("title", document_id),
        "country_code": metadata.pop("country_code", "CIV"),
        "country_name": metadata.pop("country_name", "Côte d’Ivoire"),
        "language": "fr",
        "domain": domain,
        "content_type": metadata.pop("content_type"),
        "rights_status": "open_license",
        "rights_tier": "A_REDISTRIBUTABLE",
        "license": metadata.pop("license"),
        "split": split,
        "source_path": str(text_path),
        "source_sha256": sha256_text(text),
        "generation_method": metadata.pop("generation_method"),
        **metadata,
    }


def collect_sources() -> list[dict]:
    records = read_jsonl(SOURCE_ROOT / "manifests/documents.jsonl")
    collected = []
    for record in records:
        updated = dict(record)
        updated["split"] = source_split(record["document_id"])
        updated["source_path"] = record["corpus_path"]
        updated["group_id"] = record["document_id"]
        collected.append(updated)

    for split in ("train", "validation", "test"):
        collected.append(
            new_record(
                f"math_verified_v0.1_{split}", split, "mathematics", "math_verified_v0.1",
                MATH_ROOT / f"{split}.txt",
                title=f"Exercices mathématiques vérifiés — {split}",
                content_type="verified_problem_method_solution_answer",
                license="CC0-1.0",
                generation_method="deterministic_rule_based_with_programmatic_answer_verification",
            )
        )
        collected.append(
            new_record(
                f"wikipedia_ci_fr_v0.1_{split}", split, "natural_ivoirian_french", "wikipedia_ci_fr_v0.1",
                WIKIPEDIA_ROOT / f"{split}.txt",
                title=f"Articles Wikipédia francophones liés à la Côte d’Ivoire — {split}",
                content_type="natural_encyclopedic_text",
                license="CC BY-SA 4.0",
                license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                attribution_manifest=str(WIKIPEDIA_ROOT / "pages.jsonl"),
                generation_method="mediawiki_plaintext_extract_snapshot",
            )
        )
    return collected


def main() -> None:
    if OUTPUT_ROOT.exists() or BUILD_ROOT.exists():
        raise FileExistsError(f"la construction refuse d’écraser une version existante : {OUTPUT_ROOT}")
    for required in (SOURCE_ROOT / "manifests/documents.jsonl", MATH_ROOT / "report.json", WIKIPEDIA_ROOT / "report.json"):
        if not required.is_file():
            raise FileNotFoundError(required)

    documents_dir = BUILD_ROOT / "documents"
    splits_dir = BUILD_ROOT / "splits"
    reports_dir = BUILD_ROOT / "reports"
    manifests_dir = BUILD_ROOT / "manifests"
    for directory in (documents_dir, splits_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=False)

    global_lines = {}
    duplicate_lines = []
    output_records = []
    transformations = []
    for source in collect_sources():
        path = Path(source["source_path"])
        raw = normalize_text(path.read_text(encoding="utf-8"))
        cap = CHARACTER_CAPS.get(source["document_id"])
        if source["document_id"] == "civ_worldbank_wdi_1960_2025_v0.1":
            transformed, transformation = rebalance_wdi(raw, source["document_id"], cap)
        elif cap:
            transformed, transformation = capped_text(raw, source["document_id"], cap)
        else:
            transformed, transformation = raw, {"method": "unchanged", "original_characters": len(raw), "kept_characters": len(raw)}

        kept_lines = []
        for line_number, line in enumerate(transformed.splitlines(), 1):
            key = normalized_line_key(line)
            if not key:
                continue
            if key in global_lines:
                duplicate_lines.append({"document_id": source["document_id"], "line": line_number, "duplicate_of": global_lines[key]})
                continue
            global_lines[key] = {"document_id": source["document_id"], "line": line_number}
            kept_lines.append(line)
        cleaned = "\n".join(kept_lines) + "\n"
        destination = documents_dir / source["split"] / f"{source['document_id']}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned, encoding="utf-8")
        record = {
            **{key: value for key, value in source.items() if key not in {"corpus_path", "text_sha256", "characters", "words", "lines", "atomic_facts"}},
            "corpus_path": str(destination).replace(str(BUILD_ROOT), str(OUTPUT_ROOT)),
            "text_sha256": sha256_text(cleaned),
            "characters": len(cleaned),
            "words": word_count(cleaned),
            "lines": len(kept_lines),
            "atomic_facts": len(kept_lines) if source["domain"] == "mathematics" else source.get("atomic_facts", len(kept_lines)),
            "pipeline_status": "accepted_clean_ivoireslm_corpus_v0.6.0",
        }
        output_records.append(record)
        transformations.append({"document_id": source["document_id"], **transformation, "duplicate_lines_removed": len(transformed.splitlines()) - len(kept_lines)})

    for split in ("train", "validation", "test"):
        rows = [record for record in output_records if record["split"] == split]
        with (splits_dir / f"{split}.txt").open("w", encoding="utf-8") as text_stream, (splits_dir / f"{split}.jsonl").open("w", encoding="utf-8") as jsonl_stream:
            for record in rows:
                actual_path = Path(record["corpus_path"].replace(str(OUTPUT_ROOT), str(BUILD_ROOT)))
                text = actual_path.read_text(encoding="utf-8")
                text_stream.write(text.rstrip() + "\n\n")
                jsonl_stream.write(json.dumps({**record, "text": text}, ensure_ascii=False, sort_keys=True) + "\n")

    split_stats = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in output_records if row["split"] == split]
        split_stats[split] = {
            "documents": len(rows),
            "characters": sum(row["characters"] for row in rows),
            "words": sum(row["words"] for row in rows),
            "lines": sum(row["lines"] for row in rows),
            "document_ids": [row["document_id"] for row in rows],
        }
    total_characters = sum(row["characters"] for row in output_records)
    math_characters = sum(row["characters"] for row in output_records if row["domain"] == "mathematics")
    natural_ci_characters = sum(row["characters"] for row in output_records if row["domain"] == "natural_ivoirian_french")
    group_splits = defaultdict(set)
    for row in output_records:
        group_splits[row["group_id"]].add(row["split"])
    group_leaks = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    report = {
        "corpus_version": "ivoireslm_corpus_v0.6.0",
        "parent_version": "ivoireslm_corpus_v0.5.0",
        "documents": len(output_records),
        "characters": total_characters,
        "words": sum(row["words"] for row in output_records),
        "lines": sum(row["lines"] for row in output_records),
        "domains_by_documents": dict(sorted(Counter(row["domain"] for row in output_records).items())),
        "domains_by_characters": dict(sorted(Counter({domain: sum(row["characters"] for row in output_records if row["domain"] == domain) for domain in {row["domain"] for row in output_records}}).items())),
        "mathematics_characters": math_characters,
        "mathematics_percent": 100 * math_characters / total_characters,
        "natural_ivoirian_french_characters": natural_ci_characters,
        "natural_ivoirian_french_percent": 100 * natural_ci_characters / total_characters,
        "splits": split_stats,
        "previous_test_moved_to_train": ["civ_bac_admission_1960_2026_v0.1", "civ_rgph2021_population_households_v0.1"],
        "new_test_documents": sorted(NEW_TEST_DOCUMENTS | {"math_verified_v0.1_test", "wikipedia_ci_fr_v0.1_test"}),
        "test_frozen": True,
        "group_leaks_between_splits": group_leaks,
        "duplicate_lines_removed": len(duplicate_lines),
        "pii_findings": sum(len(EMAIL_RE.findall(Path(row["corpus_path"].replace(str(OUTPUT_ROOT), str(BUILD_ROOT))).read_text(encoding="utf-8"))) + len(PHONE_RE.findall(Path(row["corpus_path"].replace(str(OUTPUT_ROOT), str(BUILD_ROOT))).read_text(encoding="utf-8"))) for row in output_records),
        "control_character_findings": sum(bool(CONTROL_RE.search(Path(row["corpus_path"].replace(str(OUTPUT_ROOT), str(BUILD_ROOT))).read_text(encoding="utf-8"))) for row in output_records),
    }
    report["quality_gate_passed"] = not report["group_leaks_between_splits"] and report["pii_findings"] == 0 and report["control_character_findings"] == 0 and 8.0 <= report["mathematics_percent"] <= 15.0
    write_jsonl(manifests_dir / "documents.jsonl", output_records)
    write_jsonl(reports_dir / "transformations.jsonl", transformations)
    write_jsonl(reports_dir / "duplicate_lines.jsonl", duplicate_lines)
    (reports_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset_card = (
        "# IvoireSLM Corpus v0.6.0\n\n"
        "Version rééquilibrée dérivée de v0.5.0, enrichie de français naturel ivoirien attribué et d’exercices mathématiques vérifiés.\n\n"
        f"- Caractères : {total_characters:,}\n- Mathématiques : {report['mathematics_percent']:.4f} %\n"
        f"- Français naturel ivoirien ajouté : {natural_ci_characters:,} caractères\n- Quality gate : {report['quality_gate_passed']}\n"
    )
    (BUILD_ROOT / "DATASET_CARD.md").write_text(dataset_card, encoding="utf-8")
    hash_lines = []
    for path in sorted(BUILD_ROOT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_lines.append(f"{digest}  {path.relative_to(BUILD_ROOT)}")
    (BUILD_ROOT / "SHA256SUMS").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    if not report["quality_gate_passed"]:
        raise RuntimeError("le quality gate v0.6.0 a échoué")
    BUILD_ROOT.replace(OUTPUT_ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
