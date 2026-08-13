import json
import hashlib
import os
import shutil
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.corpus import (
    CONTROL_RE,
    EMAIL_RE,
    PHONE_RE,
    jaccard,
    normalize_text,
    normalized_line_key,
    sha256_text,
    token_shingles,
    word_count,
)


STORAGE = Path.home() / "ivoireslm-storage"
LEGACY = STORAGE / "imports" / "legacy_validated_2026-08-13"
LEGACY_ORIGINAL_DERIVED = Path(
    "/home/gnakaleroland/ivoireslm-storage/derived/structured_factual_v0.1"
)
VERSION = os.environ.get("IVOIRESLM_CORPUS_VERSION", "ivoireslm_corpus_v0.1.0")
INCLUDE_WDI = os.environ.get("IVOIRESLM_INCLUDE_WDI", "0") == "1"
INCLUDE_FAOSTAT = os.environ.get("IVOIRESLM_INCLUDE_FAOSTAT", "0") == "1"
INCLUDE_OPEN_FRENCH = os.environ.get("IVOIRESLM_INCLUDE_OPEN_FRENCH", "0") == "1"
OUTPUT = STORAGE / "corpora" / VERSION
CURRENT_MANIFEST = STORAGE / "manifests" / "structured_factual_v0.1.jsonl"
LEGACY_MANIFEST = LEGACY / "manifests" / "structured_factual_v0.1.jsonl"

SPLIT_BY_SOURCE = {
    "civ_datagouv_fish_meat_trade": "validation",
    "civ_datagouv_rainfall_stations": "validation",
    "civ_datagouv_rgph2021": "test",
    "civ_datagouv_bac_1960_2025": "test",
}

LEGACY_ALLOWED = {
    "civ_bac_admission_1960_2026_v0.1": {"domain": "education", "atomic_facts": 67},
    "civ_cocoa_coffee_2022_2023_v0.1": {"domain": "agriculture", "atomic_facts": 24},
    "structured_agri_pop_2024_v01": {
        "document_id": "civ_agricultural_population_2024_v0.1",
        "domain": "agriculture",
        "atomic_facts": 99,
    },
    "civ_ghg_emissions_1990_2020_v0.1": {"domain": "environment_climate", "atomic_facts": 150},
    "civ_urban_population_1975_2021_v0.1": {"domain": "demography", "atomic_facts": 105},
    "civ_livestock_flows_2024_v0.1": {"domain": "agriculture", "atomic_facts": 62},
    "civ_farmgate_prices_2021_2023_v0.1": {"domain": "agriculture", "atomic_facts": 101},
}

CURRENT_ALLOWED = {
    "civ_milk_production_2024_v0.1",
    "civ_rainfall_stations_2022_2023_v0.1",
    "civ_market_prices_2020_2022_v0.1",
    "civ_food_prices_2022_v0.1",
    "civ_rgph2021_population_households_v0.1",
    "civ_fish_meat_trade_1999_2014_v0.1",
}
if INCLUDE_WDI:
    CURRENT_ALLOWED.add("civ_worldbank_wdi_1960_2025_v0.1")
if INCLUDE_FAOSTAT:
    CURRENT_ALLOWED.add("civ_faostat_production_1961_2024_v0.1")
if INCLUDE_OPEN_FRENCH:
    CURRENT_ALLOWED.update(
        {"python_docs_fr_3_14_v0.1", "frwiktionary_definitions_v0.1"}
    )

QUARANTINE = [
    {
        "document_id": "incoming_auteurs_africains",
        "path": "incoming/mac_2026-08-12/documents/auteurs africains.pdf",
        "rights_status": "excluded",
        "reason": "database_storage_prohibited_by_publisher_terms",
    },
    {
        "document_id": "incoming_kourouma_quand_on_refuse",
        "path": "incoming/mac_2026-08-12/documents/Kourouma-Ahmadou-Quand-on-refuse-on-dit-non.pdf",
        "rights_status": "copyrighted",
        "reason": "permission_required_for_training_or_redistribution",
    },
    {
        "document_id": "incoming_9782336004426",
        "path": "incoming/mac_2026-08-12/documents/747833359-9782336004426.pdf",
        "rights_status": "excluded",
        "reason": "commercial_cover_only_not_useful_training_text",
    },
    {
        "document_id": "incoming_roman_ivoirien_expose",
        "path": "incoming/mac_2026-08-12/documents/724958393-Expose-sur-l-histoire-du-roman-ivoirien.docx",
        "rights_status": "unknown",
        "reason": "rights_unknown_and_factual_quality_review_required",
    },
    {
        "document_id": "incoming_convention_collective_2020",
        "path": "incoming/mac_2026-08-12/documents/CONVENTIONS_COLLECTIVES_INTERPROFESSIONNELLES_2020-NI.pdf",
        "rights_status": "restricted_research",
        "reason": "third_party_copy_rights_not_verified",
    },
    {
        "document_id": "legacy_training_pool_v0.1",
        "path": "imports/legacy_quarantine_2026-08-13/manifests/training_pool_v0.1.jsonl",
        "rights_status": "excluded",
        "reason": "stale_pool_with_630_duplicated_catalog_records",
    },
    {
        "document_id": "legacy_data_gouv_natural_11",
        "path": "imports/legacy_quarantine_2026-08-13/cleaned_canonical_v0.3/data_gouv_ci",
        "rights_status": "open_license_needs_review",
        "reason": "interpretive_or_speculative_claims_and_overlap_with_factual_sources",
    },
    {
        "document_id": "legacy_canonical_nonopen_101",
        "path": "imports/legacy_quarantine_2026-08-13/cleaned_canonical_v0.3",
        "rights_status": "unknown_or_restricted",
        "reason": "google_books_books_legal_and_academic_texts_not_cleared_for_official_training",
    },
    {
        "document_id": "legacy_transcriptions",
        "path": "imports/legacy_quarantine_2026-08-13/raw/transcriptions",
        "rights_status": "unknown_or_restricted",
        "reason": "copyright_consent_and_personal_data_review_required",
    },
    {
        "document_id": "legacy_audio_and_transcription_datasets",
        "path": "imports/legacy_quarantine_2026-08-13/raw/legacy_datasets",
        "rights_status": "unknown_or_restricted",
        "reason": "provenance_consent_and_schema_review_required",
    },
    {
        "document_id": "legacy_instruction_candidates",
        "path": "imports/legacy_quarantine_2026-08-13/instruction_candidates",
        "rights_status": "excluded_from_pretraining_v0",
        "reason": "instruction_or_synthetic_provenance_not_validated_and_roadmap_excludes_external_synthetic_data",
    },
]

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_path(record):
    return Path(record.get("output_path") or record.get("file_path") or "")


def collect_records():
    records = []
    for record in read_jsonl(CURRENT_MANIFEST):
        if record["document_id"] not in CURRENT_ALLOWED:
            continue
        records.append(
            dict(
                record,
                provenance_generation="current_reproducible_builder",
                source_manifest_path=str(CURRENT_MANIFEST),
            )
        )
    for legacy_record in read_jsonl(LEGACY_MANIFEST):
        legacy_id = legacy_record["document_id"]
        if legacy_id not in LEGACY_ALLOWED:
            continue
        override = LEGACY_ALLOWED[legacy_id]
        record = dict(legacy_record)
        record.update(override)
        record["document_id"] = override.get("document_id", legacy_id)
        original_path = source_path(record)
        relative_path = original_path.relative_to(LEGACY_ORIGINAL_DERIVED)
        record["output_path"] = str(
            LEGACY / "derived" / "structured_factual_v0.1" / relative_path
        )
        record.pop("file_path", None)
        record["rights_tier"] = record.get("rights_tier") or "A_REDISTRIBUTABLE"
        record["license"] = record.get("license") or "Open government data; source manifest rights tier A_REDISTRIBUTABLE"
        record["group_id"] = record.get("group_id") or record["source_id"]
        record["provenance_generation"] = "legacy_validated_builder_output"
        record["source_manifest_path"] = str(LEGACY_MANIFEST)
        records.append(record)
    records.sort(key=lambda row: row["document_id"])
    expected_documents = (
        13 + int(INCLUDE_WDI) + int(INCLUDE_FAOSTAT) + 2 * int(INCLUDE_OPEN_FRENCH)
    )
    if len(records) != expected_documents:
        raise ValueError(
            f"{expected_documents} documents autorisés attendus, trouvé {len(records)}"
        )
    return records


def main():
    documents_dir = OUTPUT / "documents"
    splits_dir = OUTPUT / "splits"
    manifests_dir = OUTPUT / "manifests"
    reports_dir = OUTPUT / "reports"
    attributions_dir = OUTPUT / "attributions"
    for directory in (documents_dir, splits_dir, manifests_dir, reports_dir, attributions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_records = collect_records()
    corpus_records = []
    source_hash_failures = []
    transformations = []
    pii_findings = []
    suspicious_findings = []
    global_lines = {}
    removed_duplicate_lines = []

    for source in source_records:
        path = source_path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_text(encoding="utf-8")
        raw_hash = sha256_text(raw)
        declared_hash = source.get("text_sha256")
        if declared_hash and raw_hash != declared_hash:
            source_hash_failures.append(source["document_id"])
        cleaned = normalize_text(raw)
        kept_lines = []
        for line_number, line in enumerate(cleaned.splitlines(), 1):
            key = normalized_line_key(line)
            if not key:
                continue
            if key in global_lines:
                removed_duplicate_lines.append(
                    {
                        "document_id": source["document_id"],
                        "line": line_number,
                        "duplicate_of": global_lines[key],
                    }
                )
                continue
            global_lines[key] = {"document_id": source["document_id"], "line": line_number}
            kept_lines.append(line)
        cleaned = "\n".join(kept_lines) + "\n"
        split = SPLIT_BY_SOURCE.get(source["source_id"], "train")
        destination = documents_dir / split / f"{source['document_id']}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned, encoding="utf-8")
        corpus_attribution_path = None
        source_attribution_path = source.get("attribution_path")
        if source_attribution_path:
            source_attribution_path = Path(source_attribution_path)
            if not source_attribution_path.is_file():
                raise FileNotFoundError(source_attribution_path)
            attribution_digest = hashlib.sha256(source_attribution_path.read_bytes()).hexdigest()
            if attribution_digest != source.get("attribution_sha256"):
                raise ValueError(f"Hash d’attribution invalide : {source['document_id']}")
            corpus_attribution_path = attributions_dir / f"{source['document_id']}.jsonl"
            shutil.copyfile(source_attribution_path, corpus_attribution_path)
        transformations.append(
            {
                "document_id": source["document_id"],
                "source_sha256": raw_hash,
                "clean_sha256": sha256_text(cleaned),
                "changed_by_normalization": raw != cleaned,
                "duplicate_lines_removed": len(raw.splitlines()) - len(kept_lines),
            }
        )
        for kind, matcher in (("email", EMAIL_RE), ("phone", PHONE_RE)):
            for match in matcher.finditer(cleaned):
                pii_findings.append({"document_id": source["document_id"], "kind": kind, "value": match.group(0)})
        for marker in ("type « nan »", "None", "�"):
            if marker in cleaned:
                suspicious_findings.append({"document_id": source["document_id"], "marker": marker})
        record = {
            "document_id": source["document_id"],
            "source_id": source["source_id"],
            "group_id": source.get("group_id", source["source_id"]),
            "title": source.get("title", source["document_id"]),
            "country_code": source.get("country_code", "CIV"),
            "country_name": source.get("country_name", "Côte d’Ivoire"),
            "language": source.get("language", "fr"),
            "domain": source.get("domain") or source.get("primary_domain") or "unknown",
            "content_type": source.get("content_type", "deterministic_structured_factual_text"),
            "rights_status": "open_license" if source.get("license") else "redistributable_source",
            "rights_tier": "A_REDISTRIBUTABLE",
            "license": source.get("license"),
            "license_url": source.get("license_url"),
            "attribution": source.get("attribution"),
            "dataset_url": source.get("dataset_url"),
            "split": split,
            "source_path": str(path),
            "source_manifest_path": source["source_manifest_path"],
            "source_table": source.get("source_table"),
            "source_table_sha256": source.get("source_table_sha256"),
            "source_attribution_path": str(source_attribution_path) if source_attribution_path else None,
            "attribution_path": str(corpus_attribution_path) if corpus_attribution_path else None,
            "attribution_sha256": source.get("attribution_sha256"),
            "generation_method": source.get("generation_method"),
            "corpus_path": str(destination),
            "source_sha256": raw_hash,
            "text_sha256": sha256_text(cleaned),
            "characters": len(cleaned),
            "words": word_count(cleaned),
            "lines": len(kept_lines),
            "atomic_facts": source.get("atomic_facts", len(kept_lines)),
            "provenance_generation": source["provenance_generation"],
            "pipeline_status": f"accepted_clean_{VERSION}",
        }
        corpus_records.append(record)

    hashes = defaultdict(list)
    for record in corpus_records:
        hashes[record["text_sha256"]].append(record["document_id"])
    exact_document_duplicates = [ids for ids in hashes.values() if len(ids) > 1]

    shingles = {}
    for record in corpus_records:
        text = Path(record["corpus_path"]).read_text(encoding="utf-8")
        shingles[record["document_id"]] = token_shingles(text)
    near_pairs = []
    max_similarity = 0.0
    max_pair = None
    for left, right in combinations(corpus_records, 2):
        score = jaccard(shingles[left["document_id"]], shingles[right["document_id"]])
        if score > max_similarity:
            max_similarity = score
            max_pair = [left["document_id"], right["document_id"]]
        if score >= 0.80:
            near_pairs.append({"left": left["document_id"], "right": right["document_id"], "score": score})

    groups_by_split = defaultdict(set)
    for record in corpus_records:
        groups_by_split[record["split"]].add(record["group_id"])
    group_leaks = []
    for left, right in combinations(("train", "validation", "test"), 2):
        for group in sorted(groups_by_split[left] & groups_by_split[right]):
            group_leaks.append({"group_id": group, "splits": [left, right]})

    for split in ("train", "validation", "test"):
        rows = [record for record in corpus_records if record["split"] == split]
        with (splits_dir / f"{split}.txt").open("w", encoding="utf-8") as text_handle, (
            splits_dir / f"{split}.jsonl"
        ).open("w", encoding="utf-8") as jsonl_handle:
            for record in rows:
                text = Path(record["corpus_path"]).read_text(encoding="utf-8")
                text_handle.write(text.rstrip() + "\n\n")
                jsonl_handle.write(json.dumps({**record, "text": text}, ensure_ascii=False, sort_keys=True) + "\n")

    quarantine_rows = []
    for item in QUARANTINE:
        path = Path(item["path"])
        if not path.is_absolute():
            path = STORAGE / path
        row = dict(item, resolved_path=str(path), exists=path.exists(), pipeline_status="not_in_official_corpus")
        if path.is_file():
            row["sha256"] = sha256_text(path.read_text(encoding="utf-8", errors="replace")) if path.suffix == ".txt" else hashlib.sha256(path.read_bytes()).hexdigest()
        quarantine_rows.append(row)

    write_jsonl(manifests_dir / "documents.jsonl", corpus_records)
    write_jsonl(manifests_dir / "quarantine.jsonl", quarantine_rows)
    write_jsonl(reports_dir / "transformations.jsonl", transformations)

    split_stats = {}
    for split in ("train", "validation", "test"):
        rows = [record for record in corpus_records if record["split"] == split]
        split_stats[split] = {
            "documents": len(rows),
            "characters": sum(row["characters"] for row in rows),
            "words": sum(row["words"] for row in rows),
            "lines": sum(row["lines"] for row in rows),
            "atomic_facts": sum(row["atomic_facts"] for row in rows),
            "document_ids": [row["document_id"] for row in rows],
        }
    report = {
        "corpus_version": VERSION,
        "documents": len(corpus_records),
        "characters": sum(row["characters"] for row in corpus_records),
        "words": sum(row["words"] for row in corpus_records),
        "lines": sum(row["lines"] for row in corpus_records),
        "atomic_facts": sum(row["atomic_facts"] for row in corpus_records),
        "domains": dict(sorted(Counter(row["domain"] for row in corpus_records).items())),
        "rights_tiers": dict(sorted(Counter(row["rights_tier"] for row in corpus_records).items())),
        "splits": split_stats,
        "source_hash_failures": source_hash_failures,
        "exact_document_duplicates": exact_document_duplicates,
        "exact_normalized_lines_removed": removed_duplicate_lines,
        "near_duplicate_threshold": 0.80,
        "near_duplicate_document_pairs": near_pairs,
        "maximum_document_similarity": {"pair": max_pair, "score": max_similarity},
        "group_leaks_between_splits": group_leaks,
        "pii_findings": pii_findings,
        "suspicious_text_findings": suspicious_findings,
        "control_character_findings": sum(
            bool(CONTROL_RE.search(Path(row["corpus_path"]).read_text(encoding="utf-8"))) for row in corpus_records
        ),
        "quarantined_entries": len(quarantine_rows),
        "quality_gate_passed": not any(
            (source_hash_failures, exact_document_duplicates, near_pairs, group_leaks, pii_findings, suspicious_findings)
        ),
    }
    write_json(reports_dir / "quality_report.json", report)

    if INCLUDE_OPEN_FRENCH:
        build_script, audit_script = "build_corpus_v04.py", "audit_corpus_v04.py"
    elif INCLUDE_FAOSTAT:
        build_script, audit_script = "build_corpus_v03.py", "audit_corpus_v03.py"
    elif INCLUDE_WDI:
        build_script, audit_script = "build_corpus_v02.py", "audit_corpus_v02.py"
    else:
        build_script, audit_script = "build_corpus_v01.py", "audit_corpus_v01.py"
    attributions = "\n".join(
        f"- {row['document_id']} : {row['attribution']}"
        for row in corpus_records
        if row.get("attribution")
    )
    dataset_card = f"""# {VERSION}

Corpus factuel ivoirien nettoyé et traçable, construit le 13 août 2026.

## Contenu

- {report['documents']} documents issus de {report['documents']} groupes de sources indépendants.
- {report['lines']:,} phrases factuelles et {report['atomic_facts']:,} faits atomiques déclarés.
- {report['characters']:,} caractères et {report['words']:,} mots approximatifs.
- Langue principale : français. Périmètre : Côte d'Ivoire et ressources francophones ouvertes.
- Toutes les entrées sont classées `A_REDISTRIBUTABLE` ou sous licence ouverte dans leur manifeste source.

## Splits protégés

- Train : {split_stats['train']['documents']} documents, {split_stats['train']['lines']:,} phrases.
- Validation : {split_stats['validation']['documents']} documents, {split_stats['validation']['lines']:,} phrases.
- Test : {split_stats['test']['documents']} documents, {split_stats['test']['lines']:,} phrases.

La séparation est effectuée par groupe de source. Aucun groupe ne traverse deux splits. Le test ne doit pas servir à entraîner le tokenizer, choisir les hyperparamètres ou corriger le modèle.

## Nettoyage et contrôles

Unicode est normalisé en NFC. Les fins de ligne, espaces de fin et caractères de contrôle sont nettoyés. La graphie ivoirienne, les accents, noms propres et formulations sources ne sont pas standardisés. Les doublons exacts sont détectés après normalisation NFKC/casse/espaces. Les doublons proches sont contrôlés au niveau document par Jaccard sur shingles de cinq mots.

Le rapport vérifie les hashes sources, les doublons, les fuites entre splits, les courriels, les numéros de téléphone ivoiriens, les caractères invalides et les marqueurs techniques (`nan`, `None`, caractère de remplacement).

## Exclusions

Les livres sous copyright, documents aux droits inconnus, copies tierces, métadonnées Google Books, transcriptions non consenties et l'ancien pool contaminé ne font pas partie du corpus officiel. Ils sont conservés séparément pour audit ou demande d'autorisation. Les onze textes narratifs data.gouv.ci restent en révision car ils contiennent des interprétations non sourcées et recouvrent les mêmes tableaux que les textes factuels.

## Attributions explicites

{attributions or '- Voir le champ `license` de chaque entrée dans `manifests/documents.jsonl`.'}

## Limites

Cette version reste spécialisée dans les données factuelles structurées et comporte des libellés officiels anglais provenant de WDI et FAOSTAT. Elle convient pour entraîner le tokenizer caractère de la roadmap et de petits modèles expérimentaux, mais pas encore pour un modèle généraliste. Il faut acquérir davantage de textes naturels ivoiriens explicitement autorisés, notamment littérature, administration, éducation, santé, médias et langues locales.

Le corpus est une collection de documents conservant leurs licences propres. Les définitions du Wiktionnaire restent sous CC BY-SA 4.0 (avec GFDL comme option alternative) et disposent d'un fichier d'attribution par page dans `attributions/`. Elles ne sont pas relicenciées sous la licence des autres sources.

## Reproduction

Depuis le dépôt IvoireSLM sur la VM :

```bash
source ~/venv/bin/activate
python3 scripts/data/{build_script}
python3 scripts/data/{audit_script}
```
"""
    (OUTPUT / "DATASET_CARD.md").write_text(dataset_card, encoding="utf-8")

    checksum_paths = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(OUTPUT)}\n"
        for path in checksum_paths
    )
    (OUTPUT / "SHA256SUMS").write_text(checksums, encoding="utf-8")

    if not report["quality_gate_passed"]:
        raise ValueError("Le corpus a été produit, mais le quality gate a échoué. Voir quality_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
