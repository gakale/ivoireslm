#!/usr/bin/env python3
"""Construit le supplément ouvert v1.0 d'IvoireSLM sans créer de split test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.corpus import normalize_text, sha256_text, word_count


DATASET_ID = "ivoireslm_corpus_supplement_v1.0.0"
SCHEMA_VERSION = "ivoireslm.corpus-supplement.v1"
MAX_DOCUMENT_CHARACTERS = 24_000
MIN_DOCUMENT_CHARACTERS = 80
SPLIT_MODULUS = 20  # 19/20 train, 1/20 validation.

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
}
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)

SOURCE_POLICY = {
    "wikipedia_fr": {
        "domain": "natural_french_open",
        "language": "fr",
        "rights_tier": "B_SHAREALIKE",
        "license": "CC BY-SA 3.0 / GFDL",
        "classification": "ALLOW_SHAREALIKE",
    },
    "wikipedia_en": {
        "domain": "natural_english_open",
        "language": "en",
        "rights_tier": "B_SHAREALIKE",
        "license": "CC BY-SA 3.0 / GFDL",
        "classification": "ALLOW_SHAREALIKE",
    },
    "gsm8k_train": {
        "domain": "mathematics_reasoning",
        "language": "en",
        "rights_tier": "A_REDISTRIBUTABLE",
        "license": "MIT",
        "classification": "ALLOW_OPEN",
    },
    "aqua_train_dev": {
        "domain": "mathematics_reasoning",
        "language": "en",
        "rights_tier": "A_REDISTRIBUTABLE",
        "license": "Apache-2.0",
        "classification": "ALLOW_OPEN",
    },
    "deepseek_harness": {
        "domain": "code_agents",
        "language": "multilingual",
        "rights_tier": "A_REDISTRIBUTABLE",
        "license": "MIT",
        "classification": "ALLOW_OPEN",
    },
    "cisa_kev": {
        "domain": "cybersecurity_defensive",
        "language": "en",
        "rights_tier": "A_PUBLIC_DOMAIN",
        "license": "US Government public data",
        "classification": "ALLOW_OPEN_DEFENSIVE_ONLY",
    },
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"JSONL invalide {path}:{line_number}: {error}") from error


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_split(group_id: str) -> str:
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16)
    return "validation" if value % SPLIT_MODULUS == 0 else "train"


def redact_sensitive(text: str) -> tuple[str, list[str]]:
    findings = []
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            findings.append(name)
            text = pattern.sub(f"[REDACTED_{name.upper()}]", text)
    if EMAIL_PATTERN.search(text):
        findings.append("email")
        text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    return text, sorted(set(findings))


def split_text(text: str, maximum: int = MAX_DOCUMENT_CHARACTERS) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if len(paragraph) > maximum:
            pieces = [paragraph[i : i + maximum] for i in range(0, len(paragraph), maximum)]
        else:
            pieces = [paragraph]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > maximum:
                chunks.append(normalize_text(current))
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(normalize_text(current))
    return chunks


def discover(root: Path, patterns: tuple[str, ...]) -> Path | None:
    candidates = []
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))
    files = sorted({path for path in candidates if path.is_file()}, key=lambda p: str(p).casefold())
    return files[0] if files else None


def record(source_id: str, group_id: str, title: str, text: str, **metadata) -> dict:
    return {
        "source_id": source_id,
        "group_id": group_id,
        "title": title,
        "text": text,
        **SOURCE_POLICY[source_id],
        **metadata,
    }


def wikipedia_records(path: Path, source_id: str):
    for index, row in enumerate(read_jsonl(path)):
        text = row.get("text") or row.get("content") or ""
        identifier = str(row.get("id") or row.get("page_id") or index)
        if text:
            yield record(
                source_id,
                f"wikipedia:{row.get('language', SOURCE_POLICY[source_id]['language'])}:{identifier}",
                str(row.get("title") or identifier),
                text,
                source_url=row.get("url"),
                snapshot=row.get("snapshot"),
            )


def gsm8k_records(path: Path):
    for index, row in enumerate(read_jsonl(path)):
        question, answer = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
        if question and answer:
            yield record(
                "gsm8k_train",
                f"gsm8k:train:{index:06d}",
                f"GSM8K train {index}",
                f"Question: {question}\n\nReasoning and answer: {answer}",
                source_url="https://github.com/openai/grade-school-math",
                original_split="train",
            )


def aqua_rows_from_bytes(name: str, payload: bytes):
    text = payload.decode("utf-8")
    if name.casefold().endswith(".jsonl"):
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            if error.msg != "Extra data":
                raise
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
            return
        yield from value if isinstance(value, list) else value.get("data", [])


def aqua_records(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
        selected = [
            name for name in names
            if name.casefold().endswith((".json", ".jsonl"))
            and any(part in name.casefold() for part in ("train", "dev", "valid"))
            and "test" not in name.casefold()
        ]
        for name in selected:
            split_name = "validation" if any(x in name.casefold() for x in ("dev", "valid")) else "train"
            for index, row in enumerate(aqua_rows_from_bytes(name, archive.read(name))):
                question = str(row.get("question", "")).strip()
                options = row.get("options") or []
                rationale = str(row.get("rationale") or row.get("explanation") or "").strip()
                answer = str(row.get("correct") or row.get("answer") or "").strip()
                if not question or not answer:
                    continue
                options_text = "\n".join(f"- {item}" for item in options)
                text = f"Question: {question}"
                if options_text:
                    text += f"\n\nOptions:\n{options_text}"
                if rationale:
                    text += f"\n\nReasoning: {rationale}"
                text += f"\n\nAnswer: {answer}"
                yield record(
                    "aqua_train_dev",
                    f"aqua:{name}:{index:06d}",
                    f"AQuA {name} {index}",
                    text,
                    source_url="https://github.com/google-deepmind/AQuA",
                    original_split=split_name,
                )


def deepseek_records(path: Path):
    for index, row in enumerate(read_jsonl(path)):
        content = str(row.get("content") or row.get("text") or "")
        relative = str(row.get("path") or index)
        if content and relative.casefold() != "license":
            yield record(
                "deepseek_harness",
                f"deepseek-harness:{relative}",
                relative,
                content,
                source_url=row.get("repository") or "https://github.com/deepseek-ai/deepseek-harness",
                commit=row.get("commit"),
                format=row.get("format"),
            )


def cisa_records(path: Path):
    catalogue = json.loads(path.read_text(encoding="utf-8"))
    for row in catalogue.get("vulnerabilities", []):
        cve = str(row.get("cveID") or "").strip()
        if not cve:
            continue
        text = (
            f"CISA Known Exploited Vulnerability: {cve}\n"
            f"Vendor and product: {row.get('vendorProject', '')} {row.get('product', '')}\n"
            f"Vulnerability: {row.get('vulnerabilityName', '')}\n"
            f"Defensive description: {row.get('shortDescription', '')}\n"
            f"Required defensive action: {row.get('requiredAction', '')}\n"
            f"Date added: {row.get('dateAdded', '')}; remediation due date: {row.get('dueDate', '')}."
        )
        yield record(
            "cisa_kev",
            f"cisa-kev:{cve}",
            cve,
            text,
            source_url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
            catalog_version=catalogue.get("catalogVersion"),
        )


def resolve_sources(root: Path) -> dict[str, Path]:
    patterns = {
        "wikipedia_fr": ("wikipedia_fr_documents.jsonl",),
        "wikipedia_en": ("wikipedia_en_documents.jsonl",),
        "gsm8k_train": ("GSM8K_train.jsonl", "*gsm8k*train*.jsonl"),
        "aqua_train_dev": ("AQuA*.zip", "*aqua*.zip"),
        "deepseek_harness": ("DEEPSEEK_HARNESS_DOCS_AND_TOOLS.jsonl",),
        "cisa_kev": ("CISA_KEV_current.json",),
    }
    resolved = {source: discover(root, glob_patterns) for source, glob_patterns in patterns.items()}
    missing = sorted(source for source, path in resolved.items() if path is None)
    if missing:
        raise FileNotFoundError(f"sources obligatoires introuvables dans {root}: {missing}")
    return {source: path for source, path in resolved.items() if path is not None}


def source_generators(paths: dict[str, Path]):
    return {
        "wikipedia_fr": wikipedia_records(paths["wikipedia_fr"], "wikipedia_fr"),
        "wikipedia_en": wikipedia_records(paths["wikipedia_en"], "wikipedia_en"),
        "gsm8k_train": gsm8k_records(paths["gsm8k_train"]),
        "aqua_train_dev": aqua_records(paths["aqua_train_dev"]),
        "deepseek_harness": deepseek_records(paths["deepseek_harness"]),
        "cisa_kev": cisa_records(paths["cisa_kev"]),
    }


def build(source_root: Path, output: Path) -> dict:
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(f"refus d'écraser une construction existante: {output}")
    paths = resolve_sources(source_root)
    building.mkdir(parents=True)
    try:
        accepted, provenance, seen = [], [], {}
        rejected = Counter()
        redactions = Counter()
        for source_id, rows in source_generators(paths).items():
            for source_row in rows:
                cleaned, findings = redact_sensitive(normalize_text(source_row.pop("text")))
                for finding in findings:
                    redactions[finding] += 1
                for chunk_index, chunk in enumerate(split_text(cleaned), 1):
                    if len(chunk) < MIN_DOCUMENT_CHARACTERS:
                        rejected["too_short"] += 1
                        continue
                    digest = sha256_text(chunk)
                    if digest in seen:
                        rejected["exact_duplicate"] += 1
                        continue
                    seen[digest] = source_row["group_id"]
                    split = source_row.get("original_split")
                    if split not in {"train", "validation"}:
                        split = stable_split(source_row["group_id"])
                    document_id = f"{source_id}_{hashlib.sha256((source_row['group_id'] + ':' + str(chunk_index)).encode()).hexdigest()[:20]}"
                    accepted.append({
                        **source_row,
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "split": split,
                        "text": chunk,
                        "text_sha256": digest,
                        "characters": len(chunk),
                        "words": word_count(chunk),
                        "redactions": findings,
                    })
            provenance.append({
                "source_id": source_id,
                "input_path": str(paths[source_id]),
                "input_sha256": hashlib.sha256(paths[source_id].read_bytes()).hexdigest(),
                **SOURCE_POLICY[source_id],
            })

        accepted.sort(key=lambda row: (row["split"], row["source_id"], row["document_id"]))
        group_splits = defaultdict(set)
        for row in accepted:
            group_splits[row["group_id"]].add(row["split"])
        leaks = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
        if leaks:
            raise RuntimeError(f"fuite train/validation détectée: {len(leaks)} groupes")

        for split in ("train", "validation"):
            rows = [row for row in accepted if row["split"] == split]
            write_jsonl(building / f"{split}.jsonl", rows)
        write_jsonl(building / "documents.jsonl", [{k: v for k, v in row.items() if k != "text"} for row in accepted])
        write_jsonl(building / "admission_manifest.jsonl", provenance)

        source_stats, domain_stats, split_stats = Counter(), Counter(), Counter()
        for row in accepted:
            source_stats[row["source_id"]] += row["characters"]
            domain_stats[row["domain"]] += row["characters"]
            split_stats[row["split"]] += row["characters"]
        report = {
            "dataset_id": DATASET_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "supplement_train_validation_only_test_not_created",
            "purpose": "controlled supplement for a future IvoireSLM corpus mixture",
            "documents": len(accepted),
            "characters": sum(row["characters"] for row in accepted),
            "words": sum(row["words"] for row in accepted),
            "splits": dict(sorted(split_stats.items())),
            "source_characters": dict(sorted(source_stats.items())),
            "domain_characters": dict(sorted(domain_stats.items())),
            "rejected": dict(sorted(rejected.items())),
            "redactions": dict(sorted(redactions.items())),
            "group_leaks": 0,
            "test_created": False,
            "source_files": provenance,
            "excluded_by_policy": [
                "all benchmark test splits",
                "MATH-500 and other evaluation-only data",
                "non-commercial and quarantined sources",
                "NVD and OSV bulk feeds in this pilot",
                "exploit payloads, binaries and proof-of-concept code",
            ],
        }
        if not accepted or any(source not in source_stats for source in SOURCE_POLICY):
            raise RuntimeError("au moins une source admise n'a produit aucun document")
        (building / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (building / "DATASET_CARD.md").write_text(dataset_card(report), encoding="utf-8")
        write_checksums(building)
        building.rename(output)
        return report
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise


def dataset_card(report: dict) -> str:
    return f"""# {DATASET_ID}

Supplément contrôlé destiné à une future composition du corpus IvoireSLM. Il ne remplace pas seul le corpus général v0.9.

## Contenu

- {report['documents']:,} documents ; {report['characters']:,} caractères.
- Splits disponibles : `train` et `validation` uniquement.
- Aucun split `test` n'a été créé.
- Chaque document conserve sa source, sa licence, son URL et son empreinte SHA256.

## Politique d'admission

Wikipedia est conservé dans une couche ShareAlike distincte. GSM8K train, AQuA train/dev et les documents DeepSeek admis gardent leurs licences. CISA KEV est rendu sous forme défensive (description et remédiation), sans collecte de charges d'exploitation.

Les données d'évaluation, les sources non commerciales, les licences en quarantaine, les binaires et les preuves de concept sont exclus. Ce supplément doit être mélangé avec le corpus ivoirien/français existant au moyen de poids d'échantillonnage documentés ; il ne doit pas être concaténé aveuglément.
"""


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.name}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.source_root.expanduser().resolve(), args.output_dir.expanduser().resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nSupplément v1.0 construit, contrôlé et reproductible ✅")
    print("Aucun split test créé ✅")


if __name__ == "__main__":
    main()
