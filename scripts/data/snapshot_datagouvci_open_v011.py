#!/usr/bin/env python3
"""Fige les jeux publics de data.gouv.ci en texte ivoirien attribué.

Le collecteur n'utilise que les jeux explicitement placés sous Licence Ouverte,
rejette les schémas susceptibles de contenir des données personnelles et ne
crée que des splits train/validation au niveau du jeu de données.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import re
from pathlib import Path
import time
import urllib.parse
import urllib.request


SNAPSHOT_ID = "data_gouv_ci_open_v0.1.1"
API_ROOT = "https://data.gouv.ci/data-fair/api/v1/datasets"
PORTAL_URL = "https://data.gouv.ci"
LICENSE_URL = "https://data.gouv.ci/pages/licence"
PORTAL_ID = "yCWsyaGpA"
OWNER_ID = "organization:HAbYP_-xC"
USER_AGENT = "IvoireSLM/1.1.1 corpus research (https://github.com/gakale/ivoireslm)"
ROWS_PER_DOCUMENT = 40
PAGE_SIZE = 1_000
MINIMUM_DESCRIPTION_CHARACTERS = 80
MAXIMUM_CELL_CHARACTERS = 1_000
SENSITIVE_FIELD_RE = re.compile(
    r"(?:^|_)(?:nom|prenom|prénom|email|courriel|telephone|téléphone|contact|"
    r"adresse|matricule|numero_cni|num_cni|date_naissance|beneficiaire|"
    r"bénéficiaire|candidat)(?:$|_)",
    re.IGNORECASE,
)
SYSTEM_FIELDS = {"_id", "_i", "_rand", "_score"}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(value: object) -> str:
    parser = TextExtractor()
    parser.feed(str(value or ""))
    text = unescape(" ".join(parser.parts))
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split(dataset_id: str) -> str:
    score = int.from_bytes(
        hashlib.sha256(f"data-gouv-ci-v011:{dataset_id}".encode()).digest()[:4],
        "big",
    ) % 100
    return "validation" if score < 10 else "train"


def request_json(url: str, retries: int = 6) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("boucle de reprise impossible")


def catalogue() -> list[dict]:
    parameters = {
        "size": 1_000,
        "page": 1,
        "select": "id,slug,title,description,updatedAt,dataUpdatedAt,owner,topics,href",
        "owner": OWNER_ID,
        "publicationSites": f"data-fair-portals:{PORTAL_ID}",
        "html": "true",
        "truncate": 10_000,
        "visibility": "public",
    }
    payload = request_json(f"{API_ROOT}?{urllib.parse.urlencode(parameters)}")
    results = payload.get("results", [])
    if len(results) != int(payload.get("count", len(results))):
        raise RuntimeError("catalogue data.gouv.ci incomplet")
    return sorted(results, key=lambda row: str(row.get("id")))


def open_license(metadata: dict) -> bool:
    license_data = metadata.get("license") or {}
    title = str(license_data.get("title") or "").casefold()
    href = str(license_data.get("href") or "").casefold()
    return "licence ouverte" in title or "open licence" in title or "etalab" in href


def sensitive_schema(metadata: dict) -> list[str]:
    findings = []
    for field in metadata.get("schema") or []:
        key = str(field.get("key") or "")
        original = str(field.get("x-originalName") or "")
        normalized = re.sub(r"[^0-9A-Za-zÀ-ÿ]+", "_", f"{key}_{original}").strip("_")
        if SENSITIVE_FIELD_RE.search(normalized):
            findings.append(key or original)
    return sorted(set(findings))


def safe_cell(value: object) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    rendered = re.sub(r"\s+", " ", str(value)).strip()
    if not rendered or len(rendered) > MAXIMUM_CELL_CHARACTERS:
        return None
    return rendered


def fetch_lines(dataset_id: str) -> list[dict]:
    rows: list[dict] = []
    after: str | None = None
    while True:
        parameters: dict[str, object] = {"size": PAGE_SIZE}
        if after:
            parameters["after"] = after
        payload = request_json(
            f"{API_ROOT}/{urllib.parse.quote(dataset_id)}/lines?"
            f"{urllib.parse.urlencode(parameters)}"
        )
        page = payload.get("results", [])
        rows.extend(page)
        next_url = payload.get("next")
        if not next_url or not page:
            break
        parsed = urllib.parse.urlparse(next_url)
        after_values = urllib.parse.parse_qs(parsed.query).get("after")
        if not after_values:
            break
        after = after_values[0]
    return rows


def field_labels(metadata: dict) -> dict[str, str]:
    return {
        str(field.get("key")): plain_text(field.get("x-originalName") or field.get("title") or field.get("key"))
        for field in metadata.get("schema") or []
        if field.get("key") and str(field.get("key")) not in SYSTEM_FIELDS
    }


def render_documents(metadata: dict, rows: list[dict]) -> list[dict]:
    dataset_id = str(metadata["id"])
    title = plain_text(metadata.get("title"))
    description = plain_text(metadata.get("description"))
    owner = plain_text((metadata.get("owner") or {}).get("name")) or "Portail data.gouv.ci"
    updated = str(metadata.get("dataUpdatedAt") or metadata.get("updatedAt") or "")
    labels = field_labels(metadata)
    topics = [plain_text(topic.get("title")) for topic in metadata.get("topics") or []]
    row_sentences = []
    for row in rows:
        facts = []
        for key, label in labels.items():
            value = safe_cell(row.get(key))
            if value is not None:
                facts.append(f"{label} : {value}")
        if facts:
            row_sentences.append(" ; ".join(facts) + ".")

    chunks = [row_sentences[i:i + ROWS_PER_DOCUMENT] for i in range(0, len(row_sentences), ROWS_PER_DOCUMENT)]
    if not chunks:
        chunks = [[]]
    documents = []
    for index, chunk in enumerate(chunks):
        heading = [f"Jeu de données public ivoirien : {title}."]
        if description:
            heading.append(description)
        if chunk:
            heading.append("Données publiées :\n" + "\n".join(chunk))
        text = "\n\n".join(heading).strip()
        if len(text) < MINIMUM_DESCRIPTION_CHARACTERS:
            continue
        source_url = f"{PORTAL_URL}/datasets/{metadata.get('slug') or dataset_id}"
        documents.append({
            "document_id": f"data-gouv-ci:{dataset_id}:{index:05d}",
            "group_id": f"data-gouv-ci:{dataset_id}",
            "source_id": SNAPSHOT_ID,
            "source_url": source_url,
            "title": title,
            "language": "fr-CI",
            "country_code": "CIV",
            "domain": "natural_ivoirian_grounded_verified",
            "content_type": "open_government_dataset_description_and_rows",
            "license": "Licence Ouverte / Open Licence",
            "license_url": LICENSE_URL,
            "rights_tier": "A_REDISTRIBUTABLE",
            "attribution": owner,
            "source_updated_at": updated,
            "topics": topics,
            "split": stable_split(dataset_id),
            "text": text,
        })
    return documents


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"snapshot déjà présent : {output}")
    partial = output.with_name(output.name + ".partial")
    cache = partial / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rejected = Counter()
    all_documents: list[dict] = []
    source_rows = 0

    datasets = catalogue()
    for position, entry in enumerate(datasets, 1):
        dataset_id = str(entry["id"])
        cache_path = cache / f"{dataset_id}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            metadata, rows = cached["metadata"], cached["rows"]
        else:
            metadata = request_json(f"{API_ROOT}/{urllib.parse.quote(dataset_id)}")
            if not open_license(metadata):
                rejected["license_not_open"] += 1
                print(f"{position}/{len(datasets)} — licence rejetée : {dataset_id}", flush=True)
                continue
            sensitive = sensitive_schema(metadata)
            if sensitive:
                rejected["sensitive_schema"] += 1
                print(f"{position}/{len(datasets)} — schéma sensible rejeté : {dataset_id}", flush=True)
                continue
            rows = fetch_lines(dataset_id)
            cache_path.write_text(
                json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False),
                encoding="utf-8",
            )
        source_rows += len(rows)
        all_documents.extend(render_documents(metadata, rows))
        print(
            f"{position}/{len(datasets)} — {dataset_id} : {len(rows)} lignes, "
            f"{len(all_documents)} documents cumulés",
            flush=True,
        )

    all_documents.sort(key=lambda row: (row["split"], row["document_id"]))
    for row in all_documents:
        row["characters"] = len(row["text"])
        row["text_sha256"] = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    write_jsonl(partial / "documents.jsonl", all_documents)
    for split in ("train", "validation"):
        write_jsonl(partial / f"{split}.jsonl", [row for row in all_documents if row["split"] == split])
    train_groups = {row["group_id"] for row in all_documents if row["split"] == "train"}
    validation_groups = {row["group_id"] for row in all_documents if row["split"] == "validation"}
    if train_groups & validation_groups:
        raise RuntimeError("fuite de jeux de données entre train et validation")
    report = {
        "snapshot_id": SNAPSHOT_ID,
        "status": "train_validation_only_test_not_created",
        "source_url": PORTAL_URL,
        "license": "Licence Ouverte / Open Licence",
        "license_url": LICENSE_URL,
        "catalogue_datasets": len(datasets),
        "accepted_datasets": len({row["group_id"] for row in all_documents}),
        "source_rows": source_rows,
        "documents": len(all_documents),
        "characters": sum(row["characters"] for row in all_documents),
        "rejections": dict(sorted(rejected.items())),
        "splits": {
            split: {
                "documents": len([row for row in all_documents if row["split"] == split]),
                "characters": sum(row["characters"] for row in all_documents if row["split"] == split),
                "sha256": sha256_file(partial / f"{split}.jsonl"),
            }
            for split in ("train", "validation")
        },
        "group_leaks": 0,
        "test_created": False,
    }
    (partial / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.output_dir.expanduser().resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nSnapshot data.gouv.ci terminé ✅")
    print("Aucun split test créé ✅")


if __name__ == "__main__":
    main()
