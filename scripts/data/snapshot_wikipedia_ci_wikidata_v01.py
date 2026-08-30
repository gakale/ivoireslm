#!/usr/bin/env python3
"""Découvre via Wikidata des articles ivoiriens absents du snapshot Wikimedia v0.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.wikipedia_ci import clean_wikipedia_extract, is_useful_page, split_for_page


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
DEFAULT_OUTPUT = STORAGE_ROOT / "snapshots/wikipedia_ci_wikidata_v0.1"
PREVIOUS_SNAPSHOT = STORAGE_ROOT / "snapshots/wikipedia_fr_diverse_v0.2/pages.jsonl"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API = "https://fr.wikipedia.org/w/api.php"
USER_AGENT = "IvoireSLM/0.5 (research corpus; https://github.com/gakale/ivoireslm)"
MAX_WORKERS = 8
RESOLUTION_BATCH_SIZE = 40

WIKIDATA_QUERY_TEMPLATE = """
SELECT DISTINCT ?article ?item WHERE {
  ?item wdt:{property_id} wd:Q1008.
  {exclusive_country_filter}
  ?article schema:about ?item;
           schema:isPartOf <https://fr.wikipedia.org/>.
}
LIMIT 10000
""".strip()

WIKIDATA_ROUTES = {
    "country": (
        "P17",
        "FILTER NOT EXISTS { ?item wdt:P17 ?otherCountry. FILTER (?otherCountry != wd:Q1008) }",
    ),
    "citizenship": ("P27", ""),
    "country_of_origin": ("P495", ""),
    "country_for_sport": ("P1532", ""),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(row: dict) -> bytes:
    value = f"wikipedia-ci-wikidata-v0.1:{row['pageid']}:{row['title']}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def request_json(url: str, retries: int = 8) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=150) as response:
                raw = response.read()
            return json.loads(raw)
        except HTTPError as error:
            if attempt + 1 == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else min(30, 2**attempt))
        except (json.JSONDecodeError, TimeoutError, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("boucle de reprise impossible")


def wikipedia_request(parameters: dict) -> dict:
    query = urllib.parse.urlencode({"format": "json", "formatversion": 2, **parameters})
    return request_json(f"{WIKIPEDIA_API}?{query}")


def discover_wikidata(output_root: Path) -> list[dict]:
    path = output_root / "wikidata_results.jsonl"
    if path.is_file():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    aggregated = {}
    for relation, (property_id, exclusive_country_filter) in WIKIDATA_ROUTES.items():
        sparql = WIKIDATA_QUERY_TEMPLATE.format(
            property_id=property_id,
            exclusive_country_filter=exclusive_country_filter,
        )
        query = urllib.parse.urlencode({"query": sparql, "format": "json"})
        payload = request_json(f"{WIKIDATA_URL}?{query}")
        bindings = payload["results"]["bindings"]
        print(f"découverte Wikidata {relation} : {len(bindings):,}", flush=True)
        for binding in bindings:
            article_url = binding["article"]["value"]
            title = urllib.parse.unquote(article_url.split("/wiki/", 1)[1]).replace("_", " ")
            record = aggregated.setdefault(
                title,
                {"requested_title": title, "wikidata_items": [], "relations": []},
            )
            record["wikidata_items"].append(binding["item"]["value"].rsplit("/", 1)[-1])
            record["relations"].append(relation)
    records = []
    for record in aggregated.values():
        record["wikidata_items"] = sorted(set(record["wikidata_items"]))
        record["relations"] = sorted(set(record["relations"]))
        records.append(record)
    records.sort(key=lambda row: row["requested_title"].casefold())
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return records


def resolve_batch(batch: list[dict]) -> list[dict]:
    payload = wikipedia_request(
        {
            "action": "query",
            "titles": "|".join(row["requested_title"] for row in batch),
            "redirects": 1,
            "converttitles": 1,
            "prop": "info",
            "inprop": "url",
        }
    )
    aliases = {row["requested_title"]: row["requested_title"] for row in batch}
    for mapping in payload.get("query", {}).get("normalized", []):
        for original, current in list(aliases.items()):
            if current == mapping["from"]:
                aliases[original] = mapping["to"]
    for mapping in payload.get("query", {}).get("redirects", []):
        for original, current in list(aliases.items()):
            if current == mapping["from"]:
                aliases[original] = mapping["to"]
    metadata = {row["requested_title"]: row for row in batch}
    pages_by_title = {
        page["title"]: page
        for page in payload.get("query", {}).get("pages", [])
        if not page.get("missing")
    }
    resolved = {}
    for original, canonical in aliases.items():
        page = pages_by_title.get(canonical)
        if not page:
            continue
        record = resolved.setdefault(
            page["pageid"],
            {
                "pageid": page["pageid"],
                "title": page["title"],
                "url": page.get("fullurl"),
                "wikidata_items": [],
                "relations": [],
            },
        )
        record["wikidata_items"].extend(metadata[original]["wikidata_items"])
        record["relations"].extend(metadata[original]["relations"])
    for record in resolved.values():
        record["wikidata_items"] = sorted(set(record["wikidata_items"]))
        record["relations"] = sorted(set(record["relations"]))
    return list(resolved.values())


def resolve_pages(output_root: Path, discovered: list[dict], previous_page_ids: set[int]) -> list[dict]:
    path = output_root / "resolved_pages.jsonl"
    if path.is_file():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    resolved = []
    for start in range(0, len(discovered), RESOLUTION_BATCH_SIZE):
        resolved.extend(resolve_batch(discovered[start : start + RESOLUTION_BATCH_SIZE]))
        if start % 400 == 0:
            print(f"titres résolus : {min(start + RESOLUTION_BATCH_SIZE, len(discovered)):,}/{len(discovered):,}", flush=True)
    unique = {row["pageid"]: row for row in resolved if row["pageid"] not in previous_page_ids}
    records = sorted(unique.values(), key=stable_score)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return records


def fetch_page(output_root: Path, discovered: dict) -> None:
    target = output_root / "page_cache" / f"{discovered['pageid']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    payload = wikipedia_request(
        {
            "action": "query",
            "pageids": discovered["pageid"],
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
        }
    )
    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        record = {"excluded": True, "page_id": discovered["pageid"], "reason": "missing"}
    else:
        page = pages[0]
        text = clean_wikipedia_extract(page.get("extract", ""))
        if not is_useful_page(page["title"], text):
            record = {"excluded": True, "page_id": page["pageid"], "reason": "filter"}
        else:
            record = {
                "page_id": page["pageid"],
                "title": page["title"],
                "revision_id": page.get("lastrevid"),
                "revision_timestamp": page.get("touched"),
                "url": page.get("fullurl"),
                "history_url": "https://fr.wikipedia.org/w/index.php?title="
                + urllib.parse.quote(page["title"].replace(" ", "_"))
                + "&action=history",
                "wikidata_items": discovered["wikidata_items"],
                "relations": discovered["relations"],
                "license": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": f"Contributeurs de Wikipédia, article « {page['title']} »",
                "split": split_for_page(page["pageid"]),
                "text": text,
            }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def fetch_pages(output_root: Path, resolved: list[dict]) -> list[dict]:
    cache = output_root / "page_cache"
    cache.mkdir(parents=True, exist_ok=True)
    missing = [row for row in resolved if not (cache / f"{row['pageid']}.json").is_file()]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_page, output_root, row) for row in missing]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 100 == 0 or completed == len(futures):
                print(f"pages téléchargées : {len(resolved) - len(missing) + completed:,}/{len(resolved):,}", flush=True)
    records = []
    for row in resolved:
        cached = json.loads((cache / f"{row['pageid']}.json").read_text(encoding="utf-8"))
        if not cached.get("excluded"):
            records.append(cached)
    return sorted(records, key=lambda row: (row["split"], row["page_id"]))


def write_snapshot(output_root: Path, records: list[dict], discovered: int, resolved: int) -> None:
    pages_path = output_root / "pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    splits = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in records if row["split"] == split]
        path = output_root / f"{split}.txt"
        path.write_text(
            "".join(f"{row['title']}\n\n{row['text'].rstrip()}\n\n" for row in selected),
            encoding="utf-8",
        )
        splits[split] = {
            "pages": len(selected),
            "characters": sum(len(row["text"]) for row in selected),
            "path": str(path),
            "sha256": sha256_file(path),
        }
    report = {
        "snapshot_id": "wikipedia_ci_wikidata_v0.1",
        "source": WIKIDATA_URL,
        "wikipedia_api": WIKIPEDIA_API,
        "query_sha256": hashlib.sha256(
            json.dumps(WIKIDATA_ROUTES, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "discovered_wikidata_articles": discovered,
        "new_resolved_pages": resolved,
        "accepted_pages": len(records),
        "characters": sum(len(row["text"]) for row in records),
        "relations": dict(sorted(Counter(rel for row in records for rel in row["relations"]).items())),
        "license": "CC BY-SA 4.0",
        "attribution_manifest": str(pages_path),
        "attribution_manifest_sha256": sha256_file(pages_path),
        "splits": splits,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not PREVIOUS_SNAPSHOT.is_file():
        raise FileNotFoundError(PREVIOUS_SNAPSHOT)
    previous_page_ids = {
        json.loads(line)["page_id"]
        for line in PREVIOUS_SNAPSHOT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    discovered = discover_wikidata(args.output_root)
    resolved = resolve_pages(args.output_root, discovered, previous_page_ids)
    records = fetch_pages(args.output_root, resolved)
    write_snapshot(args.output_root, records, len(discovered), len(resolved))


if __name__ == "__main__":
    main()
