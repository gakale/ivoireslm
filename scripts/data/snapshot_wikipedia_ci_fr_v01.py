#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import Counter, deque
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.wikipedia_ci import clean_wikipedia_extract, is_useful_page, split_for_page


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
OUTPUT_ROOT = STORAGE_ROOT / "snapshots/wikipedia_ci_fr_v0.1"
API_URL = "https://fr.wikipedia.org/w/api.php"
ROOT_CATEGORY = "Catégorie:Côte d'Ivoire"
MAX_CATEGORY_DEPTH = 2
MAX_PAGES = 300
USER_AGENT = "IvoireSLM/0.1 (research corpus; https://github.com/gakale/ivoireslm)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_request(parameters: dict, retries: int = 8) -> dict:
    query = urllib.parse.urlencode({"format": "json", "formatversion": 2, **parameters})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt + 1 == retries:
                raise
            delay = float(error.headers.get("Retry-After", min(30, 2 ** attempt))) if error.code == 429 else min(30, 2 ** attempt)
            time.sleep(delay)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError("boucle de reprise impossible")


def category_members(category: str) -> list[dict]:
    rows = []
    continuation = None
    while True:
        parameters = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page|subcat",
            "cmlimit": "max",
        }
        if continuation:
            parameters["cmcontinue"] = continuation
        response = api_request(parameters)
        rows.extend(response["query"]["categorymembers"])
        continuation = response.get("continue", {}).get("cmcontinue")
        if not continuation:
            return rows


def discover_pages() -> list[dict]:
    queue = deque([(ROOT_CATEGORY, 0)])
    visited_categories = set()
    pages = {}
    while queue:
        category, depth = queue.popleft()
        if category in visited_categories:
            continue
        visited_categories.add(category)
        for member in category_members(category):
            if member["ns"] == 0:
                pages[member["pageid"]] = {"pageid": member["pageid"], "title": member["title"]}
            elif member["ns"] == 14 and depth < MAX_CATEGORY_DEPTH:
                queue.append((member["title"], depth + 1))
    return sorted(pages.values(), key=lambda row: (row["title"].casefold(), row["pageid"]))[:MAX_PAGES]


def fetch_page(discovered: dict) -> dict | None:
    cache_dir = OUTPUT_ROOT / "page_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{discovered['pageid']}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return None if cached.get("excluded") else cached
    response = api_request(
        {
            "action": "query",
            "pageids": str(discovered["pageid"]),
            "prop": "extracts|info|revisions",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "rvprop": "ids|timestamp",
            "rvlimit": 1,
        }
    )
    if "query" not in response:
        raise RuntimeError(f"réponse Wikimedia invalide : {response}")
    page = response["query"]["pages"][0]
    text = clean_wikipedia_extract(page.get("extract", ""))
    if not is_useful_page(page["title"], text):
        cache_path.write_text(json.dumps({"excluded": True, "page_id": page["pageid"]}) + "\n", encoding="utf-8")
        return None
    revision = page.get("revisions", [{}])[0]
    record = {
        "page_id": page["pageid"],
        "title": page["title"],
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "url": page.get("fullurl"),
        "history_url": f"https://fr.wikipedia.org/w/index.php?title={urllib.parse.quote(page['title'].replace(' ', '_'))}&action=history",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": f"Contributeurs de Wikipédia, article « {page['title']} »",
        "split": split_for_page(page["pageid"]),
        "text": text,
    }
    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(cache_path)
    return record


def fetch_pages(discovered: list[dict]) -> list[dict]:
    records = []
    for completed, row in enumerate(discovered, 1):
        record = fetch_page(row)
        if record is not None:
            records.append(record)
        if completed % 25 == 0 or completed == len(discovered):
            print(f"pages traitées : {completed}/{len(discovered)}", flush=True)
        time.sleep(0.4)
    return sorted(records, key=lambda row: (row["split"], row["title"].casefold(), row["page_id"]))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    discovered = discover_pages()
    records = fetch_pages(discovered)
    jsonl_path = OUTPUT_ROOT / "pages.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    artifacts = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in records if row["split"] == split]
        text_path = OUTPUT_ROOT / f"{split}.txt"
        with text_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(f"{row['title']}\n\n{row['text'].rstrip()}\n\n")
        artifacts[split] = {
            "pages": len(rows),
            "characters": len(text_path.read_text(encoding="utf-8")),
            "text_path": str(text_path),
            "text_sha256": sha256(text_path),
        }

    report = {
        "snapshot_id": "wikipedia_ci_fr_v0.1",
        "source": API_URL,
        "root_category": ROOT_CATEGORY,
        "category_depth": MAX_CATEGORY_DEPTH,
        "maximum_discovered_pages": MAX_PAGES,
        "discovered_pages": len(discovered),
        "accepted_pages": len(records),
        "license": "CC BY-SA 4.0",
        "attribution_manifest": str(jsonl_path),
        "attribution_manifest_sha256": sha256(jsonl_path),
        "splits": artifacts,
        "split_counts": dict(Counter(row["split"] for row in records)),
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
