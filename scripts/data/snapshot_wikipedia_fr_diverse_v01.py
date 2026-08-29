#!/usr/bin/env python3
"""Construit un snapshot Wikimedia français diversifié et attribué.

La découverte est déterministe à l'intérieur de chaque catégorie : les pages
sont classées par SHA-256 de leur identifiant et de leur titre. Le cache rend le
téléchargement reprenable. Le snapshot final fige les identifiants de révision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.wikipedia_ci import clean_wikipedia_extract, is_useful_page, split_for_page


STORAGE_ROOT = Path(
    os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage")
)
DEFAULT_OUTPUT = STORAGE_ROOT / "snapshots/wikipedia_fr_diverse_v0.1"
PREVIOUS_SNAPSHOT = STORAGE_ROOT / "snapshots/wikipedia_ci_fr_v0.1/pages.jsonl"
API_URL = "https://fr.wikipedia.org/w/api.php"
USER_AGENT = "IvoireSLM/0.3 (research corpus; https://github.com/gakale/ivoireslm)"
MAX_WORKERS = 8

CATEGORY_PLAN = (
    ("Catégorie:Côte d'Ivoire", 4, 2_500, 350, "cote_ivoire"),
    ("Catégorie:Afrique de l'Ouest", 2, 800, 150, "afrique_ouest"),
    ("Catégorie:Science", 1, 700, 100, "sciences"),
    ("Catégorie:Histoire", 1, 700, 100, "histoire"),
    ("Catégorie:Géographie", 1, 700, 100, "geographie"),
    ("Catégorie:Culture", 1, 700, 100, "culture"),
    ("Catégorie:Société", 1, 700, 100, "societe"),
    ("Catégorie:Éducation", 2, 600, 120, "education"),
    ("Catégorie:Santé", 1, 600, 100, "sante"),
    ("Catégorie:Informatique", 1, 600, 100, "informatique"),
    ("Catégorie:Économie", 1, 600, 100, "economie"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(page: dict) -> bytes:
    value = f"frwiki-diverse-v0.1:{page['pageid']}:{page['title']}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def api_request(parameters: dict, retries: int = 8) -> dict:
    query = urllib.parse.urlencode({"format": "json", "formatversion": 2, **parameters})
    request = urllib.request.Request(
        f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt + 1 == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(30, 2**attempt)
            time.sleep(delay)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2**attempt))
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


def discover_category(
    root: str, maximum_depth: int, quota: int, maximum_categories: int
) -> tuple[list[dict], int]:
    queue = deque([(root, 0)])
    visited_categories = set()
    pages = {}
    while queue:
        if len(visited_categories) >= maximum_categories:
            break
        category, depth = queue.popleft()
        if category in visited_categories:
            continue
        visited_categories.add(category)
        for member in category_members(category):
            if member["ns"] == 0:
                pages[member["pageid"]] = {
                    "pageid": member["pageid"],
                    "title": member["title"],
                }
            elif member["ns"] == 14 and depth < maximum_depth:
                queue.append((member["title"], depth + 1))
    return sorted(pages.values(), key=stable_score)[:quota], len(visited_categories)


def discover_pages(output_root: Path) -> list[dict]:
    discovery_path = output_root / "discovered_pages.jsonl"
    if discovery_path.is_file():
        return [
            json.loads(line)
            for line in discovery_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    pages: dict[int, dict] = {}
    for root, depth, quota, maximum_categories, label in CATEGORY_PLAN:
        selected, categories_visited = discover_category(
            root, depth, quota, maximum_categories
        )
        print(
            f"découverte {label}: {len(selected):,} pages retenues, "
            f"{categories_visited:,} catégories visitées",
            flush=True,
        )
        for page in selected:
            record = pages.setdefault(
                page["pageid"], {**page, "category_labels": [], "root_categories": []}
            )
            record["category_labels"].append(label)
            record["root_categories"].append(root)

    discovered = sorted(pages.values(), key=lambda row: (stable_score(row), row["pageid"]))
    with discovery_path.open("w", encoding="utf-8") as stream:
        for row in discovered:
            row["category_labels"] = sorted(set(row["category_labels"]))
            row["root_categories"] = sorted(set(row["root_categories"]))
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return discovered


def cache_path(output_root: Path, page_id: int) -> Path:
    return output_root / "page_cache" / f"{page_id}.json"


def fetch_page(
    output_root: Path, discovered: dict, previous_page_ids: set[int]
) -> None:
    target = cache_path(output_root, discovered["pageid"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    response = api_request(
        {
            "action": "query",
            "pageids": str(discovered["pageid"]),
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "exlimit": "max",
            "inprop": "url",
        }
    )
    pages = response.get("query", {}).get("pages", [])
    if not pages:
        payload = {
            "excluded": True,
            "page_id": discovered["pageid"],
            "reason": "missing",
        }
    else:
        page = pages[0]
        page_id = page["pageid"]
        text = clean_wikipedia_extract(page.get("extract", ""))
        if not is_useful_page(page.get("title", ""), text):
            payload = {"excluded": True, "page_id": page_id, "reason": "filter"}
        else:
            payload = {
                "page_id": page_id,
                "title": page["title"],
                "revision_id": page.get("lastrevid"),
                "revision_timestamp": page.get("touched"),
                "url": page.get("fullurl"),
                "history_url": (
                    "https://fr.wikipedia.org/w/index.php?title="
                    f"{urllib.parse.quote(page['title'].replace(' ', '_'))}&action=history"
                ),
                "license": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": f"Contributeurs de Wikipédia, article « {page['title']} »",
                "category_labels": discovered["category_labels"],
                "root_categories": discovered["root_categories"],
                "split": (
                    "train" if page_id in previous_page_ids else split_for_page(page_id)
                ),
                "seen_in_previous_snapshot": page_id in previous_page_ids,
                "text": text,
            }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def fetch_pages(
    output_root: Path, discovered: list[dict], previous_page_ids: set[int]
) -> list[dict]:
    cache_dir = output_root / "page_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        row
        for row in discovered
        if not cache_path(output_root, row["pageid"]).is_file()
    ]
    if missing:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(fetch_page, output_root, row, previous_page_ids)
                for row in missing
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 200 == 0 or completed == len(futures):
                    already_cached = len(discovered) - len(missing)
                    print(
                        f"pages traitées : {already_cached + completed:,}/"
                        f"{len(discovered):,}",
                        flush=True,
                    )

    records = []
    for row in discovered:
        cached = json.loads(cache_path(output_root, row["pageid"]).read_text(encoding="utf-8"))
        if not cached.get("excluded"):
            records.append(cached)
    return sorted(records, key=lambda row: (row["split"], row["page_id"]))


def write_snapshot(
    output_root: Path,
    records: list[dict],
    discovered_count: int,
    previous_page_ids: set[int],
) -> None:
    pages_path = output_root / "pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    artifacts = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in records if row["split"] == split]
        text_path = output_root / f"{split}.txt"
        with text_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(f"{row['title']}\n\n{row['text'].rstrip()}\n\n")
        artifacts[split] = {
            "pages": len(rows),
            "characters": text_path.stat().st_size,
            "text_path": str(text_path),
            "text_sha256": sha256_file(text_path),
        }

    labels = Counter()
    characters_by_label = defaultdict(int)
    for row in records:
        for label in row["category_labels"]:
            labels[label] += 1
            characters_by_label[label] += len(row["text"])
    report = {
        "snapshot_id": "wikipedia_fr_diverse_v0.1",
        "source": API_URL,
        "category_plan": [
            {
                "root": root,
                "depth": depth,
                "quota": quota,
                "maximum_categories": maximum_categories,
                "label": label,
            }
            for root, depth, quota, maximum_categories, label in CATEGORY_PLAN
        ],
        "discovered_pages": discovered_count,
        "accepted_pages": len(records),
        "previous_snapshot_pages_forced_to_train": sum(
            row["page_id"] in previous_page_ids for row in records
        ),
        "license": "CC BY-SA 4.0",
        "attribution_manifest": str(pages_path),
        "attribution_manifest_sha256": sha256_file(pages_path),
        "pages_by_label": dict(sorted(labels.items())),
        "characters_by_label": dict(sorted(characters_by_label.items())),
        "splits": artifacts,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_root.mkdir(parents=True, exist_ok=True)
    previous_page_ids = set()
    if PREVIOUS_SNAPSHOT.is_file():
        previous_page_ids = {
            json.loads(line)["page_id"]
            for line in PREVIOUS_SNAPSHOT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    discovered = discover_pages(args.output_root)
    records = fetch_pages(args.output_root, discovered, previous_page_ids)
    if len(records) < 1_000:
        raise RuntimeError(
            f"snapshot refusé : seulement {len(records):,} pages acceptées"
        )
    write_snapshot(
        args.output_root, records, len(discovered), previous_page_ids
    )


if __name__ == "__main__":
    main()
