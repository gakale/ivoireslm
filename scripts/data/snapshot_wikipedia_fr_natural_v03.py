#!/usr/bin/env python3
"""Collecte un supplément Wikipédia français naturel, reprenable et attribué."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError


SNAPSHOT_ID = "wikipedia_fr_natural_v0.3"
API_URL = "https://fr.wikipedia.org/w/api.php"
USER_AGENT = "IvoireSLM/1.1 (research corpus; https://github.com/gakale/ivoireslm)"
SEED = "ivoireslm-wikipedia-fr-natural-v03"
MINIMUM_TEXT_CHARACTERS = 700
MAX_WORKERS = 6
# TextExtracts limite les extraits complets à une page par requête.
PAGE_BATCH_SIZE = 1
CATEGORY_PLAN = (
    ("Catégorie:Côte d'Ivoire", 4, 2_500, 700, "cote_ivoire"),
    ("Catégorie:Afrique de l'Ouest", 3, 1_500, 450, "afrique_ouest"),
    ("Catégorie:Histoire", 2, 1_500, 300, "histoire"),
    ("Catégorie:Géographie", 2, 1_500, 300, "geographie"),
    ("Catégorie:Culture", 2, 1_500, 300, "culture"),
    ("Catégorie:Société", 2, 1_500, 300, "societe"),
    ("Catégorie:Éducation", 3, 1_200, 300, "education"),
    ("Catégorie:Santé", 2, 1_200, 250, "sante"),
    ("Catégorie:Économie", 2, 1_200, 250, "economie"),
    ("Catégorie:Littérature", 2, 1_200, 250, "litterature"),
    ("Catégorie:Art", 2, 1_000, 220, "arts"),
    ("Catégorie:Droit", 2, 1_000, 220, "droit"),
    ("Catégorie:Environnement", 2, 1_000, 220, "environnement"),
)
EXCLUDED_TITLE_PREFIXES = ("Liste de", "Liste des", "Chronologie de", "Modèle:")
SPACE_RE = re.compile(r"[ \t]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(page_id: int, title: str) -> bytes:
    return hashlib.sha256(f"{SEED}:{page_id}:{title}".encode("utf-8")).digest()


def stable_split(page_id: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"frwiki:{page_id}".encode()).digest()[:4], "big")
    return "validation" if value % 20 == 0 else "train"


def clean_extract(text: str) -> str:
    lines, blank = [], False
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = SPACE_RE.sub(" ", raw).strip()
        if not line:
            if lines and not blank:
                lines.append("")
            blank = True
            continue
        if line.startswith(("Voir aussi", "Notes et références", "Liens externes")):
            break
        lines.append(line)
        blank = False
    return "\n".join(lines).strip() + "\n"


def api_request(parameters: dict, retries: int = 8) -> dict:
    query = urllib.parse.urlencode({"format": "json", "formatversion": 2, **parameters})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt + 1 == retries:
                raise
            delay = float(error.headers.get("Retry-After") or min(30, 2**attempt))
            time.sleep(delay)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("reprise API impossible")


def category_members(category: str) -> list[dict]:
    rows, continuation = [], None
    while True:
        parameters = {
            "action": "query", "list": "categorymembers", "cmtitle": category,
            "cmtype": "page|subcat", "cmlimit": "max",
        }
        if continuation:
            parameters["cmcontinue"] = continuation
        response = api_request(parameters)
        rows.extend(response["query"]["categorymembers"])
        continuation = response.get("continue", {}).get("cmcontinue")
        if not continuation:
            return rows


def discover_category(root: str, depth_limit: int, quota: int, category_limit: int) -> list[dict]:
    queue, visited, pages = deque([(root, 0)]), set(), {}
    while queue and len(visited) < category_limit:
        category, depth = queue.popleft()
        if category in visited:
            continue
        visited.add(category)
        for member in category_members(category):
            if member["ns"] == 0:
                pages[member["pageid"]] = {"page_id": member["pageid"], "title": member["title"]}
            elif member["ns"] == 14 and depth < depth_limit:
                queue.append((member["title"], depth + 1))
    return sorted(pages.values(), key=lambda row: stable_score(row["page_id"], row["title"]))[:quota]


def discover(output: Path) -> list[dict]:
    path = output / "discovered_pages.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    pages = {}
    for root, depth, quota, category_limit, label in CATEGORY_PLAN:
        selected = discover_category(root, depth, quota, category_limit)
        print(f"Découverte {label} : {len(selected):,} pages", flush=True)
        for page in selected:
            record = pages.setdefault(page["page_id"], {**page, "category_labels": []})
            record["category_labels"].append(label)
    result = sorted(pages.values(), key=lambda row: stable_score(row["page_id"], row["title"]))
    with path.open("w", encoding="utf-8") as stream:
        for row in result:
            row["category_labels"] = sorted(set(row["category_labels"]))
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def cache_path(root: Path, page_id: int) -> Path:
    return root / "page_cache" / f"{page_id}.json"


def fetch_batch(root: Path, pages: list[dict]) -> None:
    response = api_request({
        "action": "query", "pageids": "|".join(str(row["page_id"]) for row in pages),
        "prop": "extracts|info", "explaintext": 1, "exsectionformat": "plain",
        "exlimit": "max", "inprop": "url",
    })
    returned = {row["pageid"]: row for row in response.get("query", {}).get("pages", [])}
    for discovered in pages:
        page = returned.get(discovered["page_id"])
        payload = {"excluded": True, "page_id": discovered["page_id"], "reason": "missing"}
        if page:
            text = clean_extract(page.get("extract", ""))
            title = page.get("title", discovered["title"])
            if not title.startswith(EXCLUDED_TITLE_PREFIXES) and len(text) >= MINIMUM_TEXT_CHARACTERS:
                payload = {
                    "page_id": page["pageid"], "title": title,
                    "revision_id": page.get("lastrevid"), "revision_timestamp": page.get("touched"),
                    "url": page.get("fullurl"),
                    "history_url": "https://fr.wikipedia.org/w/index.php?title="
                    + urllib.parse.quote(title.replace(" ", "_")) + "&action=history",
                    "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "attribution": f"Contributeurs de Wikipédia, article « {title} »",
                    "category_labels": discovered["category_labels"],
                    "language": "fr", "domain": "natural_french_open",
                    "split": stable_split(page["pageid"]), "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "characters": len(text),
                }
        target = cache_path(root, discovered["page_id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)


def fetch_all(root: Path, pages: list[dict]) -> list[dict]:
    missing = [row for row in pages if not cache_path(root, row["page_id"]).exists()]
    batches = [missing[index:index + PAGE_BATCH_SIZE] for index in range(0, len(missing), PAGE_BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_batch, root, batch) for batch in batches]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 25 == 0 or completed == len(futures):
                print(f"Lots téléchargés : {completed:,}/{len(futures):,}", flush=True)
    records = []
    for page in pages:
        row = json.loads(cache_path(root, page["page_id"]).read_text(encoding="utf-8"))
        if not row.get("excluded"):
            records.append(row)
    return sorted(records, key=lambda row: (row["split"], row["page_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-characters", type=int, default=30_000_000)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(f"snapshot déjà présent : {output}")
    partial.mkdir(parents=True, exist_ok=True)
    pages = discover(partial)
    records = fetch_all(partial, pages)
    pages_path = partial / "pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    labels, split_stats = Counter(), defaultdict(lambda: {"pages": 0, "characters": 0})
    for row in records:
        split_stats[row["split"]]["pages"] += 1
        split_stats[row["split"]]["characters"] += row["characters"]
        for label in row["category_labels"]:
            labels[label] += row["characters"]
    total = sum(row["characters"] for row in records)
    report = {
        "snapshot_id": SNAPSHOT_ID, "status": "train_validation_only_test_not_created",
        "source": API_URL, "license": "CC BY-SA 4.0", "discovered_pages": len(pages),
        "accepted_pages": len(records), "characters": total,
        "characters_by_label": dict(sorted(labels.items())), "splits": dict(split_stats),
        "attribution_manifest": "pages.jsonl", "attribution_manifest_sha256": sha256_file(pages_path),
        "minimum_characters_required": args.minimum_characters,
        "quality_gate_passed": total >= args.minimum_characters,
        "test_created": False,
    }
    (partial / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["quality_gate_passed"]:
        raise RuntimeError("volume français insuffisant ; cache conservé pour reprise")
    partial.rename(output)
    print("\nSnapshot Wikipédia français naturel terminé ✅")
    print("Aucun split test créé ✅")


if __name__ == "__main__":
    main()
