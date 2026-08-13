import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/wikibooks_math_fr_2026-08-13_v0.3"
PARTIAL = SNAPSHOT.with_name(SNAPSHOT.name + ".partial")
API = "https://fr.wikibooks.org/w/api.php"
ROOT_CATEGORY = "Catégorie:Mathématiques"
MAX_DEPTH = 3
USER_AGENT = "IvoireSLM corpus builder/0.5 (https://github.com/gakale/ivoireslm)"
REQUEST_LOCK = threading.Lock()
LAST_REQUEST_AT = 0.0
MIN_REQUEST_INTERVAL = 0.5


def api_call(**parameters):
    global LAST_REQUEST_AT
    parameters.setdefault("action", "query")
    parameters.update(format="json", formatversion="2")
    request = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(parameters),
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(8):
        try:
            with REQUEST_LOCK:
                delay = MIN_REQUEST_INTERVAL - (time.monotonic() - LAST_REQUEST_AT)
                if delay > 0:
                    time.sleep(delay)
                LAST_REQUEST_AT = time.monotonic()
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt == 7:
                raise
            retry_after = float(error.headers.get("Retry-After") or 0)
            time.sleep(max(retry_after, min(60, 2 ** attempt)))
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))


def category_members(category):
    result = []
    continuation = {}
    while True:
        payload = api_call(
            list="categorymembers",
            cmtitle=category,
            cmtype="page|subcat",
            cmlimit="max",
            **continuation,
        )
        result.extend(payload["query"]["categorymembers"])
        if "continue" not in payload:
            return result
        continuation = {"cmcontinue": payload["continue"]["cmcontinue"]}


if SNAPSHOT.exists():
    if not (SNAPSHOT / "snapshot_manifest.json").is_file():
        raise SystemExit(f"Snapshot existant sans manifeste : {SNAPSHOT}")
    print(f"Snapshot déjà présent : {SNAPSHOT}")
    raise SystemExit(0)
PARTIAL.mkdir(parents=True)
PAGE_CACHE = PARTIAL / "rendered_pages"
PAGE_CACHE.mkdir(exist_ok=True)

categories = {}
pages = defaultdict(set)
queue = deque([(ROOT_CATEGORY, 0)])
while queue:
    category, depth = queue.popleft()
    if category in categories or depth > MAX_DEPTH:
        continue
    categories[category] = depth
    for member in category_members(category):
        if member["ns"] == 14 and depth < MAX_DEPTH:
            queue.append((member["title"], depth + 1))
        elif member["ns"] == 0:
            pages[member["pageid"]].add(category)

def parse_page(page_id):
    cache_path = PAGE_CACHE / f"{page_id}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = api_call(
        action="parse",
        pageid=page_id,
        prop="text|revid|displaytitle",
        disableeditsection="1",
    )["parse"]
    record = {
        "requested_pageid": page_id,
        "pageid": payload["pageid"],
        "title": payload["title"],
        "revid": payload["revid"],
        "categories": sorted(pages[page_id]),
        "html": payload["text"],
    }
    temporary = cache_path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return record


output = PARTIAL / "pages.jsonl"
page_ids = sorted(pages)
records = []
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(parse_page, page_id): page_id for page_id in page_ids}
    for index, future in enumerate(as_completed(futures), 1):
        records.append(future.result())
        if index % 100 == 0:
            print(f"Pages rendues : {index}/{len(page_ids)}")

metadata_by_revision = {}
revision_ids = sorted(record["revid"] for record in records)
for offset in range(0, len(revision_ids), 50):
    batch = revision_ids[offset : offset + 50]
    payload = api_call(
        prop="revisions",
        revids="|".join(map(str, batch)),
        rvprop="ids|timestamp|sha1",
    )
    for page in payload["query"]["pages"]:
        if page.get("missing"):
            continue
        revision = page["revisions"][0]
        metadata_by_revision[revision["revid"]] = revision

for record in records:
    revision = metadata_by_revision.get(record["revid"])
    if not revision:
        raise ValueError(f"Métadonnées de révision absentes : {record['title']}")
    record["parentid"] = revision.get("parentid")
    record["revision_timestamp"] = revision["timestamp"]
    record["revision_sha1"] = revision["sha1"]

records.sort(key=lambda row: row["pageid"])
pages_with_html = sum(bool(row["html"].strip()) for row in records)
if pages_with_html < int(len(records) * 0.90):
    raise ValueError(
        f"Snapshot incomplet : {pages_with_html}/{len(records)} pages avec HTML"
    )
output.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
    encoding="utf-8",
)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
manifest = {
    "snapshot_id": SNAPSHOT.name,
    "created_at": "2026-08-13",
    "source_id": "frwikibooks_mathematics",
    "api": API,
    "root_category": ROOT_CATEGORY,
    "maximum_category_depth": MAX_DEPTH,
    "categories_visited": len(categories),
    "pages_discovered": len(pages),
    "pages_snapshotted": len(records),
    "pages_with_html": pages_with_html,
    "rendering_method": "MediaWiki action=parse HTML at exact revision id",
    "pages_file": output.name,
    "pages_file_sha256": digest,
    "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0); GFDL alternative where applicable",
    "attribution": "Contributeurs de Wikilivres en français; attribution par historique de page",
    "category_depths": categories,
}
(PARTIAL / "snapshot_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PARTIAL.rename(SNAPSHOT)
print(f"Snapshot créé : {SNAPSHOT}")
print(f"Catégories    : {len(categories)}")
print(f"Pages         : {len(records)}")
print(f"SHA-256       : {digest}")
