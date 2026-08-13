import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
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


def api_call(**parameters):
    parameters.setdefault("action", "query")
    parameters.update(format="json", formatversion="2")
    request = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(parameters),
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


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
if PARTIAL.exists():
    shutil.rmtree(PARTIAL)
PARTIAL.mkdir(parents=True)

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
    payload = api_call(
        action="parse",
        pageid=page_id,
        prop="text|revid|displaytitle",
        disableeditsection="1",
    )["parse"]
    return {
        "requested_pageid": page_id,
        "pageid": payload["pageid"],
        "title": payload["title"],
        "revid": payload["revid"],
        "categories": sorted(pages[page_id]),
        "html": payload["text"],
    }


output = PARTIAL / "pages.jsonl"
page_ids = sorted(pages)
records = []
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(parse_page, page_id): page_id for page_id in page_ids}
    for index, future in enumerate(as_completed(futures), 1):
        records.append(future.result())
        if index % 100 == 0:
            print(f"Pages rendues : {index}/{len(page_ids)}")

metadata_by_page = {}
for offset in range(0, len(page_ids), 50):
    batch = page_ids[offset : offset + 50]
    payload = api_call(
        prop="revisions",
        pageids="|".join(map(str, batch)),
        rvprop="ids|timestamp|sha1",
    )
    for page in payload["query"]["pages"]:
        if page.get("missing"):
            continue
        revision = page["revisions"][0]
        metadata_by_page[page["pageid"]] = revision

for record in records:
    revision = metadata_by_page.get(record["requested_pageid"])
    if not revision or revision["revid"] != record["revid"]:
        raise ValueError(f"Révision instable pendant le snapshot : {record['title']}")
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
