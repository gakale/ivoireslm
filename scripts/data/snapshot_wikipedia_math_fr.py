import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/wikipedia_math_fr_2026-08-13_v0.1"
PARTIAL = SNAPSHOT.with_name(SNAPSHOT.name + ".partial")
API = "https://fr.wikipedia.org/w/api.php"
CATEGORIES = (
    "Catégorie:Algèbre",
    "Catégorie:Analyse (mathématiques)",
    "Catégorie:Géométrie",
    "Catégorie:Logique mathématique",
    "Catégorie:Probabilités",
    "Catégorie:Statistiques",
)
PAGES_PER_CATEGORY = 40
USER_AGENT = "IvoireSLM corpus builder/0.5 (https://github.com/gakale/ivoireslm)"
REQUEST_LOCK = threading.Lock()
LAST_REQUEST_AT = 0.0


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
                delay = 0.5 - (time.monotonic() - LAST_REQUEST_AT)
                if delay > 0:
                    time.sleep(delay)
                LAST_REQUEST_AT = time.monotonic()
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt == 7:
                raise
            time.sleep(max(float(error.headers.get("Retry-After") or 0), min(60, 2 ** attempt)))
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))


def category_pages(category):
    rows = []
    continuation = {}
    while True:
        payload = api_call(
            list="categorymembers",
            cmtitle=category,
            cmtype="page",
            cmnamespace="0",
            cmlimit="max",
            **continuation,
        )
        rows.extend(payload["query"]["categorymembers"])
        if "continue" not in payload:
            return rows
        continuation = {"cmcontinue": payload["continue"]["cmcontinue"]}


if SNAPSHOT.exists():
    if not (SNAPSHOT / "snapshot_manifest.json").is_file():
        raise SystemExit(f"Snapshot existant sans manifeste : {SNAPSHOT}")
    print(f"Snapshot déjà présent : {SNAPSHOT}")
    raise SystemExit(0)
PARTIAL.mkdir(parents=True, exist_ok=True)
CACHE = PARTIAL / "rendered_pages"
CACHE.mkdir(exist_ok=True)

discovered = {}
category_counts = {}
for category in CATEGORIES:
    members = category_pages(category)
    category_counts[category] = len(members)
    ranked = sorted(
        members,
        key=lambda row: hashlib.sha256(
            f"{category}\0{row['pageid']}\0{row['title']}".encode()
        ).digest(),
    )[:PAGES_PER_CATEGORY]
    for row in ranked:
        discovered.setdefault(
            row["pageid"], {"pageid": row["pageid"], "title": row["title"], "categories": []}
        )["categories"].append(category)


def parse_page(row):
    cache = CACHE / f"{row['pageid']}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    parsed = api_call(
        action="parse",
        pageid=row["pageid"],
        prop="text|revid|displaytitle",
        disableeditsection="1",
    )["parse"]
    record = {
        "pageid": parsed["pageid"],
        "title": parsed["title"],
        "revid": parsed["revid"],
        "categories": sorted(row["categories"]),
        "html": parsed["text"],
    }
    temporary = cache.with_suffix(".json.partial")
    temporary.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(cache)
    return record


records = []
rows = list(discovered.values())
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(parse_page, row) for row in rows]
    for index, future in enumerate(as_completed(futures), 1):
        records.append(future.result())
        if index % 50 == 0:
            print(f"Pages rendues : {index}/{len(rows)}")

metadata = {}
revision_ids = sorted(row["revid"] for row in records)
for offset in range(0, len(revision_ids), 50):
    payload = api_call(
        prop="revisions",
        revids="|".join(map(str, revision_ids[offset : offset + 50])),
        rvprop="ids|timestamp|sha1",
    )
    for page in payload["query"]["pages"]:
        revision = page["revisions"][0]
        metadata[revision["revid"]] = revision
for record in records:
    revision = metadata[record["revid"]]
    record.update(
        parentid=revision.get("parentid"),
        revision_timestamp=revision["timestamp"],
        revision_sha1=revision["sha1"],
    )

records.sort(key=lambda row: row["pageid"])
output = PARTIAL / "pages.jsonl"
output.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
    encoding="utf-8",
)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
manifest = {
    "snapshot_id": SNAPSHOT.name,
    "created_at": "2026-08-13",
    "source_id": "frwikipedia_mathematics_complement",
    "api": API,
    "categories": CATEGORIES,
    "category_page_counts": category_counts,
    "selection": f"lowest SHA-256 ranked {PAGES_PER_CATEGORY} direct namespace-0 pages per category",
    "unique_pages": len(records),
    "pages_file": output.name,
    "pages_file_sha256": digest,
    "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0); GFDL alternative where applicable",
    "attribution": "Contributeurs de Wikipédia en français; attribution par historique de page",
}
(PARTIAL / "snapshot_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PARTIAL.rename(SNAPSHOT)
print(f"Snapshot créé : {SNAPSHOT}")
print(f"Pages         : {len(records)}")
print(f"SHA-256       : {digest}")
