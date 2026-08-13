import hashlib
import json
import sys
import urllib.parse
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import sha256_text, upsert_jsonl, write_json
from data.corpus import EMAIL_RE, PHONE_RE, normalized_line_key
from data.mathematics import rendered_math_html_to_text


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/wikibooks_math_fr_2026-08-13_v0.3"
SOURCE = SNAPSHOT / "pages.jsonl"
OUT_FILE = ROOT / "derived/open_french_v0.1/mathematics/frwikibooks_math_v0.1.txt"
ATTRIBUTION = ROOT / "derived/open_french_v0.1/attributions/frwikibooks_math_v0.1.jsonl"
MANIFEST = ROOT / "manifests/structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports/frwikibooks_math_v0.1_report.json"
BASE_CORPUS_MANIFEST = ROOT / "corpora/ivoireslm_corpus_v0.4.0/manifests/documents.jsonl"
EXPECTED_SHA256 = "a113dd4b5bc5980119f607e21644f2f606d0e435cb3f12807b11d253ae873ef4"


if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA256:
    raise ValueError("Hash du snapshot Wikilivres invalide")

excluded_title_markers = ("/version imprimable", "/sommaire", "/index")
base_records = [json.loads(line) for line in BASE_CORPUS_MANIFEST.read_text().splitlines()]
seen_lines = {
    normalized_line_key(line)
    for record in base_records
    for line in Path(record["corpus_path"]).read_text(encoding="utf-8").splitlines()
    if normalized_line_key(line)
}
pages = []
excluded = {"short": 0, "email": 0, "phone": 0, "title": 0, "duplicate": 0}
seen = set()
for line in SOURCE.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    text = rendered_math_html_to_text(row["html"])
    if any(marker in row["title"].casefold() for marker in excluded_title_markers):
        excluded["title"] += 1
        continue
    if len(text) < 500:
        excluded["short"] += 1
        continue
    if EMAIL_RE.search(text):
        excluded["email"] += 1
        continue
    if PHONE_RE.search(text):
        excluded["phone"] += 1
        continue
    key = hashlib.sha256(text.casefold().encode()).hexdigest()
    if key in seen:
        excluded["duplicate"] += 1
        continue
    seen.add(key)
    pages.append({**row, "clean_text": text})
pages.sort(key=lambda row: row["title"].casefold())

duplicate_lines_excluded = 0
deduplicated_pages = []
for row in pages:
    unique_lines = []
    for line in row["clean_text"].splitlines():
        key = normalized_line_key(line)
        if not key or key in seen_lines:
            duplicate_lines_excluded += bool(key)
            continue
        seen_lines.add(key)
        unique_lines.append(line)
    unique_text = "\n".join(unique_lines).strip()
    if len(unique_text) >= 300:
        deduplicated_pages.append({**row, "clean_text": unique_text})
pages = deduplicated_pages

chunks = []
attributions = []
position = 0
for row in pages:
    chunk = f"Chapitre Wikilivres : {row['title']}\n{row['clean_text']}\n\n"
    chunks.append(chunk)
    article = "https://fr.wikibooks.org/wiki/" + urllib.parse.quote(
        row["title"].replace(" ", "_"), safe="/'()"
    )
    attributions.append(
        {
            "pageid": row["pageid"],
            "title": row["title"],
            "revid": row["revid"],
            "revision_sha1": row["revision_sha1"],
            "article_url": article,
            "permalink": article + f"?oldid={row['revid']}",
            "history_url": article + "?action=history",
            "output_start": position,
            "output_end": position + len(chunk),
        }
    )
    position += len(chunk)
text = "".join(chunks).rstrip() + "\n"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(text, encoding="utf-8")
ATTRIBUTION.parent.mkdir(parents=True, exist_ok=True)
ATTRIBUTION.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in attributions),
    encoding="utf-8",
)
record = {
    "document_id": "frwikibooks_mathematics_v0.1",
    "source_id": "frwikibooks_mathematics",
    "group_id": "wikimedia_french_mathematics",
    "title": "Cours et livres de mathématiques en français",
    "country_code": "FRANCOPHONE",
    "country_name": "Francophonie",
    "domain": "mathematics",
    "language": "fr",
    "content_type": "open_educational_mathematics_with_latex",
    "rights_tier": "A_REDISTRIBUTABLE",
    "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0); GFDL alternative where applicable",
    "license_url": "https://foundation.wikimedia.org/wiki/Terms_of_Use/fr",
    "dataset_url": "https://fr.wikibooks.org/wiki/Catégorie:Mathématiques",
    "attribution": "Contributeurs de Wikilivres en français; permaliens et historiques par page dans le fichier d’attribution",
    "source_pages": sum(1 for _ in SOURCE.open()),
    "usable_pages": len(pages),
    "excluded_pages": excluded,
    "duplicate_lines_excluded": duplicate_lines_excluded,
    "atomic_facts": sum(len(row["clean_text"].splitlines()) for row in pages),
    "generation_method": "deterministic_rendered_html_cleanup_mathml_to_latex",
    "source_file": str(SOURCE),
    "source_file_sha256": EXPECTED_SHA256,
    "output_path": str(OUT_FILE),
    "text_sha256": sha256_text(text),
    "attribution_path": str(ATTRIBUTION),
    "attribution_sha256": hashlib.sha256(ATTRIBUTION.read_bytes()).hexdigest(),
    "training_branch": "open_french_mathematics",
}
upsert_jsonl(MANIFEST, record, key="document_id")
write_json(REPORT, record)
print("WIKILIVRES MATH FR v0.1 : OK")
print("Pages retenues :", len(pages))
print("Lignes doublons :", duplicate_lines_excluded)
print("Caractères     :", len(text))
print("SHA-256        :", sha256_text(text))
