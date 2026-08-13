import hashlib
import json
import sys
import urllib.parse
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import sha256_text, upsert_jsonl, write_json
from data.corpus import EMAIL_RE, PHONE_RE
from data.mathematics import rendered_math_html_to_text


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/wikipedia_math_fr_2026-08-13_v0.1"
SOURCE = SNAPSHOT / "pages.jsonl"
WIKIBOOKS_REPORT = ROOT / "reports/frwikibooks_math_v0.1_report.json"
OUT_FILE = ROOT / "derived/open_french_v0.1/mathematics/frwikipedia_math_complement_v0.1.txt"
ATTRIBUTION = ROOT / "derived/open_french_v0.1/attributions/frwikipedia_math_complement_v0.1.jsonl"
MANIFEST = ROOT / "manifests/structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports/frwikipedia_math_complement_v0.1_report.json"
EXPECTED_SHA256 = "563d607dac58d611a58a458f5b2fdd218d704a73db16150789bdee72921a32cc"
BASE_CORPUS_CHARACTERS = 32_534_930
MATH_TARGET_CHARACTERS = round(BASE_CORPUS_CHARACTERS / 9)


if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA256:
    raise ValueError("Hash du snapshot Wikipédia invalide")
wikibooks = json.loads(WIKIBOOKS_REPORT.read_text(encoding="utf-8"))
wikibooks_text = Path(wikibooks["output_path"]).read_text(encoding="utf-8")
target_complement = max(0, MATH_TARGET_CHARACTERS - len(wikibooks_text))

candidates = []
excluded = Counter()
for line in SOURCE.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    text = rendered_math_html_to_text(row["html"])
    if len(text) < 1_000:
        excluded["short"] += 1
        continue
    if EMAIL_RE.search(text):
        excluded["email"] += 1
        continue
    if PHONE_RE.search(text):
        excluded["phone"] += 1
        continue
    score = hashlib.sha256(f"{row['pageid']}\0{row['revid']}\0{row['title']}".encode()).digest()
    candidates.append({**row, "clean_text": text, "selection_score": score})
candidates.sort(key=lambda row: row["selection_score"])

selected = []
selected_characters = 0
for row in candidates:
    chunk = f"Titre : {row['title']}\n{row['clean_text']}\n\n"
    selected.append({**row, "chunk": chunk})
    selected_characters += len(chunk)
    if selected_characters >= target_complement:
        break
if selected_characters < target_complement:
    raise ValueError(
        f"Complément mathématique insuffisant : {selected_characters}/{target_complement}"
    )
selected.sort(key=lambda row: row["title"].casefold())

chunks = []
attributions = []
position = 0
category_counts = Counter()
for row in selected:
    chunk = row["chunk"]
    chunks.append(chunk)
    article = "https://fr.wikipedia.org/wiki/" + urllib.parse.quote(
        row["title"].replace(" ", "_"), safe="/'()"
    )
    attributions.append(
        {
            "pageid": row["pageid"],
            "title": row["title"],
            "revid": row["revid"],
            "revision_sha1": row["revision_sha1"],
            "categories": row["categories"],
            "article_url": article,
            "permalink": article + f"?oldid={row['revid']}",
            "history_url": article + "?action=history",
            "output_start": position,
            "output_end": position + len(chunk),
        }
    )
    position += len(chunk)
    category_counts.update(row["categories"])
text = "".join(chunks).rstrip() + "\n"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(text, encoding="utf-8")
ATTRIBUTION.parent.mkdir(parents=True, exist_ok=True)
ATTRIBUTION.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in attributions),
    encoding="utf-8",
)
record = {
    "document_id": "frwikipedia_mathematics_complement_v0.1",
    "source_id": "frwikipedia_mathematics_complement",
    "group_id": "wikimedia_french_mathematics",
    "title": "Articles mathématiques complémentaires de Wikipédia en français",
    "country_code": "FRANCOPHONE",
    "country_name": "Francophonie",
    "domain": "mathematics",
    "language": "fr",
    "content_type": "open_encyclopedic_mathematics_with_latex",
    "rights_tier": "A_REDISTRIBUTABLE",
    "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0); GFDL alternative where applicable",
    "license_url": "https://foundation.wikimedia.org/wiki/Terms_of_Use/fr",
    "dataset_url": "https://fr.wikipedia.org/wiki/Catégorie:Mathématiques",
    "attribution": "Contributeurs de Wikipédia en français; permaliens et historiques par page dans le fichier d’attribution",
    "base_corpus_characters": BASE_CORPUS_CHARACTERS,
    "math_target_characters": MATH_TARGET_CHARACTERS,
    "wikibooks_math_characters": len(wikibooks_text),
    "target_complement_characters": target_complement,
    "source_pages": sum(1 for _ in SOURCE.open()),
    "eligible_pages": len(candidates),
    "selected_pages": len(selected),
    "selected_categories": dict(sorted(category_counts.items())),
    "excluded_pages": dict(sorted(excluded.items())),
    "atomic_facts": sum(len(row["clean_text"].splitlines()) for row in selected),
    "generation_method": "deterministic_sha256_ranked_rendered_html_cleanup_mathml_to_latex_target_share",
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
print("WIKIPÉDIA MATH FR v0.1 : OK")
print("Pages retenues :", len(selected))
print("Caractères     :", len(text))
print("Math total     :", len(text) + len(wikibooks_text))
print("Part théorique :", (len(text) + len(wikibooks_text)) / (BASE_CORPUS_CHARACTERS + len(text) + len(wikibooks_text)))
print("SHA-256        :", sha256_text(text))
