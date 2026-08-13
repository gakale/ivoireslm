import gzip
import hashlib
import heapq
import json
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import sha256_text, upsert_jsonl, write_json
from data.french_open import dictionary_definitions, render_dictionary_entry


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/french_open_2026-08-13_v0.1"
SOURCE = SNAPSHOT / "frwiktionary_kaikki_2026-08-11.jsonl.gz"
OUT_FILE = ROOT / "derived/open_french_v0.1/dictionary/frwiktionary_definitions_v0.1.txt"
ATTRIBUTION = ROOT / "derived/open_french_v0.1/attributions/frwiktionary_definitions_v0.1.jsonl"
MANIFEST = ROOT / "manifests/structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports/frwiktionary_definitions_v0.1_report.json"
EXPECTED_SHA256 = "65dc1e86e912e28317060bd901c988944cf31c9447d39e0b3358bb3dc23ce6ad"

QUOTAS = {
    "noun": 20_000,
    "adj": 9_000,
    "verb": 9_000,
    "name": 2_500,
    "adv": 3_000,
    "phrase": 2_000,
    "intj": 1_000,
    "prep": 500,
    "conj": 250,
    "pron": 500,
    "det": 75,
    "article": 38,
    "proverb": 27,
    "prefix": 500,
    "suffix": 500,
    "onomatopoeia": 250,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if file_sha256(SOURCE) != EXPECTED_SHA256:
    raise ValueError("Hash du snapshot Wiktionnaire invalide")

heaps = defaultdict(list)
source_counts = Counter()
eligible_counts = Counter()
with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
    for line in handle:
        source_counts["rows"] += 1
        entry = json.loads(line)
        pos = entry.get("pos")
        if pos not in QUOTAS or entry.get("lang_code") != "fr":
            continue
        word = str(entry.get("word") or "").strip()
        definitions = dictionary_definitions(entry)
        if not (2 <= len(word) <= 80) or not definitions:
            continue
        eligible_counts[pos] += 1
        canonical = json.dumps(
            [word, pos, definitions, entry.get("etymology_texts") or []],
            ensure_ascii=False,
            sort_keys=True,
        )
        score = int.from_bytes(hashlib.sha256(canonical.encode()).digest(), "big")
        item = (-score, canonical, source_counts["rows"], entry)
        heap = heaps[pos]
        if len(heap) < QUOTAS[pos]:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)

selected = []
for pos in sorted(heaps):
    selected.extend(item[3] for item in heaps[pos])
selected.sort(
    key=lambda entry: (
        str(entry.get("word", "")).casefold(),
        str(entry.get("pos", "")),
        json.dumps(dictionary_definitions(entry), ensure_ascii=False),
    )
)

lines = []
attributions = []
for entry in selected:
    lines.append(render_dictionary_entry(entry))
    title = str(entry["word"])
    article_url = "https://fr.wiktionary.org/wiki/" + urllib.parse.quote(
        title.replace(" ", "_"), safe="()'"
    )
    attributions.append(
        {
            "word": title,
            "part_of_speech": entry.get("pos"),
            "article_url": article_url,
            "history_url": article_url + "?action=history",
            "upstream_dump_date": "2026-08-04",
        }
    )

text = "\n".join(lines) + "\n"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(text, encoding="utf-8")
ATTRIBUTION.parent.mkdir(parents=True, exist_ok=True)
ATTRIBUTION.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in attributions),
    encoding="utf-8",
)

selected_by_pos = Counter(entry["pos"] for entry in selected)
record = {
    "document_id": "frwiktionary_definitions_v0.1",
    "source_id": "frwiktionary_kaikki_french",
    "group_id": "frwiktionary_french_definitions",
    "title": "Définitions françaises sélectionnées du Wiktionnaire",
    "country_code": "FRANCOPHONE",
    "country_name": "Francophonie",
    "domain": "dictionary_lexicography",
    "language": "fr",
    "content_type": "open_dictionary_definitions_without_quotations",
    "rights_tier": "A_REDISTRIBUTABLE",
    "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0); GFDL alternative",
    "license_url": "https://foundation.wikimedia.org/wiki/Terms_of_Use/fr",
    "dataset_url": "https://kaikki.org/frwiktionary/Fran%C3%A7ais/index.html",
    "attribution": "Contributeurs du Wiktionnaire français; extraction Wiktextract/kaikki.org; URLs par entrée dans le fichier d’attribution",
    "upstream_dump_date": "2026-08-04",
    "source_rows": source_counts["rows"],
    "eligible_by_pos": dict(sorted(eligible_counts.items())),
    "selected_by_pos": dict(sorted(selected_by_pos.items())),
    "selected_entries": len(selected),
    "sentences_generated": len(lines),
    "atomic_facts": sum(len(dictionary_definitions(entry)) for entry in selected),
    "generation_method": "deterministic_sha256_ranked_pos_balanced_definition_selection_no_examples",
    "source_file": str(SOURCE),
    "source_file_sha256": EXPECTED_SHA256,
    "output_path": str(OUT_FILE),
    "text_sha256": sha256_text(text),
    "attribution_path": str(ATTRIBUTION),
    "attribution_sha256": hashlib.sha256(ATTRIBUTION.read_bytes()).hexdigest(),
    "training_branch": "open_french_dictionary",
}
upsert_jsonl(MANIFEST, record, key="document_id")
write_json(REPORT, record)
print("WIKTIONNAIRE FR v0.1 : OK")
print("Entrées source   :", source_counts["rows"])
print("Entrées retenues :", len(selected))
print("Faits            :", record["atomic_facts"])
print("Caractères       :", len(text))
print("SHA-256          :", sha256_text(text))
