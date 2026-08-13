import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import polib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import sha256_text, upsert_jsonl, write_json
from data.french_open import clean_open_french_text, valid_documentation_block


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/french_open_2026-08-13_v0.1"
SOURCE = SNAPSHOT / "python-docs-fr-3.14"
OUT_FILE = ROOT / "derived/open_french_v0.1/documentation/python_docs_fr_3_14_v0.1.txt"
ATTRIBUTION = ROOT / "derived/open_french_v0.1/attributions/python_docs_fr_3_14_v0.1.jsonl"
MANIFEST = ROOT / "manifests/structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports/python_docs_fr_v0.1_report.json"
EXPECTED_COMMIT = "a02a5710f906eb93fae4ce3a0a0cbed91243c5d9"


snapshot_manifest = json.loads((SNAPSHOT / "snapshot_manifest.json").read_text())
source_metadata = next(
    item for item in snapshot_manifest["sources"] if item["source_id"] == "python_docs_fr_3_14"
)
if source_metadata["commit"] != EXPECTED_COMMIT:
    raise ValueError("Commit python-docs-fr inattendu")

blocks = []
seen = set()
per_file = Counter()
translated_entries = 0
for source_file in sorted(SOURCE.rglob("*.po")):
    catalog = polib.pofile(str(source_file))
    relative = str(source_file.relative_to(SOURCE))
    for entry in catalog.translated_entries():
        translated_entries += 1
        translations = [entry.msgstr, *entry.msgstr_plural.values()]
        for translation in translations:
            text = clean_open_french_text(translation)
            key = text.casefold()
            if not valid_documentation_block(text) or key in seen:
                continue
            seen.add(key)
            blocks.append(text)
            per_file[relative] += 1

text = "\n".join(blocks) + "\n"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(text, encoding="utf-8")
ATTRIBUTION.parent.mkdir(parents=True, exist_ok=True)
ATTRIBUTION.write_text(
    "".join(
        json.dumps(
            {
                "source_file": source_file,
                "blocks": count,
                "repository": "https://github.com/python/python-docs-fr",
                "commit": EXPECTED_COMMIT,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for source_file, count in sorted(per_file.items())
    ),
    encoding="utf-8",
)

record = {
    "document_id": "python_docs_fr_3_14_v0.1",
    "source_id": "python_docs_fr_3_14",
    "group_id": "python_docs_fr_3_14",
    "title": "Documentation Python 3.14 traduite en français",
    "country_code": "FRANCOPHONE",
    "country_name": "Francophonie",
    "domain": "technical_documentation",
    "language": "fr",
    "content_type": "open_technical_documentation_translation",
    "rights_tier": "A_REDISTRIBUTABLE",
    "license": "Python Software Foundation License Version 2; French translation contributions under CC0 1.0",
    "license_url": "https://docs.python.org/fr/3.14/license.html",
    "dataset_url": "https://github.com/python/python-docs-fr/tree/3.14",
    "attribution": "Python Software Foundation and python-docs-fr contributors",
    "source_commit": EXPECTED_COMMIT,
    "source_files": len(list(SOURCE.rglob("*.po"))),
    "translated_entries": translated_entries,
    "usable_blocks": len(blocks),
    "atomic_facts": len(blocks),
    "generation_method": "deterministic_po_translation_extraction_and_rst_markup_cleanup",
    "output_path": str(OUT_FILE),
    "text_sha256": sha256_text(text),
    "attribution_path": str(ATTRIBUTION),
    "attribution_sha256": hashlib.sha256(ATTRIBUTION.read_bytes()).hexdigest(),
    "training_branch": "open_french_documentation",
}
upsert_jsonl(MANIFEST, record, key="document_id")
write_json(REPORT, record)
print("PYTHON DOCS FR v0.1 : OK")
print("Blocs retenus :", len(blocks))
print("Caractères    :", len(text))
print("SHA-256       :", sha256_text(text))
