import hashlib
import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import sha256_text, upsert_jsonl, write_json
from data.faostat import render_faostat_production_sentence


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/ivoiredata_2026-08-13_growth_v0.2/data"
SOURCE_ROOT = SNAPSHOT / "agriculture/civ_faostat"
OUT_FILE = ROOT / "derived/structured_factual_v0.2/agriculture/civ_faostat_production_1961_2024_v0.1.txt"
MANIFEST = ROOT / "manifests/structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports/faostat_production_factual_v0.1_report.json"


def exactly_one(directory):
    files = list(directory.rglob("*.parquet"))
    if len(files) != 1:
        raise ValueError(f"Une table attendue dans {directory}, trouvé {len(files)}")
    return files[0]


manifest_source = SOURCE_ROOT / "manifest.json"
if not manifest_source.is_file():
    raise FileNotFoundError(manifest_source)
table_path = exactly_one(SOURCE_ROOT / "tables/data/faostat_production_crops_livestock")
source = pd.read_parquet(table_path)
required = [
    "area", "item_code", "item_code_cpcx", "item", "element_code", "element",
    "year", "unit", "value", "flag",
]
missing = [column for column in required if column not in source]
if missing:
    raise ValueError(f"Colonnes FAOSTAT manquantes : {missing}")
if len(source) != 17_541:
    raise ValueError(f"17 541 observations FAOSTAT attendues, trouvé {len(source)}")

numeric = pd.to_numeric(source["value"], errors="coerce")
usable = source[
    numeric.notna()
    & source["year"].astype(str).str.fullmatch(r"\d{4}")
    & source["area"].eq("Côte d'Ivoire")
].copy()
for column in required:
    if column == "value":
        continue
    usable = usable[usable[column].fillna("").astype(str).str.strip().ne("")]
usable["year_number"] = usable["year"].astype(int)
usable = usable[usable["year_number"].between(1961, 2024)].copy()
if len(usable) != 16_911:
    raise ValueError(f"16 911 observations numériques attendues, trouvé {len(usable)}")
key = ["area", "item", "element", "year", "unit"]
if usable.duplicated(key).any():
    raise ValueError("Clés FAOSTAT produit/élément/année/unité dupliquées")
if usable["item"].nunique() != 121 or usable["element"].nunique() != 8:
    raise ValueError("Cardinalités FAOSTAT inattendues")

usable = usable.sort_values(
    ["item_code", "element_code", "year_number"], kind="stable"
).reset_index(drop=True)
lines = [
    render_faostat_production_sentence(row, index)
    for index, row in usable.iterrows()
]
text = "\n".join(lines) + "\n"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(text, encoding="utf-8")

record = {
    "document_id": "civ_faostat_production_1961_2024_v0.1",
    "source_id": "civ_faostat_production_crops_livestock",
    "group_id": "faostat_production_civ",
    "title": "Production végétale et animale de la Côte d’Ivoire dans FAOSTAT",
    "country_code": "CIV",
    "country_name": "Côte d’Ivoire",
    "domain": "agriculture",
    "language": "fr-en",
    "content_type": "deterministic_structured_factual_text",
    "rights_tier": "A_REDISTRIBUTABLE",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0), complétée par les conditions FAO des bases statistiques",
    "attribution": "FAO. 2026. FAOSTAT: Production: Crops and livestock products. Accessed 10 August 2026. Licence: CC-BY-4.0.",
    "license_url": "https://www.fao.org/contact-us/terms/db-terms-of-use/en",
    "dataset_url": "https://www.fao.org/faostat/en/#data/QCL",
    "observed_year_range": [1961, 2024],
    "source_rows": int(len(source)),
    "usable_rows": int(len(usable)),
    "excluded_missing_or_invalid_rows": int(len(source) - len(usable)),
    "items_observed": int(usable["item"].nunique()),
    "elements_observed": int(usable["element"].nunique()),
    "sentences_generated": len(lines),
    "atomic_facts": len(lines),
    "generation_method": "python_deterministic_source_label_and_code_preserving_templates",
    "source_table": str(table_path.relative_to(SNAPSHOT)),
    "source_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
    "output_path": str(OUT_FILE),
    "text_sha256": sha256_text(text),
    "training_branch": "derived_structured_factual",
}
upsert_jsonl(MANIFEST, record, key="document_id")
write_json(REPORT, record)
print("FAOSTAT PRODUCTION FACTUAL v0.1 : OK")
print("Observations source :", len(source))
print("Observations utiles :", len(usable))
print("Produits            :", usable["item"].nunique())
print("Éléments            :", usable["element"].nunique())
print("Document            :", OUT_FILE)
print("Caractères          :", len(text))
print("SHA-256             :", sha256_text(text))
