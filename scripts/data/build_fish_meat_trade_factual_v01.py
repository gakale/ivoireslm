import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import format_number_fr, sha256_text, upsert_jsonl, write_json


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots" / "ivoiredata_2026-08-10_v0.1" / "data"
SOURCE_ID = "civ_datagouv_fish_meat_trade"
DATASET_ID = (
    "chiffres-sur-la-production-les-importations-et-exportations-"
    "de-poissons-et-de-viande"
)
SOURCE_ROOT = SNAPSHOT / "agriculture" / SOURCE_ID
OUT_FILE = (
    ROOT
    / "derived"
    / "structured_factual_v0.1"
    / "agriculture"
    / "civ_fish_meat_trade_1999_2014_v0.1.txt"
)
MANIFEST = ROOT / "manifests" / "structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports" / "fish_meat_trade_factual_v0.1_report.json"

STATUS_RULES = {
    "Quantité (en milliers de Tonne)": (
        "quantity_thousand_tonnes",
        "une quantité de {value} milliers de tonnes",
    ),
    "Quantité (en Tonne)": (
        "quantity_tonnes",
        "une quantité de {value} tonnes",
    ),
    "Quantité en Tonne": (
        "quantity_tonnes",
        "une quantité de {value} tonnes",
    ),
    "Quantité (en tonnes)": (
        "quantity_tonnes",
        "une quantité de {value} tonnes",
    ),
    "Quantité ( en tonnes)": (
        "quantity_tonnes",
        "une quantité de {value} tonnes",
    ),
    "Quantité en Tonne Equivalent-Carcasse (TEC)": (
        "quantity_tonne_carcass_equivalent",
        "une quantité de {value} tonnes équivalent-carcasse",
    ),
    "Quantité en Tonne Equivalent-Litre (TEL)": (
        "quantity_tonne_litre_equivalent",
        "une quantité de {value} tonnes équivalent-litre",
    ),
    "Prix moyen (en f cfa/Kilo)": (
        "average_price_fcfa_per_kg",
        "un prix moyen de {value} FCFA par kilogramme",
    ),
    "Prix moyen (en f cfa/kilo)": (
        "average_price_fcfa_per_kg",
        "un prix moyen de {value} FCFA par kilogramme",
    ),
    "Valeur (en millions de F. CFA)": (
        "value_million_fcfa",
        "une valeur de {value} millions de FCFA",
    ),
    "Valeur (en millions de f. cfa)": (
        "value_million_fcfa",
        "une valeur de {value} millions de FCFA",
    ),
    "Valeur CAF (en millions de f.cfa)": (
        "caf_value_million_fcfa",
        "une valeur CAF de {value} millions de FCFA",
    ),
}


def clean_text(value):
    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    return text or None


OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

files = [
    path
    for path in SOURCE_ROOT.rglob("*.parquet")
    if "datagouv_catalog" not in str(path)
]
if len(files) != 1:
    raise SystemExit(f"1 table métier attendue, trouvé {len(files)}")

source_path = files[0]
df = pd.read_parquet(source_path).copy()
required = ["ann_e", "cat_gorie", "sous_cat_gorie", "type", "statut", "valeur"]
missing_columns = [column for column in required if column not in df]
if missing_columns:
    raise ValueError(f"Colonnes manquantes : {missing_columns}")

work = df[required].copy()
for column in required[:-1]:
    work[column] = work[column].map(clean_text)
work["value_parsed"] = pd.to_numeric(
    work["valeur"].astype("string").str.strip().str.replace(",", ".", regex=False),
    errors="coerce",
)

valid_year = work["ann_e"].fillna("").str.fullmatch(r"[0-9]{4}")
explicit_status = work["statut"].isin(STATUS_RULES)
numeric_value = work["value_parsed"].notna()
positive_value = numeric_value & (work["value_parsed"] > 0)
required_labels = (
    work["cat_gorie"].notna()
    & work["sous_cat_gorie"].notna()
    & work["type"].notna()
)
usable_mask = (
    valid_year & explicit_status & positive_value & required_labels
)
usable = work[usable_mask].copy()
usable["year"] = usable["ann_e"].astype(int)

fact_key = ["ann_e", "cat_gorie", "sous_cat_gorie", "type", "statut"]
duplicate_keys = usable.duplicated(fact_key, keep=False)
if duplicate_keys.any():
    raise ValueError(f"Clés sûres dupliquées : {int(duplicate_keys.sum())}")

if len(usable) != 770:
    raise ValueError(f"770 lignes sûres attendues, trouvé {len(usable)}")
if usable["year"].min() != 1999 or usable["year"].max() != 2014:
    raise ValueError("Période sûre inattendue")

usable = usable.sort_values(
    ["year", "cat_gorie", "sous_cat_gorie", "type", "statut"],
    kind="stable",
).reset_index(drop=True)

lines = []
facts = []
for row in usable.itertuples(index=False):
    indicator, value_template = STATUS_RULES[row.statut]
    value = format_number_fr(row.value_parsed, max_decimals=3)
    formatted_value = value_template.format(value=value)
    lines.append(
        f"En {row.year}, dans la catégorie « {row.cat_gorie} » et la "
        f"sous-catégorie « {row.sous_cat_gorie} », le type « {row.type} » "
        f"présente {formatted_value}."
    )
    facts.append(
        {
            "year": row.year,
            "category": row.cat_gorie,
            "subcategory": row.sous_cat_gorie,
            "type": row.type,
            "indicator": indicator,
            "value": float(row.value_parsed),
            "source_status": row.statut,
        }
    )

text = "\n".join(lines).strip() + "\n"
OUT_FILE.write_text(text, encoding="utf-8")

document_id = "civ_fish_meat_trade_1999_2014_v0.1"
category_counts = usable["cat_gorie"].value_counts().sort_index().to_dict()
status_counts = usable["statut"].value_counts().sort_index().to_dict()
record = {
    "document_id": document_id,
    "source_id": SOURCE_ID,
    "dataset_id": DATASET_ID,
    "title": "Production, importations et exportations de poissons et viande",
    "country_code": "CIV",
    "country_name": "Côte d'Ivoire",
    "domain": "agriculture",
    "secondary_domains": ["economy", "trade"],
    "language": "fr",
    "observed_year_range": [1999, 2014],
    "rights_tier": "A_REDISTRIBUTABLE",
    "source_access": "OPEN",
    "license": None,
    "content_type": "deterministic_structured_factual_text",
    "source_rows": int(len(df)),
    "usable_rows": int(len(usable)),
    "sentences_generated": len(lines),
    "atomic_facts": len(facts),
    "facts_review": 0,
    "generation_method": "python_deterministic_explicit_unit_rules",
    "source_table": str(source_path.relative_to(SNAPSHOT)),
    "output_path": str(OUT_FILE),
    "text_sha256": sha256_text(text),
    "group_id": SOURCE_ID,
    "training_branch": "derived_structured_factual",
}
upsert_jsonl(MANIFEST, record, key="document_id")

report = {
    "source_id": SOURCE_ID,
    "dataset_id": DATASET_ID,
    "rows_source": int(len(df)),
    "usable_rows": int(len(usable)),
    "excluded_rows": int((~usable_mask).sum()),
    "rows_with_invalid_or_blank_year": int((~valid_year).sum()),
    "rows_with_ambiguous_status": int((~explicit_status).sum()),
    "rows_with_missing_or_non_numeric_value": int((~numeric_value).sum()),
    "rows_with_nonpositive_value": int((numeric_value & ~positive_value).sum()),
    "rows_with_missing_required_labels": int((~required_labels).sum()),
    "category_counts": category_counts,
    "status_counts": status_counts,
    "sentences_generated": len(lines),
    "atomic_facts": len(facts),
    "rights_tier": "A_REDISTRIBUTABLE",
    "source_access": "OPEN",
    "license": None,
    "output": str(OUT_FILE),
}
write_json(REPORT, report)

print("=" * 72)
print("IVOIRESLM — FISH AND MEAT TRADE FACTUAL v0.1")
print("=" * 72)
print("Lignes source                :", len(df))
print("Lignes sûres                 :", len(usable))
print("Lignes exclues               :", int((~usable_mask).sum()))
print("Années                       :", "1999–2014")
print("Production                   :", category_counts.get("Production", 0))
print("Importation                  :", category_counts.get("Importation", 0))
print("Exportation                  :", category_counts.get("Exportation", 0))
print("Phrases générées             :", len(lines))
print("Faits atomiques              :", len(facts))
print()
print("=== APERÇU ===")
for line in lines[:5]:
    print("-", line)
print("...")
for line in lines[-3:]:
    print("-", line)
print()
print("Document :", OUT_FILE)
print("Manifest :", MANIFEST)
print("Rapport  :", REPORT)
