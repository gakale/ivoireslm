import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import format_number_fr, sha256_text, upsert_jsonl, write_json


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots" / "ivoiredata_2026-08-10_v0.1" / "data"
SOURCE_ID = "civ_datagouv_food_prices_2022"
DATASET_ID = "prix-hebdomadaires-des-produits-vivriers-de-cote-divoire"
SOURCE_ROOT = SNAPSHOT / "economy" / SOURCE_ID
OUT_FILE = (
    ROOT
    / "derived"
    / "structured_factual_v0.1"
    / "economy"
    / "civ_food_prices_2022_v0.1.txt"
)
MANIFEST = ROOT / "manifests" / "structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports" / "food_prices_2022_factual_v0.1_report.json"

MONTHS_FR = {
    1: "janvier",
    2: "février",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "août",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "décembre",
}
LOCATION_DISPLAY = {
    "ABIDJAN": "Abidjan",
    "BOUAKÉ": "Bouaké",
    "KORHOGO": "Korhogo",
    "MAN": "Man",
    "SAN-PÉDRO": "San-Pédro",
    "YAKRO": "Yamoussoukro",
}
AVAILABILITY_DISPLAY = {
    "bonne": "bonne",
    "moyen": "moyenne",
    "moyenne": "moyenne",
    "faible": "faible",
    "manque": "manquante",
    "rare": "rare",
}


def clean_text(value):
    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def format_date_fr(value):
    return f"{value.day} {MONTHS_FR[value.month]} {value.year}"


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
required = [
    "date",
    "cat_gorie",
    "produit",
    "prix_moyen",
    "lieu",
    "disponibilitx",
    "evolution_du_prix",
    "zone_de_disponibilitx",
]
missing_columns = [column for column in required if column not in df]
if missing_columns:
    raise ValueError(f"Colonnes manquantes : {missing_columns}")

work = df[required].copy()
for column in required:
    work[column] = work[column].map(clean_text)

work["date_parsed"] = pd.to_datetime(work["date"], errors="coerce")
work["price_parsed"] = pd.to_numeric(
    work["prix_moyen"].astype("string").str.replace(",", ".", regex=False),
    errors="coerce",
)
if work["date_parsed"].isna().any():
    raise ValueError(f"Dates invalides : {work['date_parsed'].isna().sum()}")
if set(work["date_parsed"].dt.year.unique()) != {2022}:
    raise ValueError("La source contient une année autre que 2022")

unknown_locations = sorted(
    set(work["lieu"].dropna()) - {"", *LOCATION_DISPLAY}
)
if unknown_locations:
    raise ValueError(f"Lieux non mappés : {unknown_locations}")

work["availability_normalized"] = (
    work["disponibilitx"].astype("string").str.casefold()
)
work["availability_safe"] = work["availability_normalized"].map(
    AVAILABILITY_DISPLAY
)

has_location = work["lieu"].notna() & work["lieu"].ne("")
has_product = work["produit"].notna() & work["produit"].ne("")
base_valid = has_location & has_product
has_price = base_valid & work["price_parsed"].notna() & (
    work["price_parsed"] > 0
)
has_availability = base_valid & work["availability_safe"].notna()
has_evolution = base_valid & work["evolution_du_prix"].notna() & (
    work["evolution_du_prix"] != ""
)
usable_mask = has_price | has_availability | has_evolution
usable = work[usable_mask].copy()

fact_key = ["date", "produit", "lieu"]
duplicate_keys = usable.duplicated(fact_key, keep=False)
if duplicate_keys.any():
    raise ValueError(
        f"Clés factuelles dupliquées : {int(duplicate_keys.sum())} lignes"
    )

usable = usable.sort_values(
    ["date_parsed", "lieu", "produit"], kind="stable"
).reset_index(drop=True)

lines = []
facts = []
for row in usable.itertuples(index=False):
    location = LOCATION_DISPLAY[row.lieu]
    attributes = []

    if pd.notna(row.price_parsed) and row.price_parsed > 0:
        price = format_number_fr(row.price_parsed, max_decimals=2)
        attributes.append(
            f"un prix moyen de {price}, sans devise précisée dans le fichier source"
        )
        facts.append(
            {
                "date": row.date,
                "product": row.produit,
                "location": location,
                "indicator": "average_price",
                "value": float(row.price_parsed),
                "currency": None,
            }
        )

    if row.availability_safe is not None and not pd.isna(
        row.availability_safe
    ):
        attributes.append(f"une disponibilité {row.availability_safe}")
        facts.append(
            {
                "date": row.date,
                "product": row.produit,
                "location": location,
                "indicator": "availability",
                "value": row.availability_safe,
            }
        )

    if row.evolution_du_prix is not None and not pd.isna(
        row.evolution_du_prix
    ):
        attributes.append(
            f"une évolution du prix décrite comme « {row.evolution_du_prix} »"
        )
        facts.append(
            {
                "date": row.date,
                "product": row.produit,
                "location": location,
                "indicator": "price_evolution_source_label",
                "value": row.evolution_du_prix,
            }
        )

    if not attributes:
        raise AssertionError("Observation sans attribut factuel")

    if len(attributes) == 1:
        attributes_text = attributes[0]
    else:
        attributes_text = ", ".join(attributes[:-1]) + " et " + attributes[-1]

    lines.append(
        f"Le {format_date_fr(row.date_parsed)}, à {location}, la source « Prix "
        f"hebdomadaires des produits vivriers » indique pour le produit "
        f"« {row.produit} » {attributes_text}."
    )

text = "\n".join(lines).strip() + "\n"
OUT_FILE.write_text(text, encoding="utf-8")

excluded_availability_labels = (
    work.loc[
        base_valid
        & work["disponibilitx"].notna()
        & work["availability_safe"].isna(),
        "disponibilitx",
    ]
    .value_counts()
    .to_dict()
)

document_id = "civ_food_prices_2022_v0.1"
record = {
    "document_id": document_id,
    "source_id": SOURCE_ID,
    "dataset_id": DATASET_ID,
    "title": "Prix hebdomadaires des produits vivriers en 2022",
    "country_code": "CIV",
    "country_name": "Côte d'Ivoire",
    "domain": "economy",
    "secondary_domains": ["agriculture"],
    "language": "fr",
    "year": 2022,
    "price_currency": "unspecified_in_source",
    "rights_tier": "A_REDISTRIBUTABLE",
    "source_access": "OPEN",
    "license": None,
    "content_type": "deterministic_structured_factual_text",
    "source_rows": int(len(df)),
    "usable_rows": int(len(usable)),
    "sentences_generated": len(lines),
    "atomic_facts": len(facts),
    "facts_review": 0,
    "generation_method": "python_deterministic_validated_templates",
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
    "rows_without_usable_facts": int((~usable_mask).sum()),
    "rows_without_location": int((~has_location).sum()),
    "numeric_price_facts": int(has_price.sum()),
    "safe_availability_facts": int(has_availability.sum()),
    "price_evolution_facts": int(has_evolution.sum()),
    "excluded_availability_labels": excluded_availability_labels,
    "price_currency": "unspecified_in_source",
    "sentences_generated": len(lines),
    "atomic_facts": len(facts),
    "rights_tier": "A_REDISTRIBUTABLE",
    "source_access": "OPEN",
    "license": None,
    "output": str(OUT_FILE),
}
write_json(REPORT, report)

print("=" * 72)
print("IVOIRESLM — FOOD PRICES 2022 FACTUAL v0.1")
print("=" * 72)
print("Lignes source              :", len(df))
print("Observations utilisables   :", len(usable))
print("Lignes sans fait utilisable:", int((~usable_mask).sum()))
print("Faits prix numériques      :", int(has_price.sum()))
print("Faits disponibilité sûrs   :", int(has_availability.sum()))
print("Faits évolution            :", int(has_evolution.sum()))
print("Phrases générées           :", len(lines))
print("Faits atomiques            :", len(facts))
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
