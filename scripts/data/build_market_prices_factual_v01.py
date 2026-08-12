import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import format_number_fr, sha256_text, upsert_jsonl, write_json


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots" / "ivoiredata_2026-08-10_v0.1" / "data"
SOURCE_ID = "civ_datagouv_market_prices"
DATASET_ID = "echo-du-marche"
SOURCE_ROOT = SNAPSHOT / "economy" / SOURCE_ID
OUT_FILE = (
    ROOT
    / "derived"
    / "structured_factual_v0.1"
    / "economy"
    / "civ_market_prices_2020_2022_v0.1.txt"
)
MANIFEST = ROOT / "manifests" / "structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports" / "market_prices_factual_v0.1_report.json"

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
CITY_DISPLAY = {
    "ABIDJAN": "Abidjan",
    "BOUAKE": "Bouaké",
    "KORHOGO": "Korhogo",
    "MAN": "Man",
    "SANPEDRO": "San Pedro",
    "YAMOUSSOUKRO": "Yamoussoukro",
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

business_columns = [
    "mois",
    "date",
    "annee",
    "categorie",
    "sous_categorie",
    "produits",
    "unite",
    "ville",
    "prix",
]
missing_columns = [column for column in business_columns if column not in df]
if missing_columns:
    raise ValueError(f"Colonnes manquantes : {missing_columns}")

work = df[business_columns].copy()
for column in business_columns[:-1]:
    work[column] = work[column].map(clean_text)

work["date_parsed"] = pd.to_datetime(work["date"], errors="coerce")
work["price_parsed"] = pd.to_numeric(
    work["prix"].astype("string").str.replace(",", ".", regex=False),
    errors="coerce",
)

invalid_dates = work[work["date_parsed"].isna()].index.tolist()
if invalid_dates:
    raise ValueError(f"Dates invalides : {len(invalid_dates)}")

year_mismatches = work[
    pd.to_numeric(work["annee"], errors="coerce")
    != work["date_parsed"].dt.year
]
if len(year_mismatches):
    raise ValueError(f"Années incohérentes : {len(year_mismatches)}")

month_mismatches = work[
    work.apply(
        lambda row: (row["mois"] or "").casefold()
        != MONTHS_FR[row["date_parsed"].month].casefold(),
        axis=1,
    )
]
if len(month_mismatches):
    raise ValueError(f"Mois incohérents : {len(month_mismatches)}")

unknown_cities = sorted(set(work["ville"].dropna()) - set(CITY_DISPLAY))
if unknown_cities:
    raise ValueError(f"Villes non mappées : {unknown_cities}")

missing_required_text = work[
    work[["produits", "unite", "ville"]].isna().any(axis=1)
]
if len(missing_required_text):
    raise ValueError(
        f"Champs texte requis absents : {len(missing_required_text)}"
    )

missing_price_count = int(work["price_parsed"].isna().sum())
nonpositive_price_count = int((work["price_parsed"] <= 0).sum())

valid = work[
    work["price_parsed"].notna() & (work["price_parsed"] > 0)
].copy()

fact_key = [
    "date",
    "categorie",
    "sous_categorie",
    "produits",
    "unite",
    "ville",
]
normalized_fact = [*fact_key, "price_parsed"]
exact_duplicate_rows = int(valid.duplicated(normalized_fact, keep=False).sum())
before_deduplication = len(valid)
valid = valid.drop_duplicates(normalized_fact, keep="first").copy()
exact_duplicates_removed = before_deduplication - len(valid)

price_counts = valid.groupby(fact_key, dropna=False)["price_parsed"].transform(
    "nunique"
)
conflicts = valid[price_counts > 1].copy()
conflicting_groups = int(
    conflicts[fact_key].drop_duplicates().shape[0]
)
conflicting_rows_excluded = int(len(conflicts))

conflict_sample = (
    conflicts[
        ["date", "produits", "unite", "ville", "price_parsed"]
    ]
    .sort_values(["date", "produits", "ville", "price_parsed"])
    .head(40)
    .rename(columns={"price_parsed": "price"})
    .to_dict("records")
)

usable = valid[price_counts == 1].copy()
usable = usable.sort_values(
    ["date_parsed", "ville", "produits", "unite", "price_parsed"],
    kind="stable",
).reset_index(drop=True)

lines = []
for row in usable.itertuples(index=False):
    city = CITY_DISPLAY[row.ville]
    price = format_number_fr(row.price_parsed, max_decimals=2)
    lines.append(
        f"Le {format_date_fr(row.date_parsed)}, à {city}, la source « Écho du "
        f"marché » indique un prix de {price} pour le produit « {row.produits} », "
        f"dont l'unité de vente est « {row.unite} »."
    )

text = "\n".join(lines).strip() + "\n"
OUT_FILE.write_text(text, encoding="utf-8")

document_id = "civ_market_prices_2020_2022_v0.1"
record = {
    "document_id": document_id,
    "source_id": SOURCE_ID,
    "dataset_id": DATASET_ID,
    "title": "Prix relevés dans Écho du marché de 2020 à 2022",
    "country_code": "CIV",
    "country_name": "Côte d'Ivoire",
    "domain": "economy",
    "secondary_domains": ["agriculture", "industry"],
    "language": "fr",
    "date_range": [
        usable["date_parsed"].min().date().isoformat(),
        usable["date_parsed"].max().date().isoformat(),
    ],
    "price_currency": "unspecified_in_source",
    "rights_tier": "A_REDISTRIBUTABLE",
    "source_access": "OPEN",
    "license": None,
    "content_type": "deterministic_structured_factual_text",
    "source_rows": int(len(df)),
    "usable_rows": int(len(usable)),
    "sentences_generated": len(lines),
    "atomic_facts": len(lines),
    "facts_review": 0,
    "excluded_rows": {
        "missing_or_non_numeric_price": missing_price_count,
        "nonpositive_price": nonpositive_price_count,
        "exact_duplicates_removed": exact_duplicates_removed,
        "conflicting_price_rows": conflicting_rows_excluded,
    },
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
    "valid_positive_price_rows_before_deduplication": int(
        before_deduplication
    ),
    "missing_or_non_numeric_price_rows": missing_price_count,
    "nonpositive_price_rows": nonpositive_price_count,
    "exact_duplicate_rows_involved": exact_duplicate_rows,
    "exact_duplicates_removed": exact_duplicates_removed,
    "conflicting_price_groups": conflicting_groups,
    "conflicting_price_rows_excluded": conflicting_rows_excluded,
    "conflict_sample": conflict_sample,
    "usable_rows": int(len(usable)),
    "cities": sorted(CITY_DISPLAY.values()),
    "products": int(usable["produits"].nunique()),
    "date_range": record["date_range"],
    "price_currency": "unspecified_in_source",
    "sentences_generated": len(lines),
    "atomic_facts": len(lines),
    "rights_tier": "A_REDISTRIBUTABLE",
    "source_access": "OPEN",
    "license": None,
    "output": str(OUT_FILE),
}
write_json(REPORT, report)

print("=" * 72)
print("IVOIRESLM — MARKET PRICES FACTUAL v0.1")
print("=" * 72)
print("Lignes source                  :", len(df))
print("Prix valides avant déduplication:", before_deduplication)
print("Prix absents/non numériques    :", missing_price_count)
print("Prix non positifs              :", nonpositive_price_count)
print("Doublons exacts supprimés      :", exact_duplicates_removed)
print("Groupes de prix contradictoires:", conflicting_groups)
print("Lignes contradictoires exclues :", conflicting_rows_excluded)
print("Phrases générées               :", len(lines))
print("Faits atomiques                :", len(lines))
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
