import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import format_number_fr, sha256_text, upsert_jsonl, write_json


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots" / "ivoiredata_2026-08-10_v0.1" / "data"
SOURCE_ID = "civ_datagouv_rgph2021"
DATASET_ID = "recensement-de-la-population-ivoirienne"
SOURCE_ROOT = SNAPSHOT / "demography" / SOURCE_ID
OUT_FILE = (
    ROOT
    / "derived"
    / "structured_factual_v0.1"
    / "demography"
    / "civ_rgph2021_population_households_v0.1.txt"
)
MANIFEST = ROOT / "manifests" / "structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports" / "rgph2021_factual_v0.1_report.json"

EXPECTED_CATEGORIES = {"HOMME", "FEMME", "MENAGE"}
EXPECTED_NATIONAL = {
    "men": 15_344_989,
    "women": 14_044_161,
    "population": 29_389_150,
    "households": 5_616_487,
}


def clean_text(value):
    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def fmt_count(value):
    return format_number_fr(int(value), max_decimals=0)


def add_sentence(lines, facts, scope, label, context, values):
    men = int(values["HOMME"])
    women = int(values["FEMME"])
    households = int(values["MENAGE"])
    population = men + women

    lines.append(
        f"Selon le RGPH 2021, {context} compte {fmt_count(population)} habitants, "
        f"dont {fmt_count(men)} hommes et {fmt_count(women)} femmes, ainsi que "
        f"{fmt_count(households)} ménages."
    )
    facts.extend(
        [
            {
                "scope": scope,
                "label": label,
                "indicator": "population_total",
                "value": population,
                "derivation": "men_plus_women",
            },
            {
                "scope": scope,
                "label": label,
                "indicator": "men",
                "value": men,
                "derivation": "source_sum",
            },
            {
                "scope": scope,
                "label": label,
                "indicator": "women",
                "value": women,
                "derivation": "source_sum",
            },
            {
                "scope": scope,
                "label": label,
                "indicator": "households",
                "value": households,
                "derivation": "source_sum",
            },
        ]
    )


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
geography = [
    "district",
    "region",
    "departement",
    "sous_prefecture_ou_commune",
]
required = [*geography, "categorie", "effectif"]
missing_columns = [column for column in required if column not in df]
if missing_columns:
    raise ValueError(f"Colonnes manquantes : {missing_columns}")

work = df[required].copy()
for column in [*geography, "categorie"]:
    work[column] = work[column].map(clean_text)
if work[[*geography, "categorie"]].isna().any().any():
    raise ValueError("Valeurs géographiques ou catégories absentes")

work["effectif_parsed"] = pd.to_numeric(work["effectif"], errors="coerce")
if work["effectif_parsed"].isna().any():
    raise ValueError(f"Effectifs invalides : {work['effectif_parsed'].isna().sum()}")
if (work["effectif_parsed"] <= 0).any():
    raise ValueError("Tous les effectifs doivent être strictement positifs")
if set(work["categorie"].unique()) != EXPECTED_CATEGORIES:
    raise ValueError(f"Catégories inattendues : {work['categorie'].unique()}")

category_key = [*geography, "categorie"]
duplicate_rows = work.duplicated(category_key, keep=False)
if duplicate_rows.any():
    raise ValueError(f"Clés dupliquées : {int(duplicate_rows.sum())} lignes")

local = work.pivot(
    index=geography,
    columns="categorie",
    values="effectif_parsed",
).reset_index()
if local[list(EXPECTED_CATEGORIES)].isna().any().any():
    raise ValueError("Une catégorie manque pour au moins une localité")

for column in EXPECTED_CATEGORIES:
    local[column] = local[column].astype(int)
local["POPULATION"] = local["HOMME"] + local["FEMME"]

counts = {
    "districts": int(local["district"].nunique()),
    "regions": int(local["region"].nunique()),
    "departments": int(local["departement"].nunique()),
    "localities": int(len(local)),
}
expected_counts = {
    "districts": 14,
    "regions": 33,
    "departments": 111,
    "localities": 519,
}
if counts != expected_counts:
    raise ValueError(f"Comptages territoriaux inattendus : {counts}")

national = {
    "HOMME": int(local["HOMME"].sum()),
    "FEMME": int(local["FEMME"].sum()),
    "MENAGE": int(local["MENAGE"].sum()),
}
national_check = {
    "men": national["HOMME"],
    "women": national["FEMME"],
    "population": national["HOMME"] + national["FEMME"],
    "households": national["MENAGE"],
}
if national_check != EXPECTED_NATIONAL:
    raise ValueError(f"Totaux nationaux inattendus : {national_check}")

lines = []
facts = []
add_sentence(
    lines,
    facts,
    "national",
    "Côte d'Ivoire",
    "la Côte d'Ivoire",
    national,
)

districts = (
    local.groupby(["district"], as_index=False)[["HOMME", "FEMME", "MENAGE"]]
    .sum()
    .sort_values("district")
)
for row in districts.itertuples(index=False):
    values = {"HOMME": row.HOMME, "FEMME": row.FEMME, "MENAGE": row.MENAGE}
    add_sentence(
        lines,
        facts,
        "district",
        row.district,
        f"le district « {row.district} »",
        values,
    )

regions = (
    local.groupby(["district", "region"], as_index=False)[
        ["HOMME", "FEMME", "MENAGE"]
    ]
    .sum()
    .sort_values(["district", "region"])
)
for row in regions.itertuples(index=False):
    values = {"HOMME": row.HOMME, "FEMME": row.FEMME, "MENAGE": row.MENAGE}
    add_sentence(
        lines,
        facts,
        "region",
        row.region,
        f"la région « {row.region} », dans le district « {row.district} »",
        values,
    )

departments = (
    local.groupby(["district", "region", "departement"], as_index=False)[
        ["HOMME", "FEMME", "MENAGE"]
    ]
    .sum()
    .sort_values(["district", "region", "departement"])
)
for row in departments.itertuples(index=False):
    values = {"HOMME": row.HOMME, "FEMME": row.FEMME, "MENAGE": row.MENAGE}
    add_sentence(
        lines,
        facts,
        "department",
        row.departement,
        f"le département « {row.departement} », dans la région « {row.region} »",
        values,
    )

local = local.sort_values(geography)
for row in local.itertuples(index=False):
    values = {"HOMME": row.HOMME, "FEMME": row.FEMME, "MENAGE": row.MENAGE}
    add_sentence(
        lines,
        facts,
        "locality",
        row.sous_prefecture_ou_commune,
        (
            f"la localité « {row.sous_prefecture_ou_commune} », rattachée au "
            f"département « {row.departement} »"
        ),
        values,
    )

text = "\n".join(lines).strip() + "\n"
OUT_FILE.write_text(text, encoding="utf-8")

document_id = "civ_rgph2021_population_households_v0.1"
record = {
    "document_id": document_id,
    "source_id": SOURCE_ID,
    "dataset_id": DATASET_ID,
    "title": "Population et ménages du RGPH 2021 par niveau territorial",
    "country_code": "CIV",
    "country_name": "Côte d'Ivoire",
    "domain": "demography",
    "language": "fr",
    "year": 2021,
    "units": {"population": "persons", "households": "households"},
    "rights_tier": "A_REDISTRIBUTABLE",
    "source_access": "OPEN",
    "license": None,
    "content_type": "deterministic_structured_factual_text",
    "source_rows": int(len(df)),
    **counts,
    "sentences_generated": len(lines),
    "atomic_facts": len(facts),
    "facts_review": 0,
    "generation_method": "python_deterministic_validated_territorial_aggregation",
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
    "categories": sorted(EXPECTED_CATEGORIES),
    "territorial_counts": counts,
    "national_checks": national_check,
    "expected_national": EXPECTED_NATIONAL,
    "national_valid": national_check == EXPECTED_NATIONAL,
    "sentences_generated": len(lines),
    "atomic_facts": len(facts),
    "rights_tier": "A_REDISTRIBUTABLE",
    "source_access": "OPEN",
    "license": None,
    "output": str(OUT_FILE),
}
write_json(REPORT, report)

print("=" * 72)
print("IVOIRESLM — RGPH 2021 FACTUAL v0.1")
print("=" * 72)
print("Lignes source       :", len(df))
print("Districts           :", counts["districts"])
print("Régions             :", counts["regions"])
print("Départements        :", counts["departments"])
print("Localités           :", counts["localities"])
print("Population nationale:", fmt_count(national_check["population"]))
print("Ménages nationaux   :", fmt_count(national_check["households"]))
print("Phrases générées    :", len(lines))
print("Faits atomiques     :", len(facts))
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
