import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import format_number_fr, sha256_text, upsert_jsonl, write_json


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots" / "ivoiredata_2026-08-10_v0.1" / "data"
SOURCE_ID = "civ_datagouv_rainfall_stations"
DATASET_ID = "2nn5o0u6-12xfz369yje40vp"
SOURCE_ROOT = SNAPSHOT / "environment_climate" / SOURCE_ID
OUT_FILE = (
    ROOT
    / "derived"
    / "structured_factual_v0.1"
    / "environment_climate"
    / "civ_rainfall_stations_2022_2023_v0.1.txt"
)
MANIFEST = ROOT / "manifests" / "structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports" / "rainfall_stations_factual_v0.1_report.json"

STATIONS = {
    "abidjan": "Abidjan",
    "adiakx": "Adiaké",
    "bondoukou": "Bondoukou",
    "bouakx": "Bouaké",
    "korhogo": "Korhogo",
    "man": "Man",
    "san_pedro": "San Pedro",
}
YEARS = (2022, 2023)
VARIATION_LABEL = "Variation 2023/2022"
VARIATION_TOLERANCE_PERCENTAGE_POINTS = 1e-9


def parse_number(value):
    if pd.isna(value):
        raise ValueError("Valeur numérique absente dans la source")
    text = str(value).strip().replace(" ", "").replace(",", ".")
    if not text:
        raise ValueError("Valeur numérique vide dans la source")
    return float(text)


def fmt_mm(value):
    return format_number_fr(value, max_decimals=1)


def fmt_percent(value):
    return format_number_fr(value, max_decimals=2)


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

required = ["station", *STATIONS]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes : {missing}")
if len(df) != 3:
    raise ValueError(f"3 lignes attendues, trouvé {len(df)}")

labels = df["station"].astype(str).str.strip()
if labels.duplicated().any():
    raise ValueError("Libellés de période dupliqués")

rows = {label: row for label, (_, row) in zip(labels, df.iterrows())}
expected_labels = {str(year) for year in YEARS} | {VARIATION_LABEL}
if set(rows) != expected_labels:
    raise ValueError(
        f"Périodes inattendues : attendu {sorted(expected_labels)}, "
        f"trouvé {sorted(rows)}"
    )

values = {}
variation_checks = []
issues = []

for column, station in STATIONS.items():
    value_2022 = parse_number(rows["2022"][column])
    value_2023 = parse_number(rows["2023"][column])
    source_variation = parse_number(rows[VARIATION_LABEL][column])
    if value_2022 <= 0 or value_2023 < 0:
        issues.append({"station": station, "reason": "invalid_rainfall_value"})
        continue

    calculated_variation = ((value_2023 - value_2022) / value_2022) * 100
    difference = source_variation - calculated_variation
    valid = abs(difference) <= VARIATION_TOLERANCE_PERCENTAGE_POINTS
    variation_checks.append(
        {
            "station": station,
            "rainfall_2022_mm": value_2022,
            "rainfall_2023_mm": value_2023,
            "source_variation_percent": source_variation,
            "calculated_variation_percent": calculated_variation,
            "difference_percentage_points": difference,
            "valid": valid,
        }
    )
    if not valid:
        issues.append(
            {
                "station": station,
                "reason": "published_variation_mismatch",
                "difference_percentage_points": difference,
            }
        )
    values[station] = {
        2022: value_2022,
        2023: value_2023,
        "variation": source_variation,
    }

if issues:
    raise ValueError(f"Problèmes bloquants détectés : {issues}")

lines = []
facts = []
for station, station_values in values.items():
    for year in YEARS:
        value = station_values[year]
        lines.append(
            f"En {year}, la pluviométrie annuelle mesurée à la station de "
            f"{station} est de {fmt_mm(value)} millimètres."
        )
        facts.append(
            {
                "station": station,
                "year": year,
                "indicator": "annual_rainfall_mm",
                "value": value,
                "unit": "millimetres",
            }
        )

    variation = station_values["variation"]
    direction = "augmenté" if variation >= 0 else "diminué"
    lines.append(
        f"Entre 2022 et 2023, la pluviométrie annuelle à la station de "
        f"{station} a {direction} de {fmt_percent(abs(variation))} %, passant de "
        f"{fmt_mm(station_values[2022])} à {fmt_mm(station_values[2023])} millimètres."
    )
    facts.append(
        {
            "station": station,
            "period": "2022-2023",
            "indicator": "annual_rainfall_variation_percent",
            "value": variation,
            "unit": "percent",
        }
    )

text = "\n".join(lines).strip() + "\n"
OUT_FILE.write_text(text, encoding="utf-8")

document_id = "civ_rainfall_stations_2022_2023_v0.1"
record = {
    "document_id": document_id,
    "source_id": SOURCE_ID,
    "dataset_id": DATASET_ID,
    "title": "Pluviométrie annuelle par station en 2022 et 2023",
    "country_code": "CIV",
    "country_name": "Côte d'Ivoire",
    "domain": "environment_climate",
    "language": "fr",
    "years": list(YEARS),
    "units": {"rainfall": "millimetres", "variation": "percent"},
    "license": "Licence Ouverte / Open Licence 2.0",
    "content_type": "deterministic_structured_factual_text",
    "source_rows": int(len(df)),
    "stations": len(STATIONS),
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
    "stations": len(STATIONS),
    "years": list(YEARS),
    "variation_tolerance_percentage_points": (
        VARIATION_TOLERANCE_PERCENTAGE_POINTS
    ),
    "variation_checks": variation_checks,
    "sentences_generated": len(lines),
    "atomic_facts": len(facts),
    "license": "Licence Ouverte / Open Licence 2.0",
    "output": str(OUT_FILE),
}
write_json(REPORT, report)

print("=" * 72)
print("IVOIRESLM — RAINFALL STATIONS FACTUAL v0.1")
print("=" * 72)
print("Lignes source     :", len(df))
print("Stations          :", len(STATIONS))
print("Années            :", ", ".join(map(str, YEARS)))
print("Phrases générées  :", len(lines))
print("Faits atomiques   :", len(facts))
print("Variations validées:", f"{sum(c['valid'] for c in variation_checks)}/{len(STATIONS)}")
print()
print("=== APERÇU ===")
for line in lines[:6]:
    print("-", line)
print("...")
for line in lines[-3:]:
    print("-", line)
print()
print("Document :", OUT_FILE)
print("Manifest :", MANIFEST)
print("Rapport  :", REPORT)
