import hashlib
import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import sha256_text, upsert_jsonl, write_json
from data.worldbank import render_wdi_sentence


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/ivoiredata_2026-08-13_growth_v0.2/data"
SOURCE_ROOT = SNAPSHOT / "multidomain/civ_worldbank_wdi"
OUT_FILE = ROOT / "derived/structured_factual_v0.2/multidomain/civ_worldbank_wdi_1960_2025_v0.1.txt"
MANIFEST = ROOT / "manifests/structured_factual_v0.1.jsonl"
REPORT = ROOT / "reports/worldbank_wdi_factual_v0.1_report.json"


def exactly_one(directory):
    files = list(directory.rglob("*.parquet"))
    if len(files) != 1:
        raise ValueError(f"Une table attendue dans {directory}, trouvé {len(files)}")
    return files[0]


manifest_source = SOURCE_ROOT / "manifest.json"
if not manifest_source.is_file():
    raise FileNotFoundError(manifest_source)
observations_path = exactly_one(SOURCE_ROOT / "tables/data/worldbank_wdi")
indicators_path = exactly_one(SOURCE_ROOT / "tables/data/worldbank_wdi_indicators")
observations = pd.read_parquet(observations_path)
indicators = pd.read_parquet(indicators_path)

required = ["date", "indicator__id", "indicator__value", "value", "value__v_double"]
missing = [column for column in required if column not in observations]
if missing:
    raise ValueError(f"Colonnes WDI manquantes : {missing}")
numeric = observations["value__v_double"].where(
    observations["value__v_double"].notna(),
    pd.to_numeric(observations["value"], errors="coerce"),
)
usable = observations[numeric.notna()].copy()
usable["numeric_value"] = numeric[numeric.notna()]
usable["year"] = pd.to_numeric(usable["date"], errors="coerce")
usable = usable[
    usable["year"].between(1960, 2025)
    & usable["indicator__id"].notna()
    & usable["indicator__value"].notna()
].copy()
usable["year"] = usable["year"].astype(int)
if len(usable) != 40_477:
    raise ValueError(f"40 477 observations WDI attendues, trouvé {len(usable)}")
if usable.duplicated(["indicator__id", "year"]).any():
    raise ValueError("Clés indicateur/année WDI dupliquées")
if usable["indicator__id"].nunique() != 1_423:
    raise ValueError("Nombre inattendu d’indicateurs WDI observés")
if len(indicators) != 1_498:
    raise ValueError("Nombre inattendu de métadonnées d’indicateurs WDI")

usable = usable.sort_values(["indicator__id", "year"], kind="stable").reset_index(drop=True)
lines = [
    render_wdi_sentence(
        row.year,
        " ".join(str(row.indicator__value).split()),
        row.numeric_value,
        index,
    )
    for index, row in enumerate(usable.itertuples(index=False))
]
text = "\n".join(lines) + "\n"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(text, encoding="utf-8")

record = {
    "document_id": "civ_worldbank_wdi_1960_2025_v0.1",
    "source_id": "civ_worldbank_wdi",
    "group_id": "worldbank_wdi_civ",
    "title": "Indicateurs du développement dans le monde pour la Côte d’Ivoire",
    "country_code": "CIV",
    "country_name": "Côte d’Ivoire",
    "domain": "multidomain",
    "language": "fr-en",
    "content_type": "deterministic_structured_factual_text",
    "rights_tier": "A_REDISTRIBUTABLE",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "attribution": "World Development Indicators, World Bank",
    "license_url": "https://datacatalog.worldbank.org/public-licenses",
    "dataset_url": "https://datacatalog.worldbank.org/infrastructure-data/search/dataset/0037712/world-development-indicators",
    "observed_year_range": [1960, 2025],
    "source_rows": int(len(observations)),
    "usable_rows": int(len(usable)),
    "excluded_missing_or_invalid_rows": int(len(observations) - len(usable)),
    "indicators_observed": int(usable["indicator__id"].nunique()),
    "sentences_generated": len(lines),
    "atomic_facts": len(lines),
    "generation_method": "python_deterministic_source_label_preserving_templates",
    "source_table": str(observations_path.relative_to(SNAPSHOT)),
    "source_table_sha256": hashlib.sha256(observations_path.read_bytes()).hexdigest(),
    "indicator_table": str(indicators_path.relative_to(SNAPSHOT)),
    "indicator_table_sha256": hashlib.sha256(indicators_path.read_bytes()).hexdigest(),
    "output_path": str(OUT_FILE),
    "text_sha256": sha256_text(text),
    "training_branch": "derived_structured_factual",
}
upsert_jsonl(MANIFEST, record, key="document_id")
write_json(
    REPORT,
    {
        **record,
        "source_indicator_metadata_rows": int(len(indicators)),
        "source_values_in_primary_column": int(observations["value"].notna().sum()),
        "source_values_in_double_column": int(observations["value__v_double"].notna().sum()),
        "empty_rows_excluded": int(len(observations) - len(usable)),
    },
)
print("WDI FACTUAL v0.1 : OK")
print("Observations source :", len(observations))
print("Observations utiles :", len(usable))
print("Indicateurs         :", usable["indicator__id"].nunique())
print("Période             : 1960–2025")
print("Document            :", OUT_FILE)
print("SHA-256             :", sha256_text(text))
