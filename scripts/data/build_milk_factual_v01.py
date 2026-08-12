import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path.home() / "ivoireslm-structured")
)

from lib.common import (
    format_number_fr,
    sha256_text,
    upsert_jsonl,
    write_json,
)


ROOT = Path.home() / "ivoireslm-storage"

SNAPSHOT = (
    ROOT
    / "snapshots"
    / "ivoiredata_2026-08-10_v0.1"
    / "data"
)

SOURCE_ID = "civ_datagouv_milk_2024"
DATASET_ID = "rb07nb9nxbrgpr7b1lnc8mg6"

SOURCE_ROOT = (
    SNAPSHOT
    / "agriculture"
    / SOURCE_ID
)

OUT_FILE = (
    ROOT
    / "derived"
    / "structured_factual_v0.1"
    / "agriculture"
    / "civ_milk_production_2024_v0.1.txt"
)

MANIFEST = (
    ROOT
    / "manifests"
    / "structured_factual_v0.1.jsonl"
)

REPORT = (
    ROOT
    / "reports"
    / "milk_factual_v0.1_report.json"
)

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 1. TABLE MÉTIER
# --------------------------------------------------

files = [
    p for p in SOURCE_ROOT.rglob("*.parquet")
    if "datagouv_catalog" not in str(p)
]

if len(files) != 1:
    raise SystemExit(
        f"1 table métier attendue, trouvé {len(files)}"
    )

source_path = files[0]

df = pd.read_parquet(
    source_path
).copy()


required = [
    "region",
    "lait_bovin_litres",
    "lait_caprin_litres",
    "lait_total_litres",
    "lait_moy_bovin_l_tete",
    "lait_moy_caprin_l_tete",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Colonnes manquantes : {missing}"
    )


# --------------------------------------------------
# 2. PARSING
# --------------------------------------------------

def parse_number(value):

    if pd.isna(value):
        return float("nan")

    text = str(value).strip()

    if text in {
        "",
        "-",
        "—",
        "–",
        "nan",
        "None",
    }:
        return float("nan")

    return float(
        text.replace(",", ".")
    )


numeric_cols = required[1:]

for col in numeric_cols:
    df[col] = df[col].map(
        parse_number
    )


# --------------------------------------------------
# 3. NATIONAL / RÉGIONAL
# --------------------------------------------------

national = df[
    df["region"].astype(str).str.strip()
    == "Ensemble"
].copy()

regional = df[
    df["region"].astype(str).str.strip()
    != "Ensemble"
].copy()

if len(national) != 1:
    raise ValueError(
        f"Une ligne Ensemble attendue, trouvé {len(national)}"
    )

if len(regional) != 28:
    raise ValueError(
        f"28 régions attendues, trouvé {len(regional)}"
    )

nat = national.iloc[0]


# --------------------------------------------------
# 4. VALIDATION PAR LIGNE
# --------------------------------------------------

TOLERANCE_LITRES = 0.02

row_checks = []
missing_values = []
issues = []


for _, row in regional.iterrows():

    region = str(
        row["region"]
    ).strip()

    bovin = row["lait_bovin_litres"]
    caprin = row["lait_caprin_litres"]
    total = row["lait_total_litres"]

    if pd.isna(bovin) or pd.isna(total):

        issues.append({
            "region": region,
            "reason":
                "missing_required_production_value"
        })

        continue

    if not pd.isna(caprin):

        calculated = bovin + caprin
        difference = total - calculated

        ok = (
            abs(difference)
            <= TOLERANCE_LITRES
        )

        row_checks.append({
            "region": region,
            "bovin": bovin,
            "caprin": caprin,
            "total": total,
            "calculated_total":
                calculated,
            "difference":
                difference,
            "valid":
                ok,
        })

        if not ok:

            issues.append({
                "region": region,
                "reason":
                    "bovin_plus_caprin_not_equal_total",
                "difference":
                    difference,
            })

    else:

        missing_values.append({
            "region": region,
            "field":
                "lait_caprin_litres",
            "reason":
                "missing_in_source",
        })

        difference = total - bovin

        row_checks.append({
            "region": region,
            "bovin": bovin,
            "caprin": None,
            "total": total,
            "difference_total_vs_bovin":
                difference,
            "valid":
                abs(difference)
                <= TOLERANCE_LITRES,
            "caprin_inferred":
                False,
        })


# --------------------------------------------------
# 5. VALEURS MOYENNES ABSENTES
# --------------------------------------------------

for _, row in regional.iterrows():

    region = str(
        row["region"]
    ).strip()

    if pd.isna(
        row["lait_moy_bovin_l_tete"]
    ):

        missing_values.append({
            "region": region,
            "field":
                "lait_moy_bovin_l_tete",
            "reason":
                "missing_in_source",
        })

    if pd.isna(
        row["lait_moy_caprin_l_tete"]
    ):

        missing_values.append({
            "region": region,
            "field":
                "lait_moy_caprin_l_tete",
            "reason":
                "missing_in_source",
        })


# --------------------------------------------------
# 6. CONTRÔLES NATIONAUX
# --------------------------------------------------

sum_bovin = float(
    regional["lait_bovin_litres"].sum()
)

sum_caprin = float(
    regional["lait_caprin_litres"].sum(
        skipna=True
    )
)

sum_total = float(
    regional["lait_total_litres"].sum()
)

national_checks = {
    "bovin": {
        "regional_sum":
            sum_bovin,
        "national_value":
            float(
                nat["lait_bovin_litres"]
            ),
        "difference":
            float(
                nat["lait_bovin_litres"]
                - sum_bovin
            ),
    },

    "caprin": {
        "regional_known_sum":
            sum_caprin,
        "national_value":
            float(
                nat["lait_caprin_litres"]
            ),
        "difference":
            float(
                nat["lait_caprin_litres"]
                - sum_caprin
            ),
    },

    "total": {
        "regional_sum":
            sum_total,
        "national_value":
            float(
                nat["lait_total_litres"]
            ),
        "difference":
            float(
                nat["lait_total_litres"]
                - sum_total
            ),
    },
}

for name, check in national_checks.items():

    if abs(
        check["difference"]
    ) > TOLERANCE_LITRES:

        issues.append({
            "scope": "national",
            "indicator": name,
            "reason":
                "regional_sum_mismatch",
            "difference":
                check["difference"],
        })


if issues:
    raise ValueError(
        f"Issues bloquantes détectées : {issues}"
    )


# --------------------------------------------------
# 7. FORMATAGE
# --------------------------------------------------

def fmt_litres(value):

    return format_number_fr(
        value,
        max_decimals=2
    )


def fmt_average(value):

    return format_number_fr(
        value,
        max_decimals=1
    )


# --------------------------------------------------
# 8. GÉNÉRATION
# --------------------------------------------------

lines = []
facts = []


sentence = (
    f"En 2024, la production totale de lait "
    f"en Côte d'Ivoire s'élève à "
    f"{fmt_litres(nat['lait_total_litres'])} litres, "
    f"dont {fmt_litres(nat['lait_bovin_litres'])} litres "
    f"de lait bovin et "
    f"{fmt_litres(nat['lait_caprin_litres'])} litres "
    f"de lait caprin."
)

lines.append(sentence)

facts.extend([
    {
        "scope": "national",
        "indicator":
            "lait_total_litres",
        "value":
            float(
                nat["lait_total_litres"]
            ),
    },
    {
        "scope": "national",
        "indicator":
            "lait_bovin_litres",
        "value":
            float(
                nat["lait_bovin_litres"]
            ),
    },
    {
        "scope": "national",
        "indicator":
            "lait_caprin_litres",
        "value":
            float(
                nat["lait_caprin_litres"]
            ),
    },
])


for _, row in regional.iterrows():

    region = str(
        row["region"]
    ).strip()

    bovin = row[
        "lait_bovin_litres"
    ]

    caprin = row[
        "lait_caprin_litres"
    ]

    total = row[
        "lait_total_litres"
    ]

    if pd.isna(caprin):

        production_sentence = (
            f"En 2024, la région « {region} » "
            f"enregistre une production totale "
            f"de {fmt_litres(total)} litres de lait, "
            f"avec {fmt_litres(bovin)} litres "
            f"de lait bovin."
        )

    else:

        production_sentence = (
            f"En 2024, la région « {region} » "
            f"enregistre une production totale "
            f"de {fmt_litres(total)} litres de lait, "
            f"dont {fmt_litres(bovin)} litres "
            f"de lait bovin et "
            f"{fmt_litres(caprin)} litres "
            f"de lait caprin."
        )

    lines.append(
        production_sentence
    )

    facts.extend([
        {
            "region":
                region,
            "indicator":
                "lait_total_litres",
            "value":
                float(total),
        },
        {
            "region":
                region,
            "indicator":
                "lait_bovin_litres",
            "value":
                float(bovin),
        },
    ])

    if not pd.isna(caprin):

        facts.append({
            "region":
                region,
            "indicator":
                "lait_caprin_litres",
            "value":
                float(caprin),
        })

    avg_bovin = row[
        "lait_moy_bovin_l_tete"
    ]

    avg_caprin = row[
        "lait_moy_caprin_l_tete"
    ]

    if (
        not pd.isna(avg_bovin)
        and not pd.isna(avg_caprin)
    ):

        average_sentence = (
            f"En 2024, dans la région « {region} », "
            f"la production moyenne de lait est de "
            f"{fmt_average(avg_bovin)} litres par tête "
            f"pour les bovins et de "
            f"{fmt_average(avg_caprin)} litres par tête "
            f"pour les caprins."
        )

    elif not pd.isna(avg_bovin):

        average_sentence = (
            f"En 2024, dans la région « {region} », "
            f"la production moyenne de lait bovin "
            f"est de {fmt_average(avg_bovin)} litres "
            f"par tête."
        )

    else:

        continue

    lines.append(
        average_sentence
    )

    if not pd.isna(avg_bovin):

        facts.append({
            "region":
                region,
            "indicator":
                "lait_moy_bovin_l_tete",
            "value":
                float(avg_bovin),
            "unit":
                "litres_per_head",
        })

    if not pd.isna(avg_caprin):

        facts.append({
            "region":
                region,
            "indicator":
                "lait_moy_caprin_l_tete",
            "value":
                float(avg_caprin),
            "unit":
                "litres_per_head",
        })


# --------------------------------------------------
# 9. DOCUMENT
# --------------------------------------------------

text = (
    "\n".join(lines).strip()
    + "\n"
)

OUT_FILE.write_text(
    text,
    encoding="utf-8"
)


# --------------------------------------------------
# 10. MANIFEST
# --------------------------------------------------

document_id = (
    "civ_milk_production_2024_v0.1"
)

record = {
    "document_id":
        document_id,

    "source_id":
        SOURCE_ID,

    "dataset_id":
        DATASET_ID,

    "title":
        "Production totale et moyenne de lait par région selon l'espèce en 2024",

    "country_code":
        "CIV",

    "country_name":
        "Côte d'Ivoire",

    "domain":
        "agriculture",

    "language":
        "fr",

    "year":
        2024,

    "units": {
        "production":
            "litres",
        "average":
            "litres_per_head",
    },

    "license":
        "Licence Ouverte / Open Licence 2.0",

    "content_type":
        "deterministic_structured_factual_text",

    "source_rows":
        int(len(df)),

    "regional_rows":
        int(len(regional)),

    "national_rows":
        int(len(national)),

    "sentences_generated":
        len(lines),

    "atomic_facts":
        len(facts),

    "facts_review":
        0,

    "missing_source_values":
        missing_values,

    "generation_method":
        "python_deterministic_validated_templates",

    "source_table":
        str(
            source_path.relative_to(
                SNAPSHOT
            )
        ),

    "output_path":
        str(OUT_FILE),

    "text_sha256":
        sha256_text(text),

    "group_id":
        SOURCE_ID,

    "training_branch":
        "derived_structured_factual",
}

upsert_jsonl(
    MANIFEST,
    record,
    key="document_id"
)


# --------------------------------------------------
# 11. RAPPORT
# --------------------------------------------------

report = {
    "source_id":
        SOURCE_ID,

    "dataset_id":
        DATASET_ID,

    "rows_source":
        int(len(df)),

    "regional_rows":
        int(len(regional)),

    "national_rows":
        int(len(national)),

    "national_checks":
        national_checks,

    "row_checks":
        row_checks,

    "missing_source_values":
        missing_values,

    "caprine_production_values_available":
        int(
            regional[
                "lait_caprin_litres"
            ].notna().sum()
        ),

    "bovine_average_values_available":
        int(
            regional[
                "lait_moy_bovin_l_tete"
            ].notna().sum()
        ),

    "caprine_average_values_available":
        int(
            regional[
                "lait_moy_caprin_l_tete"
            ].notna().sum()
        ),

    "sentences_generated":
        len(lines),

    "atomic_facts":
        len(facts),

    "license":
        "Licence Ouverte / Open Licence 2.0",

    "output":
        str(OUT_FILE),
}

write_json(
    REPORT,
    report
)


# --------------------------------------------------
# 12. AFFICHAGE
# --------------------------------------------------

print("=" * 72)
print("IVOIRESLM — MILK PRODUCTION FACTUAL v0.1")
print("=" * 72)

print(
    "Lignes source                :",
    len(df)
)

print(
    "Régions                      :",
    len(regional)
)

print(
    "Ligne nationale              :",
    len(national)
)

print(
    "Production bovine régionale  :",
    f"{regional['lait_bovin_litres'].notna().sum()}/28"
)

print(
    "Production caprine renseignée:",
    f"{regional['lait_caprin_litres'].notna().sum()}/28"
)

print(
    "Moyenne bovine renseignée    :",
    f"{regional['lait_moy_bovin_l_tete'].notna().sum()}/28"
)

print(
    "Moyenne caprine renseignée   :",
    f"{regional['lait_moy_caprin_l_tete'].notna().sum()}/28"
)

print(
    "Phrases générées             :",
    len(lines)
)

print(
    "Faits atomiques              :",
    len(facts)
)

print()

print("=== CONTRÔLES NATIONAUX ===")

for name, check in national_checks.items():

    print(
        f"{name:8} | "
        f"écart={check['difference']:.4f} L"
    )

print()

print("=== APERÇU ===")

for line in lines[:7]:
    print("-", line)

print("...")

for line in lines[-4:]:
    print("-", line)

print()

print("Document :", OUT_FILE)
print("Manifest :", MANIFEST)
print("Rapport  :", REPORT)
