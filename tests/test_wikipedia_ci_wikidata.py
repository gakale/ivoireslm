import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/snapshot_wikipedia_ci_wikidata_v01.py"
SPEC = importlib.util.spec_from_file_location("snapshot_wikipedia_ci_wikidata_v01", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_wikidata_query_uses_explicit_ivoirian_relations():
    assert "wd:Q1008" in MODULE.WIKIDATA_QUERY_TEMPLATE
    assert MODULE.WIKIDATA_ROUTES["country"][0] == "P17"
    assert MODULE.WIKIDATA_ROUTES["citizenship"][0] == "P27"
    assert "FILTER NOT EXISTS" in MODULE.WIKIDATA_ROUTES["country"][1]


def test_stable_score_is_repeatable():
    row = {"pageid": 123, "title": "Exemple ivoirien"}
    assert MODULE.stable_score(row) == MODULE.stable_score(dict(row))
