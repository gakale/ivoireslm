import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/snapshot_wikipedia_fr_diverse_v02.py"
SPEC = importlib.util.spec_from_file_location("snapshot_wikipedia_diverse_v02", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_extended_snapshot_has_math_and_ivoirian_routes():
    labels = {row[-1] for row in MODULE.CATEGORY_PLAN}
    assert "cote_ivoire" in labels
    assert "mathematiques" in labels
    assert sum(row[2] for row in MODULE.CATEGORY_PLAN) >= 50_000


def test_stable_score_is_repeatable():
    page = {"pageid": 42, "title": "Mathématiques en Côte d'Ivoire"}
    assert MODULE.stable_score(page) == MODULE.stable_score(dict(page))
