import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/snapshot_ivoirian_institutional_v01.py"
SPEC = importlib.util.spec_from_file_location("snapshot_ivoirian_institutional_v01", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sensitive_and_binary_documents_are_rejected():
    assert MODULE.rejection_reason(
        {"source_id": "civ_cei", "source_url": "https://example.ci/a", "text": "texte"}
    ) == "blocked_sensitive_source"
    assert MODULE.rejection_reason(
        {
            "source_id": "civ_test",
            "source_url": "https://example.ci/liste.xlsx",
            "title": "Données",
            "text": "PK!fichier binaire",
        }
    ) == "unsupported_binary_document"


def test_split_is_stable_and_exclusive():
    first = MODULE.split_for_url("civ_test", "https://example.ci/document")
    second = MODULE.split_for_url("civ_test", "https://example.ci/document")
    assert first == second
    assert first in {"train", "validation", "test"}


def test_clean_lines_removes_global_duplicates_and_boilerplate():
    seen = set()
    first, removed_first = MODULE.clean_lines("Accueil\nUne phrase institutionnelle utile.\n", seen)
    second, removed_second = MODULE.clean_lines(
        "Une phrase institutionnelle utile.\nUne autre phrase utile.\n", seen
    )
    assert "Accueil" not in first
    assert "Une phrase institutionnelle utile." in first
    assert "Une phrase institutionnelle utile." not in second
    assert "Une autre phrase utile." in second
    assert removed_first == 1
    assert removed_second == 1
