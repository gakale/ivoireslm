import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/build_corpus_v09.py"
SPEC = importlib.util.spec_from_file_location("build_corpus_v09", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_split_document_respects_character_ceiling():
    text = "\n".join("x" * 100 for _ in range(100)) + "\n"
    chunks = MODULE.split_document(text, maximum_characters=1000)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_required_privacy_sources_are_blocked():
    assert "civ_cei" in MODULE.PRIVACY_BLOCKED_SOURCES
    assert "civ_solidarity_poverty" in MODULE.PRIVACY_BLOCKED_SOURCES


def test_v09_has_separate_full_and_public_rights_profiles():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local_research_and_training_only" in source
    assert "exclude_C_PUBLIC_LOCAL_INGEST" in source
