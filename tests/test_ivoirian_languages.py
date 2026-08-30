import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.ivoirian_languages import (
    clean_transcription,
    extract_text_fields,
    quality_statistics,
    stable_record_id,
)


def test_clean_transcription_preserves_ivoirian_characters():
    assert clean_transcription("  Ɲanmiɛn\n  su lɔʼn  ") == "Ɲanmiɛn su lɔʼn"


def test_extract_text_fields_excludes_metadata_and_empty_values():
    row = {"dyu": "I ni ce", "fr": "Merci", "en": " ", "gender": "female"}
    assert extract_text_fields(row, {"dyu": "dyu", "fr": "fr", "en": "en"}) == {
        "dyu": "I ni ce",
        "fr": "Merci",
    }


def test_stable_record_id_is_deterministic_and_split_specific():
    first = stable_record_id("dataset", "train", 12)
    assert first == stable_record_id("dataset", "train", 12)
    assert first != stable_record_id("dataset", "test", 12)


def test_quality_statistics_detects_duplicates_and_split_leaks():
    result = quality_statistics({"train": ["a", "a", "b"], "test": ["b", "c"]})
    assert result["internal_duplicates"] == {"train": 1, "test": 0}
    assert result["cross_split_exact_overlaps"] == {"test__train": 1}
    assert result["distinct_characters"] == 3
