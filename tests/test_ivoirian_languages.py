import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.ivoirian_languages import (
    assign_components_to_splits,
    clean_transcription,
    deduplicate_parallel_records,
    extract_text_fields,
    leakage_components,
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


def _parallel(record_id, dyu, fr, en):
    return {"record_id": record_id, "texts": {"dyu": dyu, "fr": fr, "en": en}}


def test_parallel_splits_keep_shared_translations_together():
    rows = [
        _parallel("1", "a", "merci", "thanks"),
        _parallel("2", "b", "merci", "thank you"),
        _parallel("3", "c", "salut", "hello"),
        _parallel("4", "c", "bonjour", "good morning"),
    ]
    components = leakage_components(rows, ("dyu", "fr", "en"))
    assert sorted(map(len, components)) == [2, 2]
    splits = assign_components_to_splits(components, {"train": 0.5, "test": 0.5})
    locations = {
        row["record_id"]: split for split, records in splits.items() for row in records
    }
    assert locations["1"] == locations["2"]
    assert locations["3"] == locations["4"]


def test_parallel_deduplication_requires_every_language():
    rows = [
        _parallel("2", "a", "merci", "thanks"),
        _parallel("1", "A", "Merci", "THANKS"),
        _parallel("3", "b", "salut", ""),
    ]
    result = deduplicate_parallel_records(rows, ("dyu", "fr", "en"))
    assert [row["record_id"] for row in result] == ["1"]
