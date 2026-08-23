import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.wikipedia_ci import clean_wikipedia_extract, is_useful_page, split_for_page


def test_wikipedia_extract_cleanup_stops_before_reference_sections():
    text = "Titre\r\n\r\nUn paragraphe   utile.\r\nNotes et références\r\nRéférence"
    assert clean_wikipedia_extract(text) == "Titre\n\nUn paragraphe utile.\n"


def test_wikipedia_page_filter_and_split_are_stable():
    assert is_useful_page("Abidjan", "a" * 500)
    assert not is_useful_page("Liste des communes", "a" * 1000)
    assert not is_useful_page("Abidjan", "court")
    assert split_for_page(12345) == split_for_page(12345)

