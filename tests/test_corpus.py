import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.corpus import jaccard, normalize_text, normalized_line_key, token_shingles


def test_normalize_text_preserves_language_and_removes_technical_noise():
    original = "  Nouchi : c’est comment ?  \r\n\r\n\r\nDeuxième ligne.\x00\n"
    assert normalize_text(original) == "Nouchi : c’est comment ?\n\nDeuxième ligne.\n"


def test_normalized_line_key_detects_technical_duplicates():
    assert normalized_line_key("École   ivoirienne") == normalized_line_key("École ivoirienne")


def test_shingle_similarity():
    left = token_shingles("un deux trois quatre cinq six")
    right = token_shingles("un deux trois quatre cinq sept")
    assert 0 < jaccard(left, right) < 1
