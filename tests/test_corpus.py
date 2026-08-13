import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.corpus import PHONE_RE, jaccard, normalize_text, normalized_line_key, token_shingles


def test_normalize_text_preserves_language_and_removes_technical_noise():
    original = "  Nouchi : c’est comment ?  \r\n\r\n\r\nDeuxième ligne.\x00\n"
    assert normalize_text(original) == "Nouchi : c’est comment ?\n\nDeuxième ligne.\n"


def test_normalized_line_key_detects_technical_duplicates():
    assert normalized_line_key("École   ivoirienne") == normalized_line_key("École ivoirienne")


def test_shingle_similarity():
    left = token_shingles("un deux trois quatre cinq six")
    right = token_shingles("un deux trois quatre cinq sept")
    assert 0 < jaccard(left, right) < 1


def test_phone_detection_does_not_confuse_decimal_values():
    assert PHONE_RE.search("la valeur est 0,0189285714.") is None
    assert PHONE_RE.search("la valeur est 10.0582044025.") is None
    assert PHONE_RE.search("contact : 05 82 04 40 25") is not None
    assert PHONE_RE.search("contact : +225 07 59 96 85 53") is not None
