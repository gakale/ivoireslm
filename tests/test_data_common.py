import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.common import format_number_fr


def test_format_integer_keeps_trailing_zeroes():
    assert format_number_fr(29_389_150, max_decimals=0) == "29\u202f389\u202f150"
    assert format_number_fr(208_720, max_decimals=0) == "208\u202f720"


def test_format_decimal_removes_only_fractional_zeroes():
    assert format_number_fr(2_592.0, max_decimals=1) == "2\u202f592"
    assert format_number_fr(2_095.9, max_decimals=1) == "2\u202f095,9"
