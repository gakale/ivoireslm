import math
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.worldbank import format_worldbank_value, render_wdi_sentence


def test_format_worldbank_value_preserves_large_and_decimal_values():
    assert format_worldbank_value(58_061_073_692_800) == "58 061 073 692 800"
    assert format_worldbank_value(1370.370370) == "1 370,37037"


def test_format_worldbank_value_uses_scientific_notation_for_tiny_values():
    assert format_worldbank_value(3.6454016148439e-15) == "3,64540161484 × 10^-15"


def test_format_worldbank_value_rejects_nonfinite_values():
    try:
        format_worldbank_value(math.nan)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN aurait dû être refusé")


def test_render_wdi_sentence_keeps_official_indicator_name():
    sentence = render_wdi_sentence(
        2024, "Population, total", 31_000_000, template_index=1
    )
    assert "Population, total" in sentence
    assert "31 000 000" in sentence
    assert "Côte d’Ivoire" in sentence
