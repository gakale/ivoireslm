import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.faostat import format_faostat_value, render_faostat_production_sentence


def test_format_faostat_value_preserves_source_precision():
    assert format_faostat_value("58061073.692800") == "58 061 073,6928"
    assert format_faostat_value("548.000000") == "548"


def test_format_faostat_value_rejects_nonfinite_or_non_numeric_values():
    for value in ("", "NaN", "Infinity"):
        try:
            format_faostat_value(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{value!r} aurait dû être refusé")


def test_render_faostat_sentence_preserves_official_labels_and_codes():
    row = {
        "year": "2024",
        "item": "Cocoa beans",
        "item_code": "661",
        "item_code_cpcx": "'01640",
        "element": "Production",
        "element_code": "5510",
        "value": "1050.250000",
        "unit": "t",
        "flag": "A",
    }
    sentence = render_faostat_production_sentence(row, template_index=1)
    assert "Cocoa beans" in sentence
    assert "Production" in sentence
    assert "1 050,25 t" in sentence
    assert "01640" in sentence
    assert "Côte d’Ivoire" not in sentence or "ivoirienne" in sentence
