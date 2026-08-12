import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.fish_meat_trade import render_fish_meat_trade_sentence


def test_render_sentence_with_type():
    sentence = render_fish_meat_trade_sentence(
        2014,
        "Importation",
        "Viande",
        "Bovine",
        "une quantité de 12 tonnes",
    )

    assert "le type « Bovine »" in sentence
    assert sentence.endswith("une quantité de 12 tonnes.")


def test_render_sentence_without_optional_type():
    sentence = render_fish_meat_trade_sentence(
        2005,
        "Production",
        "Production pêche artisanale",
        None,
        "une quantité de 25 653 tonnes",
    )

    assert "Production pêche artisanale" in sentence
    assert "type" not in sentence
    assert sentence.endswith("une quantité de 25 653 tonnes.")
