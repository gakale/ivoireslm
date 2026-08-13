import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.french_open import (
    clean_open_french_text,
    dictionary_definitions,
    dictionary_senses,
    render_dictionary_entry,
    valid_documentation_block,
)


def test_clean_open_french_text_removes_rst_roles_without_losing_labels():
    value = "Voir :func:`ouvrir <open>` et ``fichier``."
    assert clean_open_french_text(value) == "Voir ouvrir et fichier."


def test_dictionary_definitions_excludes_forms_and_deduplicates():
    entry = {
        "senses": [
            {"glosses": ["Définition française suffisamment longue."]},
            {"glosses": ["Définition française suffisamment longue."]},
            {"glosses": ["Forme conjuguée de lire."], "form_of": [{"word": "lire"}]},
        ]
    }
    assert dictionary_definitions(entry) == ["Définition française suffisamment longue."]


def test_dictionary_senses_preserves_usage_labels():
    entry = {
        "senses": [
            {
                "glosses": ["Définition française suffisamment longue."],
                "raw_tags": ["Péjoratif", "Vieilli"],
            }
        ]
    }
    assert dictionary_senses(entry)[0]["labels"] == ["Péjoratif", "Vieilli"]


def test_render_dictionary_entry_uses_only_source_fields():
    entry = {
        "word": "accueil",
        "pos_title": "Nom commun",
        "etymology_texts": ["Déverbal de accueillir."],
        "senses": [{"glosses": ["Fait de recevoir une personne qui arrive."]}],
    }
    text = render_dictionary_entry(entry)
    assert "accueil" in text
    assert "Nom commun" in text
    assert "Déverbal de accueillir" in text
    assert "Fait de recevoir" in text


def test_documentation_block_requires_substantial_text():
    assert valid_documentation_block("Cette phrase de documentation contient assez de texte français utile.")
    assert not valid_documentation_block("Titre bref")
