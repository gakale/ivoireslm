import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/build_corpus_v08.py"
SPEC = importlib.util.spec_from_file_location("build_corpus_v08", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_template_signature_masks_numbers_and_quoted_indicators():
    first = (
        "En 1985, l’indicateur « PPP (current US$) » atteint "
        "11 319 163 en Côte d’Ivoire."
    )
    second = (
        "En 2022, l’indicateur « Population totale » atteint "
        "29 389 150 en Côte d’Ivoire."
    )
    assert MODULE.template_signature(first) == MODULE.template_signature(second)


def test_template_source_detection_is_narrow():
    assert MODULE.uses_repeated_templates(
        {"generation_method": "python_deterministic_source_label_preserving_templates"}
    )
    assert not MODULE.uses_repeated_templates(
        {"generation_method": "mediawiki_plaintext_extract_snapshot"}
    )


def test_split_document_preserves_text_and_respects_line_boundaries():
    text = "ligne alpha\nligne bêta\nligne gamma\n"
    chunks = MODULE.split_document(text, maximum_characters=24)
    assert len(chunks) == 2
    assert "".join(chunks) == text
    assert all(chunk.endswith("\n") for chunk in chunks)
