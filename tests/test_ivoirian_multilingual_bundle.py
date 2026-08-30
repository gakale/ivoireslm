import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/build_ivoirian_multilingual_bundle_v01.py"
SPEC = importlib.util.spec_from_file_location("build_ivoirian_multilingual_bundle_v01", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_render_dyu_fr_keeps_language_labels_and_text():
    rendered = MODULE.render_dyu_fr({"dyu": "I ni ce", "fr": "Merci", "en": "Thanks"})
    assert rendered == "Dioula : I ni ce\nFrançais : Merci"
    assert "Thanks" not in rendered
