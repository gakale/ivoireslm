import importlib.util
import sys
from pathlib import Path

import pytest


pytest.importorskip("tokenizers")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/tokenizer/compare_bpe_tokenizers_v01.py"
SPEC = importlib.util.spec_from_file_location("compare_bpe_tokenizers_v01", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_percent_change_reports_regression_and_reduction():
    assert MODULE.percent_change(100, 110) == pytest.approx(10.0)
    assert -MODULE.percent_change(100, 75) == pytest.approx(25.0)
