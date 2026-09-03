from pathlib import Path
import runpy
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def test_tokenizer_wrapper_uses_only_v111_sources(monkeypatch):
    implementation = ModuleType("tokenize_corpus_supplement_v10")
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    runpy.run_path(
        ROOT / "scripts/tokenizer/tokenize_corpus_supplement_v111.py",
        run_name="ivoireslm_tokenizer_v111_test",
    )

    assert implementation.DATASET_ID == "ivoireslm_corpus_supplement_v1.1.1"
    assert set(implementation.BUCKETS) == {
        "wikipedia_fr_natural_v0.3",
        "openassistant_fr_v0.1",
        "data_gouv_ci_open_v0.1.1",
    }


def test_cpt_wrapper_is_small_and_balanced(monkeypatch):
    implementation = ModuleType("continue_pretraining_mix_v10_17m")
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    runpy.run_path(
        ROOT / "scripts/training/continue_pretraining_mix_v111_17m.py",
        run_name="ivoireslm_cpt_v111_test",
    )

    assert implementation.MAX_STEPS == 500
    assert implementation.EVALUATION_INTERVAL == 125
    assert implementation.LEARNING_RATE == 1e-5
    assert sum(implementation.WEIGHTS.values()) == 1.0
    assert implementation.WEIGHTS["base_v09"] == 0.70
    assert implementation.WEIGHTS["natural_ivoirian_grounded_verified"] == 0.05
    assert implementation.MAXIMUM_BASE_VALIDATION_INCREASE == 0.015
