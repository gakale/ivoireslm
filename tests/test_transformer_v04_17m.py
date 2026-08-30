import importlib.util
import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")
pytest.importorskip("tokenizers")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/training/train_transformer_ci_v04_17m.py"
SPEC = importlib.util.spec_from_file_location("train_transformer_v04_17m", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_v04_uses_new_corpus_and_tokenizer_with_same_architecture():
    config = MODULE.v04_config()
    assert config.model_id == "microivoire_transformer_v0.4_17m"
    assert config.tokenizer_id == "ivoireslm_bpe_v0.4"
    assert config.corpus_id == "ivoireslm_corpus_v0.9.0"
    model = MODULE.MicroIvoireTransformer17M(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 17_129_280


def test_v04_seed_differs_from_v03_experiment():
    assert MODULE.v04_config().seed == 20260830
