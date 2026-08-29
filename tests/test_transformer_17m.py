import importlib.util
import sys
from pathlib import Path

import pytest
import torch


pytest.importorskip("tokenizers")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/training/train_transformer_ci_v03_17m.py"
SPEC = importlib.util.spec_from_file_location("train_transformer_17m", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_default_model_has_expected_size_and_tied_embeddings():
    config = MODULE.TrainingConfig()
    model = MODULE.MicroIvoireTransformer17M(config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 17_129_280
    assert model.language_head.weight is model.token_embedding.weight


def test_small_model_forward_and_loss_are_finite():
    config = MODULE.TrainingConfig(
        vocab_size=128,
        block_size=32,
        embedding_dim=64,
        attention_heads=4,
        transformer_layers=2,
        feed_forward_dim=128,
        dropout=0.0,
    )
    model = MODULE.MicroIvoireTransformer17M(config)
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_learning_rate_warmup_and_floor():
    config = MODULE.TrainingConfig()
    assert MODULE.learning_rate(0, config) < MODULE.learning_rate(
        config.warmup_steps - 1, config
    )
    assert MODULE.learning_rate(config.max_steps, config) == pytest.approx(
        config.minimum_learning_rate
    )
