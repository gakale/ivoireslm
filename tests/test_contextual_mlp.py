import sys
from pathlib import Path

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from models.contextual_mlp import ContextualCharacterMLP


def make_model(dropout: float = 0.0):
    return ContextualCharacterMLP(
        12,
        context_size=4,
        embedding_dim=3,
        hidden_dim=8,
        dropout=dropout,
    )


def test_forward_shape_and_special_tokens_are_masked():
    model = make_model()
    logits = model(torch.tensor([[4, 5, 6, 7], [5, 6, 7, 8]]))
    assert logits.shape == (2, 12)
    assert torch.all(logits[:, :4] == torch.finfo(logits.dtype).min)


def test_forward_rejects_wrong_context_size():
    with pytest.raises(ValueError, match="forme"):
        make_model()(torch.tensor([[4, 5, 6]]))


def test_generate_is_reproducible_and_avoids_special_tokens():
    model = make_model()
    first = model.generate(
        [4, 5, 6, 7],
        20,
        temperature=1.0,
        generator=torch.Generator().manual_seed(42),
    )
    second = model.generate(
        [4, 5, 6, 7],
        20,
        temperature=1.0,
        generator=torch.Generator().manual_seed(42),
    )
    assert first == second
    assert min(first) >= 4

