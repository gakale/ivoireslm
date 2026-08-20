import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from models.bigram import BigramCountModel


def test_bigram_learns_deterministic_transition():
    model = BigramCountModel(7, special_tokens=4, alpha=0.1)
    model.add_transitions(np.array([4, 4, 4]), np.array([5, 5, 5]))
    model.fit()
    assert model.log_probabilities[4, 5] > model.log_probabilities[4, 6]
    assert model.training_negative_log_likelihood() < np.log(3)


def test_bigram_never_generates_special_token():
    model = BigramCountModel(7, special_tokens=4, alpha=0.1)
    model.add_transitions(np.array([4, 5, 6]), np.array([5, 6, 4]))
    model.fit()
    generated = model.generate(4, 100, seed=42)
    assert min(generated) >= 4


def test_bigram_rejects_mismatched_transition_arrays():
    model = BigramCountModel(7, special_tokens=4)
    try:
        model.add_transitions(np.array([4, 5]), np.array([5]))
    except ValueError as error:
        assert "même forme" in str(error)
    else:
        raise AssertionError("des transitions incohérentes doivent être rejetées")
