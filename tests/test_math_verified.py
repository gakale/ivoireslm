import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.math_verified import FAMILIES, generate_exercise, split_for_exercise, verify_exercise


def test_every_math_family_is_deterministic_and_verified():
    for family in FAMILIES:
        first = generate_exercise(family, 42)
        second = generate_exercise(family, 42)
        assert first == second
        assert verify_exercise(first)
        assert all(label in first["text"] for label in ("Problème :", "Méthode :", "Solution :", "Réponse :"))


def test_math_splits_are_stable_and_disjoint():
    assignments = {split_for_exercise(f"math_{index}") for index in range(1000)}
    assert assignments == {"train", "validation", "test"}
    assert split_for_exercise("math_123") == split_for_exercise("math_123")

