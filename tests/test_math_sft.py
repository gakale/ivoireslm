import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.math_sft import SFT_FAMILIES, generate_sft_exercise, verify_sft_exercise


def test_sft_families_are_deterministic_and_verified():
    for family in SFT_FAMILIES:
        first = generate_sft_exercise(family, 123)
        second = generate_sft_exercise(family, 123)
        assert first == second
        assert verify_sft_exercise(first)
        assert first["prompt"] + first["target"] == first["text"]
        assert all(label in first["text"] for label in ("Problème :", "Méthode :", "Solution :", "Réponse :"))


def test_sft_curriculum_has_four_difficulty_levels():
    observed = {generate_sft_exercise("addition", index)["difficulty"] for index in range(4)}
    assert observed == {1, 2, 3, 4}
