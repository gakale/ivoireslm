import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.math_sft import SFT_FAMILIES, generate_sft_exercise
from evaluation.math_benchmark import (
    BENCHMARK_FAMILIES,
    generate_benchmark_exercise,
    score_completion,
)
from tools.math_solver import UnsupportedMathProblem, solve_math_problem


def _expected_from_sft(record: dict) -> dict:
    verification = record["verification"]
    family = record["family"]
    if family == "fraction_reduction":
        return {
            "kind": "fraction",
            "numerator": verification["reduced_numerator"],
            "denominator": verification["reduced_denominator"],
        }
    if family == "quadratic_factorization":
        return {"kind": "roots", "values": verification["roots"]}
    if family == "rectangle":
        return {
            "kind": "rectangle",
            "area": verification["area"],
            "perimeter": verification["perimeter"],
        }
    if family == "ivorian_cooperative_sharing":
        return {
            "kind": "sharing",
            "quotient": verification["quotient"],
            "remainder": verification["remainder"],
        }
    value_key = "change" if family == "ivorian_market_change" else "root" if family == "linear_equation" else "result"
    return {"kind": "integer", "value": verification[value_key]}


def test_solver_scores_100_percent_on_generated_development_benchmark():
    for family in BENCHMARK_FAMILIES:
        for index in range(50):
            record = generate_benchmark_exercise(family, index)
            solution = solve_math_problem(record["problem"])
            score = score_completion(record, solution.completion)
            assert score["formatted"]
            assert score["correct"], (family, index, record["problem"], solution)


def test_solver_scores_100_percent_on_sft_distribution():
    for family in SFT_FAMILIES:
        for index in range(50):
            source = generate_sft_exercise(family, index)
            record = {"expected": _expected_from_sft(source)}
            solution = solve_math_problem(source["problem"])
            score = score_completion(record, solution.completion)
            assert score["formatted"]
            assert score["correct"], (family, index, source["problem"], solution)


def test_solver_uses_problem_text_only():
    problem = "Calculer 12345 + 67890."
    assert solve_math_problem(problem).answer == "80235"


def test_solver_accepts_natural_school_quadratic_notation():
    problem = "Soit x un nombre réel. Résoudre l’équation x² − 5x + 6 = 0."
    assert solve_math_problem(problem).answer == "x = 2 ou x = 3"


@pytest.mark.parametrize(
    "problem",
    (
        "Explique les mathématiques.",
        "Calculer 1 / 0.",
        "Résoudre 0x + (1) = 1.",
        "Awa paie environ 1000 FCFA.",
    ),
)
def test_solver_fails_closed_on_unsupported_or_ambiguous_problem(problem):
    with pytest.raises(UnsupportedMathProblem):
        solve_math_problem(problem)
