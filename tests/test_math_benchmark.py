import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from evaluation.math_benchmark import BENCHMARK_FAMILIES, generate_benchmark_exercise, score_completion, verify_benchmark_exercise


def test_benchmark_is_deterministic_unique_and_verified():
    problems = set()
    for family in BENCHMARK_FAMILIES:
        first = generate_benchmark_exercise(family, 7)
        second = generate_benchmark_exercise(family, 7)
        assert first == second
        assert verify_benchmark_exercise(first)
        assert first["problem"] not in problems
        problems.add(first["problem"])


def test_reference_answers_score_as_correct():
    for family in BENCHMARK_FAMILIES:
        record = generate_benchmark_exercise(family, 11)
        completion = f" méthode de référence\nSolution : calcul vérifié.\nRéponse : {record['reference_answer']}\n"
        result = score_completion(record, completion)
        assert result["formatted"]
        assert result["correct"], (family, result)


def test_missing_answer_label_is_not_accepted():
    record = generate_benchmark_exercise("addition_large", 3)
    result = score_completion(record, f"Le résultat est {record['reference_answer']}")
    assert not result["formatted"]
    assert not result["correct"]
