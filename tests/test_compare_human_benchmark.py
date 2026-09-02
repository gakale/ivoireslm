import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluation/compare_human_benchmark_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("human_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_detects_repeated_canned_outputs():
    rows = [
        {"prediction": "Je suis IvoireSLM.", "human_expected_answer": "Réponse A", "stopped_on_eos": True, "generated_tokens": 5},
        {"prediction": "Je suis IvoireSLM.", "human_expected_answer": "Réponse B", "stopped_on_eos": True, "generated_tokens": 5},
        {"prediction": "Une réponse différente.", "human_expected_answer": "Une réponse différente.", "stopped_on_eos": False, "generated_tokens": 8},
    ]
    result = MODULE.summarize(rows)
    assert result["unique_outputs"] == 2
    assert result["most_common_output_count"] == 2
    assert result["canned_marker_count"] == 2
    assert result["exact_matches_to_human_answer"] == 1
