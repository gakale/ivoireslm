from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/training/finetune_assistant_v1_6_stage6b_17m.py"
RUNNER = ROOT / "scripts/colab/run_assistant_v1_6_stage6b.py"


def test_stage6b_is_short_targeted_and_keeps_raw_replay():
    source = TRAINER.read_text(encoding="utf-8")
    assert "max_steps: int = 250" in source
    assert "raw_language_probability: float = 0.25" in source
    assert "learning_rate: float = 4e-6" in source
    assert sum(source.count(f'"{family}"') for family in (
        "conversation", "identity", "general_knowledge", "ivoire_grounded",
        "dioula_basic", "math_exact", "calibrated_uncertainty",
    )) >= 7


def test_stage6b_runner_uses_frozen_stage6_parent_and_keeps_test_sealed():
    source = RUNNER.read_text(encoding="utf-8")
    assert "651eecf263399d872d3d1b20d2ef16d12f1dec0787e7631762d2d322084c8856" in source
    assert '"--stop-step", "125"' in source
    assert '"sealed_test_opened": False' in source
    assert "human_holdout.jsonl" in source
