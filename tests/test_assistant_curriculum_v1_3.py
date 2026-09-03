import json
from pathlib import Path
import subprocess
import sys
import re


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalized_question(prompt):
    prompt = prompt.removeprefix("Utilisateur :").removesuffix("Assistant :")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœ']+", " ", prompt.casefold().replace("’", "'")).strip()


def test_stage4_is_diverse_disjoint_and_test_free(tmp_path):
    output = tmp_path / "stage4"
    subprocess.run([sys.executable, str(ROOT / "scripts/data/build_assistant_curriculum_v1_3.py"), "--output-dir", str(output)], check=True)
    train, validation = load(output / "train.jsonl"), load(output / "validation.jsonl")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(train) == 1221
    assert len(validation) == 260
    assert not (output / "test.jsonl").exists()
    assert report["audit"]["train_validation_prompt_overlap"] == 0
    assert report["audit"]["most_common_train_target_rate"] < 0.02
    assert {row["task_family"] for row in train} == {
        "conversation", "calibrated_uncertainty", "ivoire_grounded",
        "math_exact", "instruction_following", "reading_comprehension",
    }


def test_human_feedback_is_a_strict_holdout(tmp_path):
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text(json.dumps({"question": "Quelle est la capitale politique et administrative de la Côte d’Ivoire ?"}, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "stage4"
    subprocess.run([sys.executable, str(ROOT / "scripts/data/build_assistant_curriculum_v1_3.py"), "--output-dir", str(output), "--held-out-feedback", str(feedback)], check=True)
    rows = load(output / "train.jsonl") + load(output / "validation.jsonl")
    held_out = normalized_question("Quelle est la capitale politique et administrative de la Côte d’Ivoire ?")
    assert all(normalized_question(row["prompt"]) != held_out for row in rows)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["audit"]["human_holdout_questions"] == 1
    assert report["audit"]["rows_excluded_by_human_holdout"]["train"] == 1


def test_stage4_training_wrapper_imports():
    for name in (
        "scripts/training/finetune_assistant_v1_3_stage4_17m.py",
        "scripts/colab/run_assistant_v1_3_stage4.py",
    ):
        script = ROOT / name
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
