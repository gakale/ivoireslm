import json
from pathlib import Path
import subprocess
import sys

from src.evaluation.assistant_direct_answer import (
    is_echo_fragment,
    required_answer_success,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/data/build_assistant_curriculum_v1_4.py"


def build_dataset(tmp_path, feedback=None):
    output = tmp_path / "dataset"
    command = [sys.executable, str(BUILDER), "--output-dir", str(output)]
    if feedback is not None:
        command.extend(["--development-feedback", str(feedback)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    rows = {
        split: [
            json.loads(line)
            for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for split in ("train", "validation")
    }
    return output, report, rows


def test_stage5_builds_disjoint_train_validation_without_test(tmp_path):
    output, report, rows = build_dataset(tmp_path)
    expected_families = {
        "conversation",
        "general_knowledge",
        "ivoire_grounded",
        "math_exact",
        "instruction_following",
        "reading_comprehension",
        "calibrated_uncertainty",
    }
    assert report["status"] == "train_validation_only_test_not_created"
    assert not (output / "test.jsonl").exists()
    assert {row["task_family"] for row in rows["train"]} == expected_families
    assert {row["task_family"] for row in rows["validation"]} == expected_families
    train_prompts = {row["prompt"].casefold() for row in rows["train"]}
    validation_prompts = {row["prompt"].casefold() for row in rows["validation"]}
    assert train_prompts.isdisjoint(validation_prompts)
    assert report["audit"]["train_validation_prompt_overlap"] == 0


def test_human_answers_are_not_imported_and_questions_are_excluded(tmp_path):
    feedback = tmp_path / "feedback.jsonl"
    sentinel = "CETTE REPONSE HUMAINE NE DOIT JAMAIS ETRE ENTRAINEE"
    feedback.write_text(
        json.dumps(
            {
                "question": "C’est quoi Google ?",
                "human_correction": sentinel,
                "rating": "Incorrecte",
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "question": "Combien font 2 + 2 ?",
                "human_correction": "4",
                "rating": "Incorrecte",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output, report, rows = build_dataset(tmp_path, feedback)
    combined = "\n".join(
        (output / f"{split}.jsonl").read_text(encoding="utf-8")
        for split in ("train", "validation")
    )
    prompts = {row["prompt"].casefold() for split_rows in rows.values() for row in split_rows}
    assert sentinel not in combined
    assert "utilisateur : c’est quoi google ?\nassistant :" not in prompts
    assert report["audit"]["development_questions"] == 2
    assert report["human_feedback_policy"].startswith("questions_used_only")


def test_math_targets_are_complete_verified_answers(tmp_path):
    _, _, rows = build_dataset(tmp_path)
    math_rows = [
        row for split_rows in rows.values() for row in split_rows
        if row["task_family"] == "math_exact"
    ]
    assert math_rows
    for row in math_rows:
        expression = row["expression"]
        left, operator, right = expression.split()
        left, right = int(left), int(right)
        if operator == "+":
            expected = left + right
        elif operator == "−":
            expected = left - right
        elif operator == "×":
            expected = left * right
        else:
            expected = left // right
        assert row["required_answers"] == [str(expected)]
        assert row["target"].strip() == f"Le résultat de {expression} est {expected}."
        assert row["target"].strip() != str(expected)


def test_stage5_evaluator_detects_echo_and_accepts_semantic_answer():
    fact = {
        "task_family": "ivoire_grounded",
        "target": " La capitale politique est Yamoussoukro.\n",
        "required_answers": ["Yamoussoukro"],
    }
    math = {
        "task_family": "math_exact",
        "target": " Le résultat de 10 + 10 est 20.\n",
        "required_answers": ["20"],
    }
    assert required_answer_success(fact, "C’est Yamoussoukro.")
    assert required_answer_success(math, "Après calcul, la réponse est 20.")
    assert not required_answer_success(math, "Après calcul, la réponse est 19.")
    assert is_echo_fragment("C’est quoi Google ?", "Google", "Google est une entreprise.")
    assert not is_echo_fragment(
        "C’est quoi Google ?",
        "Google est une entreprise technologique.",
        "Google est une entreprise technologique.",
    )
