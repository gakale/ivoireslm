import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/data/build_assistant_curriculum_v1_5.py"


def record(index: int, correction: str, consent: bool = True) -> dict:
    return {
        "question": f"Question humaine numéro {index}",
        "human_correction": correction,
        "category": "conversation_ordinaire" if index % 2 else "autre",
        "feedback_id": f"feedback-{index}",
        "consent_for_research_and_training": consent,
        "submitter_attested_no_personal_data": True,
    }


def test_v15_filters_feedback_and_seals_human_holdout(tmp_path):
    feedback = tmp_path / "feedback.jsonl"
    rows = [record(index, f"Voici une réponse humaine utile numéro {index}.") for index in range(80)]
    rows += [record(100, "Bonne réponse"), record(101, "Réponse sans consentement.", False)]
    feedback.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "dataset"
    subprocess.run(
        [sys.executable, str(BUILDER), "--output-dir", str(output), "--human-feedback", str(feedback)],
        check=True, capture_output=True, text=True,
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    train = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    validation = [json.loads(line) for line in (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
    holdout = [json.loads(line) for line in (output / "human_holdout.jsonl").read_text(encoding="utf-8").splitlines()]

    assert report["test_created"] is False
    assert report["audit"]["feedback"]["accepted_unique"] == 80
    assert report["audit"]["feedback"]["rejected"] == {
        "consent_missing": 1,
        "generic_correction": 1,
    }
    assert holdout
    assert {row["task_family"] for row in train} == {
        "conversation", "identity", "general_knowledge", "ivoire_grounded",
        "dioula_basic", "math_exact", "instruction_following",
        "reading_comprehension", "calibrated_uncertainty",
    }
    train_prompts = {row["prompt"] for row in train}
    validation_prompts = {row["prompt"] for row in validation}
    assert not train_prompts & validation_prompts
    holdout_questions = {row["question"] for row in holdout}
    extracted_train_questions = {
        prompt.removeprefix("Utilisateur :").removesuffix("\nAssistant :")
        for prompt in train_prompts
    }
    assert not holdout_questions & extracted_train_questions
    assert not (output / "test.jsonl").exists()


def test_v15_keeps_verified_dioula_seed_out_of_human_holdout(tmp_path):
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text(
        json.dumps(record(1, "Une correction suffisamment détaillée."), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "dataset"
    subprocess.run(
        [sys.executable, str(BUILDER), "--output-dir", str(output), "--human-feedback", str(feedback)],
        check=True, capture_output=True, text=True,
    )
    all_text = (output / "train.jsonl").read_text(encoding="utf-8") + (output / "validation.jsonl").read_text(encoding="utf-8")
    assert "dioula_basic" in all_text
    assert "i ni ce" in all_text
    assert "sɔgɔma" in all_text
