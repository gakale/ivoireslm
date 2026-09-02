import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/data/build_assistant_curriculum_v1_4_1.py"


def test_stage5b_preserves_operators_and_never_imports_human_answers(tmp_path):
    feedback = tmp_path / "feedback.jsonl"
    sentinel = "REPONSE HUMAINE INTERDITE DANS LE CURRICULUM"
    feedback.write_text(
        "".join(
            json.dumps(
                {
                    "question": f"Question humaine {index}",
                    "human_correction": sentinel,
                },
                ensure_ascii=False,
            ) + "\n"
            for index in range(66)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--output-dir",
            str(output),
            "--development-feedback",
            str(feedback),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    train_text = (output / "train.jsonl").read_text(encoding="utf-8")
    validation_text = (output / "validation.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in train_text.splitlines()]
    math_rows = [row for row in rows if row["task_family"] == "math_exact"]
    operators = {operator for row in math_rows for operator in ("+", "−", "×") if operator in row["expression"]}
    assert operators == {"+", "−", "×"}
    assert len(math_rows) == 2523
    assert report["audit"]["development_questions"] == 66
    assert report["audit"]["train_validation_prompt_overlap"] == 0
    assert sentinel not in train_text + validation_text
    assert not (output / "test.jsonl").exists()

