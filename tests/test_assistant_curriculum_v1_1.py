import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_stage2_builder_is_disjoint_and_has_no_test(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_assistant_curriculum_v1_1.py"),
            "--output-dir",
            str(tmp_path / "dataset"),
        ],
        check=True,
    )
    output = tmp_path / "dataset"
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert len(train) == 480
    assert len(validation) == 132
    assert not (output / "test.jsonl").exists()
    assert {row["prompt"] for row in train}.isdisjoint(
        {row["prompt"] for row in validation}
    )


def test_stage2_contains_all_target_families(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_assistant_curriculum_v1_1.py"),
            "--output-dir",
            str(tmp_path / "dataset"),
        ],
        check=True,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset/validation.jsonl").read_text().splitlines()
    ]
    assert {row["task_family"] for row in rows} == {
        "assistant_core",
        "instruction_following",
        "ivoire_grounded",
        "reading_comprehension",
        "uncertainty_refusal",
    }
