import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_stage3_is_disjoint_balanced_and_test_free(tmp_path):
    output = tmp_path / "stage3"
    subprocess.run([sys.executable, str(ROOT / "scripts/data/build_assistant_curriculum_v1_2.py"), "--output-dir", str(output)], check=True)
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert (len(train), len(validation)) == (866, 122)
    assert not (output / "test.jsonl").exists()
    assert {item["prompt"] for item in train}.isdisjoint({item["prompt"] for item in validation})
    train_upper = [item for item in train if item["task_subfamily"] == "uppercase"]
    validation_upper = [item for item in validation if item["task_subfamily"] == "uppercase"]
    assert len(train_upper) == 288
    assert len(validation_upper) == 24
