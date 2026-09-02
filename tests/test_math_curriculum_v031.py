import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/build_math_curriculum_v031.py"
SPEC = importlib.util.spec_from_file_location("build_math_curriculum_v031", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stage1_is_deterministic_verified_and_disjoint():
    first = MODULE.build_rows()
    second = MODULE.build_rows()
    assert first == second
    MODULE.validate(first)
    assert len(first["train"]) == 448
    assert len(first["validation"]) == 114
    assert all(MODULE.verify_row(row) for rows in first.values() for row in rows)


def test_stage1_has_only_easy_operations_and_canonical_targets():
    rows = MODULE.build_rows()
    for split_rows in rows.values():
        for row in split_rows:
            values = row["verification"]
            assert row["target"] == f" {values['result']}\n"
            if row["task_family"] == "addition_easy":
                assert 0 <= values["left"] <= 20
                assert 0 <= values["right"] <= 20
            else:
                assert 2 <= values["left"] <= 12
                assert 2 <= values["right"] <= 12


def test_stage1_has_no_test_split():
    assert set(MODULE.build_rows()) == {"train", "validation"}
