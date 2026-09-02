import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load("scripts/data/build_assistant_curriculum_v1.py")


def test_curriculum_has_all_capabilities_and_no_prompt_leak():
    rows = {split: BUILDER.build_rows(split) for split in ("train", "validation")}
    BUILDER.validate(rows)
    expected = {"assistant_core", "uncertainty_refusal", "ivoire_grounded", "reading_comprehension", "instruction_following"}
    assert {row["task_family"] for row in rows["train"]} == expected
    assert {row["task_family"] for row in rows["validation"]} == expected


def test_facts_are_sourced_and_targets_are_short():
    for split in ("train", "validation"):
        for item in BUILDER.build_rows(split):
            assert len(item["target"]) <= 240
            if item["task_family"] in {"ivoire_grounded", "reading_comprehension"}:
                assert item["source_url"].startswith("https://")


def test_no_test_split_is_constructed():
    assert set(BUILDER.build_rows("train")[0]) >= {"prompt", "target", "split"}
    try:
        BUILDER.build_rows("test")
    except KeyError:
        pass
    else:
        raise AssertionError("un split test ne doit pas être construit")
