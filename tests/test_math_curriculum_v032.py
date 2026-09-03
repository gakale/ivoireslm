"""Contrôles déterministes du curriculum mathématique v0.3.2."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/data/build_math_curriculum_v032.py"
SPEC = importlib.util.spec_from_file_location("build_math_curriculum_v032", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_no_test_split_and_expected_sizes() -> None:
    rows = MODULE.build_rows()
    assert set(rows) == {"train", "validation", "diagnostic_generalization"}
    assert len(rows["train"]) == 1850
    assert len(rows["validation"]) == 562
    assert len(rows["diagnostic_generalization"]) == 64


def test_all_answers_are_programmatically_correct() -> None:
    rows = MODULE.build_rows()
    for split_rows in rows.values():
        for item in split_rows:
            values = item["verification"]
            expected = (
                values["left"] + values["right"]
                if values["operation"] == "addition"
                else values["left"] * values["right"]
            )
            assert expected == values["result"]
            assert item["target"].strip() == str(expected)


def test_validation_prompts_are_unseen_and_diagnostic_facts_are_disjoint() -> None:
    rows = MODULE.build_rows()
    train_prompts = {MODULE.normalized_prompt(item["prompt"]) for item in rows["train"]}
    validation_prompts = {
        MODULE.normalized_prompt(item["prompt"]) for item in rows["validation"]
    }
    diagnostic_prompts = {
        MODULE.normalized_prompt(item["prompt"])
        for item in rows["diagnostic_generalization"]
    }
    assert train_prompts.isdisjoint(validation_prompts)
    assert train_prompts.isdisjoint(diagnostic_prompts)

    train_multiplications = {
        (item["verification"]["left"], item["verification"]["right"])
        for item in rows["train"]
        if item["verification"]["operation"] == "multiplication"
    }
    diagnostic_facts = {
        (item["verification"]["left"], item["verification"]["right"])
        for item in rows["diagnostic_generalization"]
    }
    assert train_multiplications.isdisjoint(diagnostic_facts)
    assert min(min(pair) for pair in diagnostic_facts) == 13


def test_deterministic() -> None:
    assert MODULE.build_rows() == MODULE.build_rows()
