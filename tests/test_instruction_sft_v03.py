from data.assistant_core_v03 import INTENTS, assistant_core_rows


def test_v03_assistant_prompts_are_disjoint_and_provenanced():
    splits = {split: assistant_core_rows(split) for split in ("train", "validation", "test")}
    assert len(splits["train"]) == len(INTENTS) * 6
    assert len(splits["validation"]) == len(INTENTS)
    assert len(splits["test"]) == len(INTENTS)
    prompt_sets = {split: {row["prompt"] for row in rows} for split, rows in splits.items()}
    assert prompt_sets["train"].isdisjoint(prompt_sets["validation"])
    assert prompt_sets["train"].isdisjoint(prompt_sets["test"])
    assert prompt_sets["validation"].isdisjoint(prompt_sets["test"])
    assert all(
        row["review_status"] == "machine_curated_pending_human_spot_check"
        for rows in splits.values()
        for row in rows
    )


def test_v03_intents_have_eight_unique_prompts_and_three_responses():
    for intent in INTENTS:
        assert len(intent["prompts"]) == 8
        assert len(set(intent["prompts"])) == 8
        assert len(intent["responses"]) == 3
        assert all(response.endswith((".", "?", "!")) for response in intent["responses"])
