import json

import pytest

from src.evaluation.human_feedback import append_record, create_record


INTERACTION = {
    "question": "Comment dit-on bonjour en dioula ?",
    "response": "Bonjour ! Comment puis-je t’aider ?",
    "mode": "Modèle seul",
    "route": "microivoire_transformer_v1.1_17m_assistant_stage3",
    "status": "fin EOS",
    "model_id": "microivoire_transformer_v1.1_17m_assistant_stage3",
    "checkpoint_step": 250,
}


def test_incorrect_feedback_requires_correction():
    with pytest.raises(ValueError):
        create_record(
            INTERACTION,
            rating="Incorrecte",
            correction="",
            category="traduction",
            consent=True,
            checkpoint_sha256="a" * 64,
        )


def test_feedback_is_explicit_and_never_auto_trains(tmp_path):
    record = create_record(
        INTERACTION,
        rating="Incorrecte",
        correction="Traduction à faire vérifier par un locuteur compétent.",
        category="traduction",
        consent=True,
        checkpoint_sha256="a" * 64,
    )
    path = tmp_path / "feedback.jsonl"
    assert append_record(path, record) == 1
    saved = json.loads(path.read_text().strip())
    assert saved["consent_for_research_and_training"] is True
    assert saved["automatically_added_to_training"] is False
