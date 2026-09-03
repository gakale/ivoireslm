"""Enregistrement explicite et auditable des retours humains IvoireSLM."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid


RATINGS = (
    "Correcte",
    "Partiellement correcte",
    "Incorrecte",
    "Je ne sais pas évaluer",
)
CATEGORIES = (
    "conversation_ordinaire",
    "orthographe_formulation",
    "connaissance_ivoirienne",
    "identite_limites",
    "prudence_incertain",
    "traduction",
    "instruction",
    "autre",
)
_WRITE_LOCK = threading.Lock()


def create_record(
    interaction: dict,
    *,
    rating: str,
    correction: str,
    category: str,
    consent: bool,
    checkpoint_sha256: str,
) -> dict:
    """Valide un retour ; aucune donnée n'est enregistrée sans consentement."""
    if not consent:
        raise ValueError("le consentement est nécessaire pour enregistrer ce retour")
    if rating not in RATINGS:
        raise ValueError("jugement invalide")
    if category not in CATEGORIES:
        raise ValueError("catégorie invalide")
    required = {"question", "response", "mode", "route", "status", "model_id", "checkpoint_step"}
    if not required.issubset(interaction) or not str(interaction["question"]).strip():
        raise ValueError("génère d'abord une réponse")
    correction = correction.strip()
    if rating in {"Incorrecte", "Partiellement correcte"} and not correction:
        raise ValueError("écris la réponse corrigée avant d'enregistrer")
    return {
        "schema_version": "ivoireslm.human-feedback.v1",
        "feedback_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": str(interaction["question"]).strip(),
        "model_response": str(interaction["response"]).strip(),
        "mode": interaction["mode"],
        "route": interaction["route"],
        "status": interaction["status"],
        "model_id": interaction["model_id"],
        "checkpoint_step": int(interaction["checkpoint_step"]),
        "checkpoint_sha256": checkpoint_sha256,
        "category": category,
        "rating": rating,
        "human_correction": correction or None,
        "consent_for_research_and_training": True,
        "submitter_attested_no_personal_data": True,
        "automatically_added_to_training": False,
    }


def append_record(path: Path, record: dict) -> int:
    """Ajoute une ligne JSON durablement et renvoie le nombre total de retours."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        with path.open("r", encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
