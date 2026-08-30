"""Nettoyage déterministe des corpus textuels en langues ivoiriennes."""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from typing import Any


WHITESPACE = re.compile(r"\s+")


def clean_transcription(value: Any) -> str:
    """Normalise une transcription sans modifier son orthographe."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFC", value)
    return WHITESPACE.sub(" ", value).strip()


def stable_record_id(dataset: str, split: str, row_index: int) -> str:
    payload = f"{dataset}\0{split}\0{row_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def extract_text_fields(row: dict[str, Any], field_languages: dict[str, str]) -> dict[str, str]:
    """Extrait uniquement les champs textuels explicitement autorisés."""
    extracted = {}
    for field, language in field_languages.items():
        text = clean_transcription(row.get(field))
        if text:
            extracted[language] = text
    return extracted


def quality_statistics(split_texts: dict[str, list[str]]) -> dict[str, Any]:
    """Mesure doublons et fuites exactes entre splits d'une langue."""
    text_sets = {split: set(texts) for split, texts in split_texts.items()}
    overlaps = {}
    for left, right in itertools.combinations(sorted(text_sets), 2):
        overlaps[f"{left}__{right}"] = len(text_sets[left] & text_sets[right])
    return {
        "internal_duplicates": {
            split: len(texts) - len(text_sets[split]) for split, texts in split_texts.items()
        },
        "cross_split_exact_overlaps": overlaps,
        "distinct_characters": len(set("".join(text for texts in split_texts.values() for text in texts))),
    }
