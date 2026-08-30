"""Nettoyage déterministe des corpus textuels en langues ivoiriennes."""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from typing import Any, Iterable


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


def deduplicate_parallel_records(
    records: Iterable[dict[str, Any]], languages: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Conserve un seul exemplaire de chaque tuple parallèle exact."""
    selected = {}
    for record in sorted(records, key=lambda row: row["record_id"]):
        texts = record.get("texts", {})
        if not all(clean_transcription(texts.get(language)) for language in languages):
            continue
        key = tuple(clean_transcription(texts[language]).casefold() for language in languages)
        selected.setdefault(key, record)
    return list(selected.values())


def leakage_components(
    records: list[dict[str, Any]], languages: tuple[str, ...]
) -> list[list[dict[str, Any]]]:
    """Regroupe les lignes partageant un texte, même indirectement, dans une langue."""
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    owners: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for language in languages:
            text_key = clean_transcription(record["texts"][language]).casefold()
            key = (language, text_key)
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return list(grouped.values())


def assign_components_to_splits(
    components: list[list[dict[str, Any]]],
    ratios: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    """Répartit des composantes indivisibles en approchant les ratios demandés."""
    if not ratios or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Les ratios doivent être positifs et totaliser 1")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("Les ratios doivent être strictement positifs")
    total = sum(len(component) for component in components)
    targets = {split: total * ratio for split, ratio in ratios.items()}
    assigned = {split: [] for split in ratios}

    def component_key(component: list[dict[str, Any]]) -> tuple[int, str]:
        identity = "\0".join(sorted(row["record_id"] for row in component))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return (-len(component), digest)

    for component in sorted(components, key=component_key):
        split = max(
            ratios,
            key=lambda name: (
                (targets[name] - len(assigned[name])) / targets[name],
                ratios[name],
                name,
            ),
        )
        assigned[split].extend(component)
    for rows in assigned.values():
        rows.sort(key=lambda row: row["record_id"])
    return assigned
