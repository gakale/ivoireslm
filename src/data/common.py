"""Utilitaires déterministes pour les artefacts de données IvoireSLM."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def format_number_fr(value: Any, *, max_decimals: int = 2) -> str:
    """Formate un nombre pour le texte français, sans zéros décimaux inutiles."""
    if value is None:
        return "non renseigné"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "non renseigné"

    if not math.isfinite(number):
        return "non renseigné"

    rendered = f"{number:,.{max_decimals}f}".rstrip("0").rstrip(".")
    return rendered.replace(",", "\u202f").replace(".", ",")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def upsert_jsonl(path: Path, record: dict[str, Any], *, key: str) -> None:
    """Insère ou remplace un enregistrement JSONL, en conservant les autres lignes."""
    if key not in record:
        raise KeyError(f"La clé d'identification {key!r} manque dans l'enregistrement")

    records: list[dict[str, Any]] = []
    replaced = False
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                existing_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL invalide dans {path}, ligne {line_number}") from error
            if not isinstance(existing_record, dict):
                raise ValueError(f"JSONL invalide dans {path}, ligne {line_number}")
            if existing_record.get(key) == record[key]:
                records.append(record)
                replaced = True
            else:
                records.append(existing_record)

    if not replaced:
        records.append(record)

    content = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records
    )
    _write_atomically(path, content)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_atomically(path, content)
