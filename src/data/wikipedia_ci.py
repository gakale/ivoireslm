from __future__ import annotations

import hashlib
import re
import unicodedata


SPACE_RE = re.compile(r"[ \t]+")


def clean_wikipedia_extract(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = []
    blank = False
    for raw_line in text.splitlines():
        line = SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if lines and not blank:
                lines.append("")
            blank = True
            continue
        if line.startswith(("Voir aussi", "Notes et références", "Liens externes")):
            break
        lines.append(line)
        blank = False
    return "\n".join(lines).strip() + "\n"


def split_for_page(page_id: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"frwiki:{page_id}".encode("utf-8")).digest()[:4], "big") % 100
    if bucket < 5:
        return "test"
    if bucket < 10:
        return "validation"
    return "train"


def is_useful_page(title: str, text: str) -> bool:
    excluded_prefixes = ("Liste de", "Liste des", "Chronologie de", "Modèle:")
    return not title.startswith(excluded_prefixes) and len(text) >= 500

