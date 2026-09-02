#!/usr/bin/env python3
"""Fige les messages français humains d'OASST1, sans créer de split test."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path


SNAPSHOT_ID = "openassistant_fr_v0.1"
REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
BASE_URL = f"https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/{REVISION}"
FILES = {
    "train": {
        "name": "train-00000-of-00001-b42a775f407cee45.parquet",
        "url": f"{BASE_URL}/data/train-00000-of-00001-b42a775f407cee45.parquet",
        "sha256": "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e",
    },
    "validation": {
        "name": "validation-00000-of-00001-134b8fd0c89408b6.parquet",
        "url": f"{BASE_URL}/data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
        "sha256": "24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416",
    },
}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
SECRET_RES = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(text: str) -> tuple[str, list[str]]:
    findings = []
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if EMAIL_RE.search(text):
        findings.append("email")
        text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    for pattern in SECRET_RES:
        if pattern.search(text):
            findings.append("secret_shape")
            text = pattern.sub("[REDACTED_SECRET]", text)
    return text, sorted(set(findings))


def accepted_record(row: dict, split: str) -> tuple[dict | None, str | None]:
    if str(row.get("lang", "")).casefold() not in {"fr", "fra", "fr-fr"}:
        return None, "not_french"
    if bool(row.get("deleted")) or bool(row.get("synthetic")):
        return None, "deleted_or_synthetic"
    if str(row.get("review_result")).casefold() not in {"true", "1"}:
        return None, "review_not_passed"
    text, redactions = clean_text(str(row.get("text") or ""))
    if not 20 <= len(text) <= 6_000:
        return None, "invalid_length"
    message_id = str(row.get("message_id") or "").strip()
    tree_id = str(row.get("message_tree_id") or "").strip()
    role = str(row.get("role") or "").strip()
    if not message_id or not tree_id or role not in {"prompter", "assistant"}:
        return None, "invalid_structure"
    return {
        "document_id": f"oasst1:{message_id}",
        "group_id": f"oasst1-tree:{tree_id}",
        "message_id": message_id,
        "parent_id": (
            None
            if row.get("parent_id") is None or str(row.get("parent_id")).casefold() == "nan"
            else str(row.get("parent_id"))
        ),
        "message_tree_id": tree_id,
        "role": role,
        "language": "fr",
        "domain": "natural_french_conversation_open",
        "content_type": "human_assistant_conversation_message",
        "split": split,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "redactions": redactions,
        "review_count": int(row.get("review_count") or 0),
        "rank": None,
        "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "source_revision": REVISION,
        "license": "Apache-2.0",
        "rights_tier": "A_REDISTRIBUTABLE",
    }, None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "IvoireSLM/1.1 corpus research"})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1 << 20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(f"snapshot déjà présent : {output}")
    (partial / "raw").mkdir(parents=True, exist_ok=True)

    import pandas as pd

    rows_by_split: dict[str, list[dict]] = {}
    rejected = Counter()
    redactions = Counter()
    source_files = []
    for split, metadata in FILES.items():
        raw = partial / "raw" / metadata["name"]
        if not raw.exists():
            print(f"Téléchargement OASST1 {split}…", flush=True)
            download(metadata["url"], raw)
        actual = sha256_file(raw)
        if actual != metadata["sha256"]:
            raise ValueError(f"SHA256 OASST1 invalide : {split}")
        source_files.append({**metadata, "split": split, "bytes": raw.stat().st_size})
        accepted = []
        for row in pd.read_parquet(raw).to_dict(orient="records"):
            record, reason = accepted_record(row, split)
            if record is None:
                rejected[reason] += 1
                continue
            accepted.append(record)
            redactions.update(record["redactions"])
        accepted.sort(key=lambda row: (row["group_id"], row["message_id"]))
        rows_by_split[split] = accepted

    train_groups = {row["group_id"] for row in rows_by_split["train"]}
    validation_groups = {row["group_id"] for row in rows_by_split["validation"]}
    if train_groups & validation_groups:
        raise RuntimeError("fuite d'arbres OASST1 entre train et validation")
    for split, rows in rows_by_split.items():
        write_jsonl(partial / f"{split}.jsonl", rows)
    report = {
        "snapshot_id": SNAPSHOT_ID,
        "status": "train_validation_only_test_not_created",
        "source_revision": REVISION,
        "license": "Apache-2.0",
        "source_files": source_files,
        "splits": {
            split: {
                "messages": len(rows),
                "trees": len({row["group_id"] for row in rows}),
                "characters": sum(row["characters"] for row in rows),
                "sha256": sha256_file(partial / f"{split}.jsonl"),
            }
            for split, rows in rows_by_split.items()
        },
        "rejected": dict(sorted(rejected.items())),
        "redactions": dict(sorted(redactions.items())),
        "group_leaks": 0,
        "test_created": False,
    }
    (partial / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.rename(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nSnapshot français OASST1 terminé ✅")
    print("Aucun split test créé ✅")


if __name__ == "__main__":
    main()
