#!/usr/bin/env python3
"""Assemble le supplément français naturel v1.1 sans données d'évaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


DATASET_ID = "ivoireslm_natural_french_supplement_v1.1.0"
ALLOWED_CONSENT_LICENSES = {"CC0-1.0", "CC-BY-4.0"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized_key(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wikipedia_documents(snapshot: Path) -> list[dict]:
    documents = []
    for row in read_jsonl(snapshot / "pages.jsonl"):
        documents.append({
            "document_id": f"frwiki-v03:{row['page_id']}",
            "group_id": f"frwiki:{row['page_id']}",
            "source_id": "wikipedia_fr_natural_v0.3",
            "source_url": row["url"],
            "title": row["title"],
            "language": "fr",
            "domain": "natural_french_open",
            "content_type": "natural_encyclopedic_text",
            "license": "CC BY-SA 4.0",
            "rights_tier": "B_SHAREALIKE",
            "attribution": row["attribution"],
            "split": row["split"],
            "text": row["text"].strip(),
        })
    return documents


def oasst_dialogues(snapshot: Path) -> list[dict]:
    documents = []
    for split in ("train", "validation"):
        messages = read_jsonl(snapshot / f"{split}.jsonl")
        by_id = {row["message_id"]: row for row in messages}
        for row in messages:
            if row["role"] != "assistant" or not row.get("parent_id"):
                continue
            parent = by_id.get(row["parent_id"])
            if not parent or parent["role"] != "prompter" or parent["group_id"] != row["group_id"]:
                continue
            text = f"Utilisateur : {parent['text'].strip()}\nAssistant : {row['text'].strip()}"
            documents.append({
                "document_id": f"oasst1-fr-pair:{parent['message_id']}:{row['message_id']}",
                "group_id": row["group_id"],
                "source_id": "openassistant_fr_v0.1",
                "source_url": row["source_url"],
                "title": "Dialogue OpenAssistant français",
                "language": "fr",
                "domain": "natural_french_conversation_open",
                "content_type": "human_assistant_conversation_pair",
                "license": "Apache-2.0",
                "rights_tier": "A_REDISTRIBUTABLE",
                "split": split,
                "text": text,
            })
    return documents


def consented_ivoirian_dialogues(path: Path | None) -> tuple[list[dict], Counter]:
    if path is None:
        return [], Counter()
    accepted, rejected = [], Counter()
    for index, row in enumerate(read_jsonl(path), 1):
        if row.get("consent_for_training") is not True:
            rejected["missing_training_consent"] += 1
            continue
        if row.get("no_personal_data") is not True:
            rejected["personal_data_not_cleared"] += 1
            continue
        if row.get("license") not in ALLOWED_CONSENT_LICENSES:
            rejected["license_not_allowed"] += 1
            continue
        if row.get("split") not in {"train", "validation"}:
            rejected["invalid_split"] += 1
            continue
        prompt, response = str(row.get("prompt", "")).strip(), str(row.get("response", "")).strip()
        if len(prompt) < 2 or len(response) < 2:
            rejected["empty_dialogue"] += 1
            continue
        identifier = str(row.get("id") or f"line-{index:06d}")
        accepted.append({
            "document_id": f"ivoirian-consented:{identifier}",
            "group_id": f"ivoirian-consented:{row.get('group_id') or identifier}",
            "source_id": "ivoirian_conversations_consented_v0.1",
            "source_url": None,
            "title": "Conversation ivoirienne consentie",
            "language": str(row.get("language") or "fr-CI"),
            "domain": "natural_ivoirian_conversation_verified",
            "content_type": "consented_human_conversation_pair",
            "license": row["license"],
            "rights_tier": "A_REDISTRIBUTABLE",
            "contributor_pseudonym": row.get("contributor_pseudonym"),
            "split": row["split"],
            "text": f"Utilisateur : {prompt}\nAssistant : {response}",
        })
    return accepted, rejected


def build(wikipedia: Path, oasst: Path, output: Path, ivoirian: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"refus d'écraser {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"construction inachevée présente : {building}")
    for required in (wikipedia / "pages.jsonl", oasst / "train.jsonl", oasst / "validation.jsonl"):
        if not required.is_file():
            raise FileNotFoundError(required)
    consented, rejected_consent = consented_ivoirian_dialogues(ivoirian)
    documents = wikipedia_documents(wikipedia) + oasst_dialogues(oasst) + consented
    documents.sort(key=lambda row: (row["split"], row["source_id"], row["document_id"]))

    unique, seen, duplicate_count = [], {}, 0
    for row in documents:
        text = row["text"].strip()
        key = normalized_key(text)
        if not key or key in seen:
            duplicate_count += 1
            continue
        seen[key] = row["document_id"]
        unique.append({
            **row,
            "text": text,
            "characters": len(text),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    group_splits = defaultdict(set)
    for row in unique:
        group_splits[row["group_id"]].add(row["split"])
    leaks = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaks:
        raise RuntimeError(f"fuite train/validation : {len(leaks)} groupes")

    building.mkdir(parents=True)
    split_stats, domains, sources = {}, Counter(), Counter()
    for split in ("train", "validation"):
        rows = [row for row in unique if row["split"] == split]
        path = building / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        split_stats[split] = {
            "documents": len(rows), "characters": sum(row["characters"] for row in rows),
            "sha256": sha256_file(path),
        }
        for row in rows:
            domains[row["domain"]] += row["characters"]
            sources[row["source_id"]] += row["characters"]
    report = {
        "dataset_id": DATASET_ID,
        "status": "candidate_train_validation_only_test_not_created",
        "documents": len(unique), "characters": sum(row["characters"] for row in unique),
        "domain_characters": dict(sorted(domains.items())),
        "source_characters": dict(sorted(sources.items())),
        "splits": split_stats, "exact_duplicates_removed": duplicate_count,
        "consented_ivoirian_rejections": dict(sorted(rejected_consent.items())),
        "group_leaks": 0, "test_created": False,
    }
    (building / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    building.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia-dir", type=Path, required=True)
    parser.add_argument("--oasst-dir", type=Path, required=True)
    parser.add_argument("--ivoirian-conversations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.wikipedia_dir, args.oasst_dir, args.output_dir, args.ivoirian_conversations)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nCandidat de français naturel v1.1 construit ✅")
    print("Aucun entraînement lancé ; aucun split test créé ✅")


if __name__ == "__main__":
    main()
