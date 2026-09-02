#!/usr/bin/env python3
"""Reconstruit un curriculum SFT traçable depuis les tokens BPE v0.4.

Cette branche ne prétend pas reproduire instruction_sft_v0.2. Elle extrait uniquement
des paires bilingues et des phrases WDI déjà présentes dans le corpus, puis ajoute des
exercices mathématiques déterministes et un petit noyau d'assistant versionné.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.assistant_core_v03 import assistant_core_rows
from data.instruction_sft import grounded_wdi_row
from data.math_sft import SFT_FAMILIES, generate_sft_exercise, verify_sft_exercise


MATH_SEED = 20260902
MATH_TRAIN_PER_FAMILY = 2_500
MATH_VALIDATION_PER_FAMILY = 250
WDI_TRAIN_LIMIT = 8_000
WDI_VALIDATION_LIMIT = 800
TRANSLATION_TRAIN_LIMIT = 8_000
TRANSLATION_VALIDATION_LIMIT = 1_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(rows: list[dict], salt: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{row['example_id']}".encode("utf-8")
        ).digest(),
    )


def decoded_documents(path: Path, tokenizer: Tokenizer, eos_id: int):
    """Décode document par document sans charger tout le split en mémoire."""
    tokens = np.memmap(path, mode="r", dtype=np.uint16)
    start = 0
    chunk_size = 1_000_000
    for chunk_start in range(0, len(tokens), chunk_size):
        chunk_end = min(len(tokens), chunk_start + chunk_size)
        chunk = np.asarray(tokens[chunk_start:chunk_end])
        for relative_end in np.flatnonzero(chunk == eos_id):
            end = chunk_start + int(relative_end)
            if end > start:
                document = tokenizer.decode(
                    np.asarray(tokens[start:end], dtype=np.uint16).tolist(),
                    skip_special_tokens=True,
                ).strip()
                if document:
                    yield document
            start = end + 1
    if start < len(tokens):
        document = tokenizer.decode(
            np.asarray(tokens[start:], dtype=np.uint16).tolist(),
            skip_special_tokens=True,
        ).strip()
        if document:
            yield document


def translation_pair(document: str) -> tuple[str, str] | None:
    match = re.search(
        r"(?:^|\n)Dioula\s*:\s*(?P<dyu>.+?)\nFrançais\s*:\s*(?P<fr>.+?)(?:\n|$)",
        document,
        flags=re.DOTALL,
    )
    if not match:
        return None
    dyu = " ".join(match.group("dyu").split())
    french = " ".join(match.group("fr").split())
    if not (2 <= len(dyu) <= 500 and 2 <= len(french) <= 500):
        return None
    if "�" in dyu + french:
        return None
    return dyu, french


def extract_corpus_sources(data_dir: Path, tokenizer: Tokenizer) -> tuple[list[str], dict[str, list[tuple[str, str]]]]:
    eos_id = tokenizer.token_to_id("<EOS>")
    if eos_id is None:
        raise RuntimeError("token <EOS> absent")
    wdi_sentences: dict[str, str] = {}
    translations = {"train": [], "validation": []}
    seen_pairs: set[tuple[str, str]] = set()
    for split in ("train", "validation"):
        path = data_dir / f"{split}.uint16.bin"
        for document in decoded_documents(path, tokenizer, eos_id):
            pair = translation_pair(document)
            if pair is not None and pair not in seen_pairs:
                seen_pairs.add(pair)
                translations[split].append(pair)
            for line in document.splitlines():
                line = line.strip()
                row = grounded_wdi_row(line, 0)
                if row is not None:
                    identity = hashlib.sha256(line.encode("utf-8")).hexdigest()
                    wdi_sentences[identity] = line
        print(
            f"Extraction {split} : {len(translations[split]):,} paires bilingues, "
            f"{len(wdi_sentences):,} phrases WDI cumulées",
            flush=True,
        )
    return list(wdi_sentences.values()), translations


def math_rows(split: str) -> list[dict]:
    count = MATH_TRAIN_PER_FAMILY if split == "train" else MATH_VALIDATION_PER_FAMILY
    offset = 0 if split == "train" else 1_000_000
    rows = []
    for family in SFT_FAMILIES:
        for local_index in range(count):
            record = generate_sft_exercise(
                family,
                offset + local_index,
                seed=MATH_SEED,
            )
            if not verify_sft_exercise(record):
                raise RuntimeError(record["example_id"])
            rows.append(
                {
                    "example_id": f"recovery_math_{split}_{family}_{local_index:07d}",
                    "task_family": "math_exact",
                    "task_subfamily": family,
                    "prompt": record["prompt"].removesuffix("Méthode :").rstrip()
                    + "\nRéponse exacte :",
                    "target": f" {record['answer']}\n",
                    "license": "CC0-1.0",
                    "source": "deterministic_math_recovery_v0.3",
                    "split": split,
                }
            )
    return rows


def translation_rows(pairs: list[tuple[str, str]], split: str, limit: int) -> list[dict]:
    candidates = []
    for dyu, french in pairs:
        identity = hashlib.sha256(f"{dyu}\0{french}".encode("utf-8")).hexdigest()[:24]
        candidates.extend(
            [
                {
                    "example_id": f"recovery_dyu_fr_{identity}",
                    "task_family": "translation_dyu_fr",
                    "prompt": f"Instruction : Traduis ce texte du dioula vers le français.\nTexte : {dyu}\nRéponse :",
                    "target": f" {french}\n",
                    "license": "CC-BY-SA-4.0",
                    "source": "koumankan_recovered_from_corpus_v0.4",
                    "split": split,
                },
                {
                    "example_id": f"recovery_fr_dyu_{identity}",
                    "task_family": "translation_fr_dyu",
                    "prompt": f"Instruction : Traduis ce texte du français vers le dioula.\nTexte : {french}\nRéponse :",
                    "target": f" {dyu}\n",
                    "license": "CC-BY-SA-4.0",
                    "source": "koumankan_recovered_from_corpus_v0.4",
                    "split": split,
                },
            ]
        )
    return stable_order(candidates, f"translation:{split}")[:limit]


def wdi_rows(sentences: list[str]) -> dict[str, list[dict]]:
    output = {"train": [], "validation": []}
    for index, sentence in enumerate(sorted(sentences)):
        row = grounded_wdi_row(sentence, index)
        if row is None:
            continue
        # Le corpus BPE récupéré ne contient qu'un petit sous-ensemble de phrases
        # WDI et le découpage historique par indicateur peut laisser la validation
        # vide. On effectue donc ici un split déterministe par phrase. Les prompts
        # restent strictement disjoints, même si un indicateur peut apparaître dans
        # les deux splits ; cette limite est déclarée dans report.json.
        bucket = int.from_bytes(
            hashlib.sha256(f"recovery-wdi:{sentence}".encode("utf-8")).digest()[:4],
            "big",
        ) % 100
        split = "validation" if bucket < 20 else "train"
        row["split"] = split
        row["example_id"] = f"recovery_wdi_{split}_{index:07d}"
        output[split].append(row)
    output["train"] = stable_order(output["train"], "wdi:train")[:WDI_TRAIN_LIMIT]
    output["validation"] = stable_order(
        output["validation"], "wdi:validation"
    )[:WDI_VALIDATION_LIMIT]
    return output


def validate(rows_by_split: dict[str, list[dict]]) -> None:
    identifiers = set()
    prompts = {}
    expected_families = {
        "assistant_core",
        "grounded_wdi",
        "math_exact",
        "translation_dyu_fr",
        "translation_fr_dyu",
    }
    for split, rows in rows_by_split.items():
        families = {row["task_family"] for row in rows}
        if families != expected_families:
            raise RuntimeError(f"familles {split} inattendues : {sorted(families)}")
        prompts[split] = set()
        for row in rows:
            if row["split"] != split or row["example_id"] in identifiers:
                raise RuntimeError(row["example_id"])
            identifiers.add(row["example_id"])
            prompts[split].add(" ".join(row["prompt"].casefold().split()))
    overlap = prompts["train"] & prompts["validation"]
    if overlap:
        raise RuntimeError(f"fuite train/validation : {len(overlap)}")


def remove_prompt_duplicates(rows_by_split: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Supprime les doublons internes et toute fuite exacte train/validation."""
    dropped: dict[str, Counter] = {
        "train_internal_duplicates": Counter(),
        "validation_internal_duplicates": Counter(),
        "validation_overlap_with_train": Counter(),
    }

    train_clean, train_prompts = [], set()
    for row in rows_by_split["train"]:
        prompt = " ".join(row["prompt"].casefold().split())
        if prompt in train_prompts:
            dropped["train_internal_duplicates"][row["task_family"]] += 1
            continue
        train_prompts.add(prompt)
        train_clean.append(row)

    validation_clean, validation_prompts = [], set()
    for row in rows_by_split["validation"]:
        prompt = " ".join(row["prompt"].casefold().split())
        if prompt in validation_prompts:
            dropped["validation_internal_duplicates"][row["task_family"]] += 1
            continue
        validation_prompts.add(prompt)
        if prompt in train_prompts:
            dropped["validation_overlap_with_train"][row["task_family"]] += 1
            continue
        validation_clean.append(row)

    rows_by_split["train"] = train_clean
    rows_by_split["validation"] = validation_clean
    return {
        reason: dict(sorted(counts.items()))
        for reason, counts in dropped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refus d'écraser {args.output_dir}")
    tokenizer = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))
    sentences, pairs = extract_corpus_sources(args.data_dir, tokenizer)
    grounded = wdi_rows(sentences)
    rows_by_split = {}
    for split in ("train", "validation"):
        translation_limit = (
            TRANSLATION_TRAIN_LIMIT if split == "train" else TRANSLATION_VALIDATION_LIMIT
        )
        rows_by_split[split] = (
            assistant_core_rows(split)
            + grounded[split]
            + math_rows(split)
            + translation_rows(pairs[split], split, translation_limit)
        )
        rows_by_split[split].sort(key=lambda row: row["example_id"])
    prompt_rows_dropped = remove_prompt_duplicates(rows_by_split)
    validate(rows_by_split)
    args.output_dir.mkdir(parents=True)
    split_reports = {}
    for split, rows in rows_by_split.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        split_reports[split] = {
            "examples": len(rows),
            "characters": sum(len(row["prompt"]) + len(row["target"]) for row in rows),
            "task_families": dict(sorted(Counter(row["task_family"] for row in rows).items())),
            "sha256": sha256(path),
        }
    report = {
        "dataset_id": "instruction_sft_v0.3_recovery",
        "status": "train_validation_only_test_not_created",
        "parent_checkpoint": "microivoire_transformer_v0.4_17m_sft_v0.2_step500",
        "reproducibility_note": "new branch; not an exact continuation of instruction_sft_v0.2",
        "math_seed": MATH_SEED,
        "sources": {
            "assistant_core": "curated teacher-synthetic CC0 prompts",
            "math_exact": "deterministic and programmatically verified",
            "translation": "Koumankan pairs recovered from the licensed v0.4 corpus",
            "grounded_wdi": "World Bank WDI sentences recovered from the v0.4 corpus",
        },
        "known_limits": {
            "grounded_wdi_split": (
                "deterministic sentence-level 80/20 split because the recovered WDI "
                "subset did not populate validation under the historical indicator-group split; "
                "exact prompts remain disjoint but indicators may occur in both splits"
            ),
            "grounded_wdi_recovered_sentences": len(sentences),
        },
        "prompt_rows_dropped": prompt_rows_dropped,
        "splits": split_reports,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
