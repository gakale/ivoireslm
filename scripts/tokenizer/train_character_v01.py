#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from array import array
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tokenizer.character import CharacterTokenizer, SPECIAL_TOKENS


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
CORPUS_VERSION = os.environ.get("IVOIRESLM_CORPUS_VERSION", "ivoireslm_corpus_v0.5.0")
TOKENIZER_VERSION = os.environ.get("IVOIRESLM_TOKENIZER_VERSION", "character_v0.1")
TOKENIZER_ID = os.environ.get("IVOIRESLM_TOKENIZER_ID", "ivoireslm_character_v0.1")
CORPUS_ROOT = STORAGE_ROOT / "corpora" / CORPUS_VERSION
OUTPUT_ROOT = STORAGE_ROOT / "tokenizers" / TOKENIZER_VERSION
CHUNK_SIZE = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_inventory(path: Path):
    characters = Counter()
    total = 0
    with path.open("r", encoding="utf-8") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            characters.update(chunk)
            total += len(chunk)
    return characters, total


def tokenize_split(path: Path, output_path: Path, tokenizer: CharacterTokenizer) -> dict:
    mapping = tokenizer.token_to_id
    unknown_id = mapping["<UNK>"]
    token_count = 0
    unknown_count = 0
    unknown_characters = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r", encoding="utf-8") as source, output_path.open("wb") as target:
        while chunk := source.read(CHUNK_SIZE):
            ids = array("H")
            for character in chunk:
                token_id = mapping.get(character, unknown_id)
                ids.append(token_id)
                if token_id == unknown_id:
                    unknown_count += 1
                    unknown_characters[character] += 1
            ids.tofile(target)
            token_count += len(ids)
    if output_path.stat().st_size != token_count * 2:
        raise RuntimeError(f"taille binaire incohérente pour {output_path}")
    return {
        "source_path": str(path),
        "source_sha256": sha256(path),
        "token_path": str(output_path),
        "token_sha256": sha256(output_path),
        "tokens": token_count,
        "unknown_tokens": unknown_count,
        "unknown_rate": unknown_count / token_count if token_count else 0.0,
        "unknown_characters": dict(sorted(unknown_characters.items(), key=lambda item: ord(item[0]))),
    }


def main() -> None:
    train_path = CORPUS_ROOT / "splits/train.txt"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    frequencies, train_characters = training_inventory(train_path)
    tokenizer = CharacterTokenizer.from_characters(frequencies)
    if tokenizer.vocab_size >= 65536:
        raise RuntimeError("le vocabulaire dépasse la capacité uint16")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tokenizer_path = OUTPUT_ROOT / "tokenizer.json"
    tokenizer_record = {
        "tokenizer_id": TOKENIZER_ID,
        "corpus_version": CORPUS_VERSION,
        "training_split": "train",
        "algorithm": "unicode_character_sorted_codepoint",
        "binary_dtype": "uint16_native_little_endian",
        "special_tokens": list(SPECIAL_TOKENS),
        "id_to_token": list(tokenizer.id_to_token),
        "vocab_size": tokenizer.vocab_size,
        "training_characters": train_characters,
        "character_frequencies": dict(sorted(frequencies.items(), key=lambda item: ord(item[0]))),
    }
    tokenizer_path.write_text(json.dumps(tokenizer_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split_reports = {}
    for split in ("train", "validation", "test"):
        split_reports[split] = tokenize_split(
            CORPUS_ROOT / f"splits/{split}.txt",
            OUTPUT_ROOT / f"{split}.uint16.bin",
            tokenizer,
        )

    sample = "La Côte d’Ivoire apprend, calcule et innove.\n"
    encoded = tokenizer.encode(sample, add_bos=True, add_eos=True)
    if tokenizer.decode(encoded[1:-1]) != sample:
        raise RuntimeError("échec du round-trip tokenizer")
    if split_reports["train"]["unknown_tokens"]:
        raise RuntimeError("le split train contient des caractères inconnus")

    report = {
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "training_split_only": True,
        "roundtrip_sample_passed": True,
        "splits": split_reports,
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
