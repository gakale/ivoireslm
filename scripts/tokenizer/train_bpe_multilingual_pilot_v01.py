#!/usr/bin/env python3
"""Entraîne un BPE pilote sur le train français v0.7 et le train multilingue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import chain
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
DEFAULT_FRENCH_TRAIN = STORAGE_ROOT / "corpora/ivoireslm_corpus_v0.7.0/splits/train.jsonl"
DEFAULT_MULTILINGUAL_TRAIN = STORAGE_ROOT / "derived/ivoirian_multilingual_bundle_v0.1/train.jsonl"
DEFAULT_OUTPUT = STORAGE_ROOT / "tokenizers/bpe_multilingual_pilot_v0.1"
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_texts(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)["text"]


def source_statistics(path: Path) -> dict:
    documents = 0
    characters = 0
    for text in iter_texts(path):
        documents += 1
        characters += len(text)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "documents": documents,
        "characters": characters,
    }


def build_tokenizer(paths: list[Path], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>", byte_fallback=True, fuse_unk=False))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(chain.from_iterable(iter_texts(path) for path in paths), trainer=trainer)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--french-train", type=Path, default=DEFAULT_FRENCH_TRAIN)
    parser.add_argument("--multilingual-train", type=Path, default=DEFAULT_MULTILINGUAL_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vocab-size", type=int, default=8_192)
    args = parser.parse_args()
    paths = [args.french_train, args.multilingual_train]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"refus d'écraser {args.output}")
    building = args.output.with_name(args.output.name + ".building")
    if building.exists():
        raise FileExistsError(building)
    building.mkdir(parents=True)

    tokenizer = build_tokenizer(paths, args.vocab_size)
    if tokenizer.get_vocab_size() != args.vocab_size:
        raise AssertionError("taille de vocabulaire inattendue")
    tokenizer_path = building / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)

    samples = [
        "À Abidjan, l’agriculture ivoirienne évolue.",
        "Dioula : I ni ce\nFrançais : Merci",
        "Ɲanmiɛn su lɔʼn ti ble kɛ nzueʼn tɔ danʼn.",
    ]
    for sample in samples:
        encoding = tokenizer.encode(sample, add_special_tokens=False)
        if tokenizer.decode(encoding.ids, skip_special_tokens=False) != sample:
            raise AssertionError(f"aller-retour invalide : {sample!r}")

    report = {
        "tokenizer_id": "ivoireslm_bpe_multilingual_pilot_v0.1",
        "purpose": "comparison_only_not_production",
        "vocab_size": tokenizer.get_vocab_size(),
        "training_split_only": True,
        "normalization": "NFC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "byte_fallback": True,
        "special_tokens": SPECIAL_TOKENS,
        "sources": {
            "french_train": source_statistics(args.french_train),
            "multilingual_train": source_statistics(args.multilingual_train),
        },
        "tokenizer_path": str(args.output / "tokenizer.json"),
        "tokenizer_sha256": sha256(tokenizer_path),
        "roundtrip_samples_passed": len(samples),
    }
    (building / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    building.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

