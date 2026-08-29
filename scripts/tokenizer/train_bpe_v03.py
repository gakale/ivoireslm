#!/usr/bin/env python3
"""Entraîne le BPE v0.3 sur train uniquement et encode les trois splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


STORAGE = Path(
    os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage")
)
DEFAULT_CORPUS = STORAGE / "corpora/ivoireslm_corpus_v0.7.0"
DEFAULT_OUTPUT = STORAGE / "tokenizers/bpe_v0.3"
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_documents(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)["text"]


def build_tokenizer(train_path: Path, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(
        models.BPE(unk_token="<UNK>", byte_fallback=True, fuse_unk=False)
    )
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iter_documents(train_path), trainer=trainer)
    return tokenizer


def encode_split(
    tokenizer: Tokenizer, source: Path, destination: Path, batch_size: int = 128
) -> dict:
    bos = tokenizer.token_to_id("<BOS>")
    eos = tokenizer.token_to_id("<EOS>")
    unknown = tokenizer.token_to_id("<UNK>")
    if None in (bos, eos, unknown):
        raise ValueError("tokens spéciaux absents du tokenizer")

    tokens = 0
    unknown_tokens = 0
    characters = 0
    documents = 0
    batch = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as binary:
        for text in iter_documents(source):
            batch.append(text)
            if len(batch) < batch_size:
                continue
            encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
            for text_value, encoding in zip(batch, encodings):
                ids = [bos, *encoding.ids, eos]
                np.asarray(ids, dtype=np.uint16).tofile(binary)
                tokens += len(ids)
                unknown_tokens += ids.count(unknown)
                characters += len(text_value)
                documents += 1
            batch.clear()
        if batch:
            encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
            for text_value, encoding in zip(batch, encodings):
                ids = [bos, *encoding.ids, eos]
                np.asarray(ids, dtype=np.uint16).tofile(binary)
                tokens += len(ids)
                unknown_tokens += ids.count(unknown)
                characters += len(text_value)
                documents += 1

    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "token_path": str(destination),
        "token_sha256": sha256_file(destination),
        "documents": documents,
        "characters": characters,
        "tokens": tokens,
        "characters_per_token": characters / tokens,
        "unknown_tokens": unknown_tokens,
        "unknown_rate": unknown_tokens / tokens,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vocab-size", type=int, default=8_192)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocabulaire trop grand pour uint16")
    if args.output_root.exists():
        raise FileExistsError(f"refus d'écraser {args.output_root}")
    for split in ("train", "validation", "test"):
        path = args.corpus_root / f"splits/{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)

    build_root = args.output_root.with_name(args.output_root.name + ".building")
    if build_root.exists():
        raise FileExistsError(build_root)
    build_root.mkdir(parents=True)

    train_path = args.corpus_root / "splits/train.jsonl"
    tokenizer = build_tokenizer(train_path, args.vocab_size)
    observed_vocab = tokenizer.get_vocab_size()
    if observed_vocab != args.vocab_size:
        raise AssertionError(
            f"vocabulaire attendu {args.vocab_size}, obtenu {observed_vocab}"
        )
    tokenizer_path = build_root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)

    sample = "À Abidjan, l’agriculture ivoirienne évolue en 2026."
    sample_ids = tokenizer.encode(sample, add_special_tokens=False).ids
    reconstructed = tokenizer.decode(sample_ids, skip_special_tokens=False)
    if reconstructed != sample:
        raise AssertionError(
            f"aller-retour invalide : {sample!r} != {reconstructed!r}"
        )

    split_reports = {}
    for split in ("train", "validation", "test"):
        split_reports[split] = encode_split(
            tokenizer,
            args.corpus_root / f"splits/{split}.jsonl",
            build_root / f"{split}.uint16.bin",
        )
        print(
            f"{split}: {split_reports[split]['tokens']:,} tokens, "
            f"{split_reports[split]['characters_per_token']:.2f} caractères/token",
            flush=True,
        )

    report = {
        "tokenizer_id": "ivoireslm_bpe_v0.3",
        "tokenizer_type": "byte_level_bpe",
        "tokenizers_version": __import__("tokenizers").__version__,
        "corpus_id": "ivoireslm_corpus_v0.7.0",
        "vocab_size": observed_vocab,
        "special_tokens": SPECIAL_TOKENS,
        "training_split_only": True,
        "normalization": "NFC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "byte_fallback": True,
        "roundtrip_sample_passed": True,
        "tokenizer_path": str(args.output_root / "tokenizer.json"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "splits": split_reports,
    }
    (build_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hash_lines = []
    for path in sorted(build_root.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            hash_lines.append(f"{sha256_file(path)}  {path.name}")
    (build_root / "SHA256SUMS").write_text(
        "\n".join(hash_lines) + "\n", encoding="utf-8"
    )
    build_root.replace(args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Tokenizer BPE validé : {args.output_root} ✅")


if __name__ == "__main__":
    main()
