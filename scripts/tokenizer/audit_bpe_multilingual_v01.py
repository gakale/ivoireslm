#!/usr/bin/env python3
"""Mesure la fragmentation du BPE v0.3 sur le dioula et le baoulé."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

from tokenizers import Tokenizer


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
DEFAULT_TOKENIZER = STORAGE_ROOT / "tokenizers/bpe_v0.3/tokenizer.json"
DEFAULT_BUNDLE = STORAGE_ROOT / "derived/ivoirian_multilingual_bundle_v0.1"
DEFAULT_OUTPUT = STORAGE_ROOT / "reports/bpe_v0.3_multilingual_audit_v0.1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def audit_texts(tokenizer: Tokenizer, texts: list[str]) -> dict:
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    lengths = [len(encoding.ids) for encoding in encodings]
    tokens = sum(lengths)
    characters = sum(map(len, texts))
    words = sum(len(text.split()) for text in texts)
    unknown_id = tokenizer.token_to_id("<UNK>")
    unknown_tokens = sum(encoding.ids.count(unknown_id) for encoding in encodings)
    roundtrip_failures = sum(
        tokenizer.decode(encoding.ids, skip_special_tokens=False) != text
        for text, encoding in zip(texts, encodings)
    )
    return {
        "documents": len(texts),
        "characters": characters,
        "words": words,
        "tokens": tokens,
        "characters_per_token": characters / tokens,
        "tokens_per_word": tokens / words,
        "unknown_tokens": unknown_tokens,
        "unknown_rate": unknown_tokens / tokens,
        "roundtrip_failures": roundtrip_failures,
        "tokens_per_document_median": statistics.median(lengths),
        "tokens_per_document_p95": percentile(lengths, 0.95),
    }


def load_bundle(path: Path) -> dict[str, list[str]]:
    texts = {"bci": [], "dyu": [], "fr": []}
    for split in ("train", "validation", "test"):
        for line in (path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["language"] == "bci":
                texts["bci"].append(row["text"])
            else:
                first, second = row["text"].split("\n", 1)
                texts["dyu"].append(first.removeprefix("Dioula : "))
                texts["fr"].append(second.removeprefix("Français : "))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.tokenizer.is_file():
        raise FileNotFoundError(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    texts = load_bundle(args.bundle)
    languages = {language: audit_texts(tokenizer, rows) for language, rows in texts.items()}
    french_efficiency = languages["fr"]["characters_per_token"]
    for language, metrics in languages.items():
        metrics["relative_token_cost_vs_french"] = french_efficiency / metrics["characters_per_token"]
    report = {
        "audit_id": "bpe_multilingual_audit_v0.1",
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": sha256(args.tokenizer),
        "bundle_path": str(args.bundle),
        "bundle_report_sha256": sha256(args.bundle / "report.json"),
        "languages": languages,
        "interpretation": {
            "relative_token_cost_1": "même coût que le français",
            "relative_token_cost_above_1": "langue davantage fragmentée que le français",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
