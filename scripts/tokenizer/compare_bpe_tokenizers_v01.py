#!/usr/bin/env python3
"""Compare le BPE français v0.3 au pilote multilingue avec des seuils explicites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tokenizers import Tokenizer


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
DEFAULT_BASELINE = STORAGE_ROOT / "tokenizers/bpe_v0.3/tokenizer.json"
DEFAULT_CANDIDATE = STORAGE_ROOT / "tokenizers/bpe_multilingual_pilot_v0.1/tokenizer.json"
DEFAULT_CORPUS = STORAGE_ROOT / "corpora/ivoireslm_corpus_v0.7.0/splits"
DEFAULT_BASELINE_AUDIT = STORAGE_ROOT / "reports/bpe_v0.3_multilingual_audit_v0.1.json"
DEFAULT_CANDIDATE_AUDIT = STORAGE_ROOT / "reports/bpe_multilingual_pilot_v0.1_audit.json"
DEFAULT_OUTPUT = STORAGE_ROOT / "reports/bpe_multilingual_pilot_v0.1_comparison.json"
MAX_PRIMARY_TOKEN_REGRESSION_PERCENT = 0.25
MIN_LOW_RESOURCE_TOKEN_REDUCTION_PERCENT = 15.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent_change(before: int, after: int) -> float:
    return (after / before - 1.0) * 100.0


def encode_split(tokenizers: dict[str, Tokenizer], path: Path, batch_size: int = 128) -> dict:
    totals = {name: 0 for name in tokenizers}
    characters = 0
    documents = 0
    batch = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            batch.append(text)
            characters += len(text)
            documents += 1
            if len(batch) == batch_size:
                for name, tokenizer in tokenizers.items():
                    totals[name] += sum(
                        len(item.ids) for item in tokenizer.encode_batch(batch, add_special_tokens=False)
                    )
                batch.clear()
    if batch:
        for name, tokenizer in tokenizers.items():
            totals[name] += sum(
                len(item.ids) for item in tokenizer.encode_batch(batch, add_special_tokens=False)
            )
    return {
        "documents": documents,
        "characters": characters,
        "tokens": totals,
        "candidate_token_change_percent": percent_change(totals["baseline"], totals["candidate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--baseline-audit", type=Path, default=DEFAULT_BASELINE_AUDIT)
    parser.add_argument("--candidate-audit", type=Path, default=DEFAULT_CANDIDATE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    required = [args.baseline, args.candidate, args.baseline_audit, args.candidate_audit]
    required += [args.corpus / f"{split}.jsonl" for split in ("train", "validation", "test")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    tokenizers = {
        "baseline": Tokenizer.from_file(str(args.baseline)),
        "candidate": Tokenizer.from_file(str(args.candidate)),
    }
    corpus = {
        split: encode_split(tokenizers, args.corpus / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    baseline_audit = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
    candidate_audit = json.loads(args.candidate_audit.read_text(encoding="utf-8"))
    multilingual = {}
    for language in ("bci", "dyu", "fr"):
        before = baseline_audit["languages"][language]
        after = candidate_audit["languages"][language]
        multilingual[language] = {
            "baseline_tokens": before["tokens"],
            "candidate_tokens": after["tokens"],
            "token_reduction_percent": -percent_change(before["tokens"], after["tokens"]),
            "baseline_characters_per_token": before["characters_per_token"],
            "candidate_characters_per_token": after["characters_per_token"],
            "candidate_unknown_tokens": after["unknown_tokens"],
            "candidate_roundtrip_failures": after["roundtrip_failures"],
        }

    gates = {
        "primary_corpus_regression_within_limit": all(
            metrics["candidate_token_change_percent"] <= MAX_PRIMARY_TOKEN_REGRESSION_PERCENT
            for metrics in corpus.values()
        ),
        "dioula_reduction_reaches_minimum": multilingual["dyu"]["token_reduction_percent"]
        >= MIN_LOW_RESOURCE_TOKEN_REDUCTION_PERCENT,
        "baoule_reduction_reaches_minimum": multilingual["bci"]["token_reduction_percent"]
        >= MIN_LOW_RESOURCE_TOKEN_REDUCTION_PERCENT,
        "no_unknown_or_roundtrip_failure": all(
            metrics["candidate_unknown_tokens"] == 0
            and metrics["candidate_roundtrip_failures"] == 0
            for metrics in multilingual.values()
        ),
    }
    report = {
        "comparison_id": "bpe_multilingual_pilot_v0.1_comparison",
        "baseline_sha256": sha256(args.baseline),
        "candidate_sha256": sha256(args.candidate),
        "thresholds": {
            "maximum_primary_token_regression_percent": MAX_PRIMARY_TOKEN_REGRESSION_PERCENT,
            "minimum_low_resource_token_reduction_percent": MIN_LOW_RESOURCE_TOKEN_REDUCTION_PERCENT,
        },
        "primary_corpus": corpus,
        "multilingual": multilingual,
        "gates": gates,
        "accepted_for_future_multilingual_corpus": all(gates.values()),
        "accepted_for_current_17m_checkpoint": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

