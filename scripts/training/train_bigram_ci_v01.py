#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from models.bigram import BigramCountModel
from tokenizer.character import CharacterTokenizer, SPECIAL_TOKENS


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
TOKENIZER_ROOT = STORAGE_ROOT / "tokenizers/character_v0.1"
OUTPUT_ROOT = STORAGE_ROOT / "models/bigram_ci_v0.1"
CONFIG_PATH = REPOSITORY_ROOT / "configs/bigram.yaml"
CHUNK_TOKENS = 4_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokens(split: str):
    return np.memmap(TOKENIZER_ROOT / f"{split}.uint16.bin", mode="r", dtype=np.uint16)


def fit_train(model: BigramCountModel, tokens) -> int:
    transitions = len(tokens) - 1
    for start in range(0, transitions, CHUNK_TOKENS):
        stop = min(start + CHUNK_TOKENS, transitions)
        model.add_transitions(tokens[start:stop], tokens[start + 1 : stop + 1])
    model.fit()
    return transitions


def evaluate(model: BigramCountModel, tokens) -> float:
    transitions = len(tokens) - 1
    total_loss = 0.0
    for start in range(0, transitions, CHUNK_TOKENS):
        stop = min(start + CHUNK_TOKENS, transitions)
        count = stop - start
        total_loss += model.negative_log_likelihood(tokens[start:stop], tokens[start + 1 : stop + 1]) * count
    return total_loss / transitions


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    tokenizer_record = json.loads((TOKENIZER_ROOT / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(tuple(tokenizer_record["id_to_token"]))
    settings = config["training"]
    model = BigramCountModel(
        tokenizer.vocab_size,
        special_tokens=len(SPECIAL_TOKENS),
        alpha=float(settings["laplace_alpha"]),
    )

    train = load_tokens("train")
    validation = load_tokens("validation")
    test = load_tokens("test")
    train_transitions = fit_train(model, train)
    losses = {
        "uniform_baseline": math.log(tokenizer.vocab_size - len(SPECIAL_TOKENS)),
        "train": model.training_negative_log_likelihood(),
        "validation": evaluate(model, validation),
        "test": evaluate(model, test),
    }

    seed_text = "Côte d’Ivoire "
    prefix = tokenizer.encode(seed_text)
    generated = model.generate(prefix[-1], int(settings["generation_tokens"]), seed=int(settings["seed"]))
    sample = seed_text + tokenizer.decode(generated[1:], skip_special=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_ROOT / "bigram_ci_v0.1.npz"
    np.savez_compressed(
        checkpoint_path,
        counts=model.counts,
        log_probabilities=model.log_probabilities,
        vocab_size=np.asarray([tokenizer.vocab_size], dtype=np.int64),
        special_tokens=np.asarray([len(SPECIAL_TOKENS)], dtype=np.int64),
        alpha=np.asarray([model.alpha], dtype=np.float64),
    )
    sample_path = OUTPUT_ROOT / "sample.txt"
    sample_path.write_text(sample + "\n", encoding="utf-8")
    report = {
        "model_id": "bigram_ci_v0.1",
        "model_type": "maximum_likelihood_character_bigram",
        "training_data": str(TOKENIZER_ROOT / "train.uint16.bin"),
        "validation_data": str(TOKENIZER_ROOT / "validation.uint16.bin"),
        "test_data": str(TOKENIZER_ROOT / "test.uint16.bin"),
        "train_only_fit": True,
        "vocab_size": tokenizer.vocab_size,
        "parameters": tokenizer.vocab_size * tokenizer.vocab_size,
        "train_transitions": train_transitions,
        "laplace_alpha": model.alpha,
        "seed": int(settings["seed"]),
        "losses_nats": losses,
        "perplexities": {name: BigramCountModel.perplexity(loss) for name, loss in losses.items()},
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "sample_path": str(sample_path),
        "sample_sha256": sha256(sample_path),
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nÉCHANTILLON\n" + sample[:1000])


if __name__ == "__main__":
    main()
