#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from models.contextual_mlp import ContextualCharacterMLP
from tokenizer.character import CharacterTokenizer, SPECIAL_TOKENS


STORAGE_ROOT = Path(os.environ.get("IVOIRESLM_STORAGE_ROOT", Path.home() / "ivoireslm-storage"))
TOKENIZER_ROOT = STORAGE_ROOT / "tokenizers/character_v0.1"
OUTPUT_ROOT = STORAGE_ROOT / "models/contextual_mlp_ci_v0.1"
CONFIG_PATH = REPOSITORY_ROOT / "configs/contextual_mlp.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokens(split: str):
    return np.memmap(TOKENIZER_ROOT / f"{split}.uint16.bin", mode="r", dtype=np.uint16)


def random_batch(tokens, context_size: int, batch_size: int, rng: np.random.Generator):
    starts = rng.integers(0, len(tokens) - context_size, size=batch_size)
    offsets = np.arange(context_size)
    contexts = np.asarray(tokens[starts[:, None] + offsets[None, :]], dtype=np.int64)
    targets = np.asarray(tokens[starts + context_size], dtype=np.int64)
    return torch.from_numpy(contexts), torch.from_numpy(targets)


@torch.inference_mode()
def evaluate(model, tokens, context_size: int, batch_size: int, batches: int, seed: int) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(batches):
        contexts, targets = random_batch(tokens, context_size, batch_size, rng)
        total += float(F.cross_entropy(model(contexts), targets).item())
    return total / batches


@torch.inference_mode()
def evaluate_complete(model, tokens, context_size: int, batch_size: int) -> float:
    """Évalue chaque cible du split, sans échantillonnage aléatoire."""
    model.eval()
    total_loss = 0.0
    total_examples = len(tokens) - context_size
    offsets = np.arange(context_size)
    for start in range(0, total_examples, batch_size):
        stop = min(start + batch_size, total_examples)
        starts = np.arange(start, stop)
        contexts = np.asarray(
            tokens[starts[:, None] + offsets[None, :]], dtype=np.int64
        )
        targets = np.asarray(tokens[starts + context_size], dtype=np.int64)
        loss = F.cross_entropy(
            model(torch.from_numpy(contexts)),
            torch.from_numpy(targets),
            reduction="sum",
        )
        total_loss += float(loss.item())
    return total_loss / total_examples


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_settings = config["model"]
    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    tokenizer_record = json.loads((TOKENIZER_ROOT / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(tuple(tokenizer_record["id_to_token"]))
    context_size = int(model_settings["context_size"])
    model = ContextualCharacterMLP(
        tokenizer.vocab_size,
        context_size=context_size,
        embedding_dim=int(model_settings["embedding_dim"]),
        hidden_dim=int(model_settings["hidden_dim"]),
        dropout=float(model_settings["dropout"]),
        special_tokens=len(SPECIAL_TOKENS),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
    train = load_tokens("train")
    validation = load_tokens("validation")
    test = load_tokens("test")
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    best_validation_loss = math.inf
    best_step = 0
    best_state = None

    for step in range(1, int(training["max_steps"]) + 1):
        model.train()
        contexts, targets = random_batch(
            train, context_size, int(training["batch_size"]), rng
        )
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(contexts), targets)
        loss.backward()
        optimizer.step()
        if step == 1 or step % int(training["log_every"]) == 0:
            validation_loss = evaluate(
                model,
                validation,
                context_size,
                int(training["batch_size"]),
                min(32, int(training["evaluation_batches"])),
                seed + 10,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"étape {step:4d}/{training['max_steps']} | "
                f"train {loss.item():.4f} | validation {validation_loss:.4f} | "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("aucun état de validation n'a été enregistré")
    model.load_state_dict(best_state)

    losses = {
        "validation": evaluate_complete(
            model, validation, context_size, int(training["batch_size"])
        ),
        "test": evaluate_complete(
            model, test, context_size, int(training["batch_size"])
        ),
    }
    train_evaluation_loss = evaluate(
        model,
        train,
        context_size,
        int(training["batch_size"]),
        int(training["evaluation_batches"]),
        seed + 3,
    )
    losses = {"train_sample": train_evaluation_loss, **losses}

    seed_text = "Contexte ivoirien : Côte d’Ivoire "
    prefix = tokenizer.encode(seed_text)
    generated = model.generate(
        prefix,
        int(training["generation_tokens"]),
        temperature=float(training["temperature"]),
        generator=torch.Generator().manual_seed(seed),
    )
    sample = tokenizer.decode(generated, skip_special=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_ROOT / "contextual_mlp_ci_v0.1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": dict(model_settings),
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "seed": seed,
        },
        checkpoint_path,
    )
    sample_path = OUTPUT_ROOT / "sample.txt"
    sample_path.write_text(sample + "\n", encoding="utf-8")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "model_id": "contextual_mlp_ci_v0.1",
        "model_type": "fixed_context_character_mlp",
        "device": "cpu",
        "training_data": str(TOKENIZER_ROOT / "train.uint16.bin"),
        "validation_data": str(TOKENIZER_ROOT / "validation.uint16.bin"),
        "test_data": str(TOKENIZER_ROOT / "test.uint16.bin"),
        "train_only_fit": True,
        "vocab_size": tokenizer.vocab_size,
        "parameters": parameters,
        "context_size": context_size,
        "embedding_dim": int(model_settings["embedding_dim"]),
        "hidden_dim": int(model_settings["hidden_dim"]),
        "dropout": float(model_settings["dropout"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "training_steps": int(training["max_steps"]),
        "training_examples_seen": int(training["batch_size"]) * int(training["max_steps"]),
        "evaluation_method": "complete sequential evaluation for validation and test; deterministic sample for train",
        "best_validation_step": best_step,
        "best_validation_sample_loss": best_validation_loss,
        "seed": seed,
        "elapsed_seconds": time.monotonic() - started,
        "losses_nats": losses,
        "perplexities": {name: math.exp(loss) for name, loss in losses.items()},
        "bigram_test_perplexity_reference": 15.54371160508373,
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
