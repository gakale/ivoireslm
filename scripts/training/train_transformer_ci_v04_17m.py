#!/usr/bin/env python3
"""Entraîne le Transformer 17,13 M v0.4 sur le corpus diversifié v0.9."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from train_transformer_ci_v03_17m import (
    MicroIvoireTransformer17M,
    TrainingConfig,
    evaluate_complete,
    generate,
    load_tokens,
    main,
    random_batch,
)


def v04_config() -> TrainingConfig:
    return TrainingConfig(
        model_id="microivoire_transformer_v0.4_17m",
        tokenizer_id="ivoireslm_bpe_v0.4",
        corpus_id="ivoireslm_corpus_v0.9.0",
        seed=20260830,
    )


if __name__ == "__main__":
    main(v04_config(), default_output=Path("/content/checkpoints_v04_17m"))
