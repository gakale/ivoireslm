#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["IVOIRESLM_CORPUS_VERSION"] = "ivoireslm_corpus_v0.6.0"
os.environ["IVOIRESLM_TOKENIZER_VERSION"] = "character_v0.2"
os.environ["IVOIRESLM_TOKENIZER_ID"] = "ivoireslm_character_v0.2"
runpy.run_path(
    Path(__file__).with_name("train_character_v01.py"),
    run_name="__main__",
)
