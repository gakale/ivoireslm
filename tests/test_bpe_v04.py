import importlib.util
import json
import sys
from pathlib import Path

import pytest


pytest.importorskip("tokenizers")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/tokenizer/train_bpe_v04.py"
SPEC = importlib.util.spec_from_file_location("train_bpe_v04", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_bpe_v04_roundtrip(tmp_path):
    source = tmp_path / "train.jsonl"
    texts = [
        "Le français ivoirien naturel doit rester présent dans le corpus.\n" * 20,
        "Une démonstration mathématique explique chaque étape clairement.\n" * 20,
    ]
    with source.open("w", encoding="utf-8") as stream:
        for index, text in enumerate(texts):
            stream.write(json.dumps({"document_id": index, "text": text}) + "\n")
    tokenizer = MODULE.build_tokenizer(source, vocab_size=300)
    sample = "À Abidjan, 12 + 7 = 19."
    ids = tokenizer.encode(sample, add_special_tokens=False).ids
    assert tokenizer.decode(ids, skip_special_tokens=False) == sample
