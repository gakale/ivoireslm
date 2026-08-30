import importlib.util
import json
import sys
from pathlib import Path

import pytest


pytest.importorskip("tokenizers")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/tokenizer/train_bpe_multilingual_pilot_v01.py"
SPEC = importlib.util.spec_from_file_location("train_bpe_multilingual_pilot_v01", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_multilingual_pilot_roundtrip(tmp_path):
    french = tmp_path / "fr.jsonl"
    multilingual = tmp_path / "multi.jsonl"
    french.write_text(json.dumps({"text": "Le français ivoirien naturel. " * 30}) + "\n")
    multilingual.write_text(
        json.dumps({"text": "Dioula : I ni ce\nFrançais : Merci " * 30}, ensure_ascii=False) + "\n"
        + json.dumps({"text": "Ɲanmiɛn su lɔʼn ti ble. " * 30}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tokenizer = MODULE.build_tokenizer([french, multilingual], vocab_size=300)
    for sample in ("À Abidjan.", "I ni ce", "Ɲanmiɛn su lɔʼn"):
        ids = tokenizer.encode(sample, add_special_tokens=False).ids
        assert tokenizer.decode(ids, skip_special_tokens=False) == sample
