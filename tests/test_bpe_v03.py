import importlib.util
import json
import sys
from pathlib import Path

import pytest


pytest.importorskip("tokenizers")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/tokenizer/train_bpe_v03.py"
SPEC = importlib.util.spec_from_file_location("train_bpe_v03", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_documents(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index, text in enumerate(texts):
            stream.write(
                json.dumps(
                    {"document_id": f"doc-{index}", "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_bpe_roundtrip_and_uint16_encoding(tmp_path):
    source = tmp_path / "train.jsonl"
    texts = [
        "À Abidjan, une élève étudie les mathématiques et l'agriculture.\n" * 20,
        "La Côte d’Ivoire développe une documentation française naturelle.\n" * 20,
    ]
    write_documents(source, texts)
    tokenizer = MODULE.build_tokenizer(source, vocab_size=300)
    sample = "À Abidjan, l’agriculture ivoirienne évolue."
    ids = tokenizer.encode(sample, add_special_tokens=False).ids
    assert tokenizer.decode(ids, skip_special_tokens=False) == sample

    destination = tmp_path / "train.uint16.bin"
    report = MODULE.encode_split(tokenizer, source, destination, batch_size=1)
    assert destination.stat().st_size == report["tokens"] * 2
    assert report["documents"] == 2
    assert report["unknown_tokens"] == 0
