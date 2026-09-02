#!/usr/bin/env python3
"""Encode le supplément v1.0 avec le BPE v0.4, dans un fichier par domaine."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


DATASET_ID = "ivoireslm_corpus_supplement_v1.0.0"
TOKENIZED_ID = "ivoireslm_corpus_supplement_v1.0.0_bpe_v0.4"
BUCKETS = {
    "wikipedia_fr": "wikipedia_fr",
    "wikipedia_en": "wikipedia_en",
    "gsm8k_train": "mathematics_reasoning",
    "aqua_train_dev": "mathematics_reasoning",
    "deepseek_harness": "code_agents",
    "cisa_kev": "cybersecurity_defensive",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"JSONL invalide {path}:{line_number}") from error


def encode(supplement_root: Path, tokenizer_path: Path, output_root: Path) -> dict:
    building = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or building.exists():
        raise FileExistsError(f"refus d'écraser {output_root}")
    for split in ("train", "validation"):
        if not (supplement_root / f"{split}.jsonl").is_file():
            raise FileNotFoundError(supplement_root / f"{split}.jsonl")
    if (supplement_root / "test.jsonl").exists():
        raise RuntimeError("le supplément ne doit pas contenir de split test")
    source_report = json.loads((supplement_root / "report.json").read_text())
    if source_report.get("dataset_id") != DATASET_ID or source_report.get("test_created") is not False:
        raise ValueError("rapport du supplément inattendu")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 8192:
        raise ValueError("le tokenizer doit contenir exactement 8192 tokens")
    bos, eos, unknown = (tokenizer.token_to_id(token) for token in ("<BOS>", "<EOS>", "<UNK>"))
    if None in (bos, eos, unknown):
        raise ValueError("tokens spéciaux absents")

    building.mkdir(parents=True)
    reports = {}
    try:
        (building / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
        for split in ("train", "validation"):
            streams = {
                bucket: (building / f"{split}.{bucket}.uint16.bin").open("wb")
                for bucket in sorted(set(BUCKETS.values()))
            }
            stats = {bucket: Counter() for bucket in streams}
            try:
                for row in iter_jsonl(supplement_root / f"{split}.jsonl"):
                    source_id = row.get("source_id")
                    if source_id not in BUCKETS:
                        raise ValueError(f"source inconnue: {source_id}")
                    bucket = BUCKETS[source_id]
                    text = row["text"]
                    ids = [bos, *tokenizer.encode(text, add_special_tokens=False).ids, eos]
                    np.asarray(ids, dtype=np.uint16).tofile(streams[bucket])
                    stats[bucket]["documents"] += 1
                    stats[bucket]["characters"] += len(text)
                    stats[bucket]["tokens"] += len(ids)
                    stats[bucket]["unknown_tokens"] += ids.count(unknown)
            finally:
                for stream in streams.values():
                    stream.close()
            reports[split] = {}
            for bucket, values in stats.items():
                path = building / f"{split}.{bucket}.uint16.bin"
                if values["tokens"] < 514:
                    raise RuntimeError(f"pas assez de tokens pour {split}/{bucket}")
                reports[split][bucket] = {
                    **dict(values),
                    "unknown_rate": values["unknown_tokens"] / values["tokens"],
                    "sha256": sha256_file(path),
                    "file": path.name,
                }
                print(f"{split}/{bucket}: {values['tokens']:,} tokens", flush=True)

        report = {
            "tokenized_id": TOKENIZED_ID,
            "source_dataset_id": DATASET_ID,
            "tokenizer_id": "ivoireslm_bpe_v0.4",
            "tokenizer_sha256": sha256_file(building / "tokenizer.json"),
            "source_report_sha256": sha256_file(supplement_root / "report.json"),
            "test_created": False,
            "buckets": reports,
        }
        (building / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        checksums = []
        for path in sorted(building.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                checksums.append(f"{sha256_file(path)}  {path.name}")
        (building / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
        building.rename(output_root)
        return report
    except Exception:
        import shutil
        shutil.rmtree(building, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = encode(args.supplement_root, args.tokenizer, args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nSupplément tokenisé par domaine ✅")
    print("Aucun test créé ou ouvert ✅")


if __name__ == "__main__":
    main()
