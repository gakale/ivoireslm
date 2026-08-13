import hashlib
import json
import os
from pathlib import Path


VERSION = os.environ.get("IVOIRESLM_CORPUS_VERSION", "ivoireslm_corpus_v0.1.0")
EXPECTED_DOCUMENTS = int(os.environ.get("IVOIRESLM_EXPECTED_DOCUMENTS", "13"))
ROOT = Path.home() / "ivoireslm-storage" / "corpora" / VERSION
MANIFEST = ROOT / "manifests" / "documents.jsonl"
REPORT = ROOT / "reports" / "quality_report.json"
CHECKSUMS = ROOT / "SHA256SUMS"

rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
report = json.loads(REPORT.read_text(encoding="utf-8"))
errors = []
for row in rows:
    path = Path(row["corpus_path"])
    if not path.is_file():
        errors.append(f"fichier absent: {path}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["text_sha256"]:
        errors.append(f"hash invalide: {row['document_id']}")
    if row.get("rights_tier") != "A_REDISTRIBUTABLE":
        errors.append(f"droits non admissibles: {row['document_id']}")

split_groups = {}
for split in ("train", "validation", "test"):
    split_groups[split] = {row["group_id"] for row in rows if row["split"] == split}
if split_groups["train"] & split_groups["validation"]:
    errors.append("fuite train/validation")
if split_groups["train"] & split_groups["test"]:
    errors.append("fuite train/test")
if split_groups["validation"] & split_groups["test"]:
    errors.append("fuite validation/test")
if len(rows) != EXPECTED_DOCUMENTS:
    errors.append(f"{EXPECTED_DOCUMENTS} documents attendus, trouvé {len(rows)}")
if not report.get("quality_gate_passed"):
    errors.append("quality gate déclaré en échec")

for split in ("train", "validation", "test"):
    split_rows = [row for row in rows if row["split"] == split]
    expected_text = "".join(
        Path(row["corpus_path"]).read_text(encoding="utf-8").rstrip() + "\n\n"
        for row in split_rows
    )
    if (ROOT / "splits" / f"{split}.txt").read_text(encoding="utf-8") != expected_text:
        errors.append(f"concaténation invalide: {split}.txt")
    jsonl_rows = [
        json.loads(line)
        for line in (ROOT / "splits" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [row["document_id"] for row in jsonl_rows] != [row["document_id"] for row in split_rows]:
        errors.append(f"ordre ou contenu invalide: {split}.jsonl")
    for row in jsonl_rows:
        if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["text_sha256"]:
            errors.append(f"texte JSONL invalide: {row['document_id']}")

for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
    expected, relative = line.split("  ", 1)
    path = ROOT / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(f"checksum paquet invalide: {relative}")

if errors:
    raise SystemExit("AUDIT ÉCHOUÉ\n- " + "\n- ".join(errors))
print(f"AUDIT {VERSION} : OK")
print("Documents :", len(rows))
print("Phrases   :", report["lines"])
print("Faits     :", report["atomic_facts"])
print("Splits    :", {name: report["splits"][name]["documents"] for name in ("train", "validation", "test")})
