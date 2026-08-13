import hashlib
import json
from pathlib import Path


ROOT = Path.home() / "ivoireslm-storage" / "corpora" / "ivoireslm_corpus_v0.1.0"
MANIFEST = ROOT / "manifests" / "documents.jsonl"
REPORT = ROOT / "reports" / "quality_report.json"

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

split_groups = {}
for split in ("train", "validation", "test"):
    split_groups[split] = {row["group_id"] for row in rows if row["split"] == split}
if split_groups["train"] & split_groups["validation"]:
    errors.append("fuite train/validation")
if split_groups["train"] & split_groups["test"]:
    errors.append("fuite train/test")
if split_groups["validation"] & split_groups["test"]:
    errors.append("fuite validation/test")
if len(rows) != 13:
    errors.append(f"13 documents attendus, trouvé {len(rows)}")
if not report.get("quality_gate_passed"):
    errors.append("quality gate déclaré en échec")

if errors:
    raise SystemExit("AUDIT ÉCHOUÉ\n- " + "\n- ".join(errors))
print("AUDIT CORPUS v0.1.0 : OK")
print("Documents :", len(rows))
print("Phrases   :", report["lines"])
print("Faits     :", report["atomic_facts"])
print("Splits    :", {name: report["splits"][name]["documents"] for name in ("train", "validation", "test")})
