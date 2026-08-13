import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


SAFE_NAME = re.compile(r"^[a-z0-9_]+$")
STORAGE = Path.home() / "ivoireslm-storage"


def file_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


parser = argparse.ArgumentParser()
parser.add_argument("domain")
parser.add_argument("source_id")
parser.add_argument("--snapshot", default="ivoiredata_2026-08-13_growth_v0.2")
parser.add_argument("--container", default="ivoiredata-api-1")
args = parser.parse_args()

for value in (args.domain, args.source_id, args.snapshot, args.container):
    if not SAFE_NAME.fullmatch(value.replace("-", "_")):
        raise SystemExit(f"Nom non sûr : {value}")

container_source = f"/app/data_lake/domains/{args.domain}/{args.source_id}"
probe = subprocess.run(
    ["docker", "exec", args.container, "test", "-d", container_source],
    check=False,
)
if probe.returncode:
    raise SystemExit(f"Source absente : {container_source}")

destination = (
    STORAGE / "snapshots" / args.snapshot / "data" / args.domain / args.source_id
)
if destination.exists():
    manifest = destination / "snapshot_sha256.json"
    if not manifest.exists():
        raise SystemExit(f"Destination existante sans manifeste : {destination}")
    print(f"Snapshot déjà présent : {destination}")
    raise SystemExit(0)

destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(destination.name + ".partial")
if temporary.exists():
    shutil.rmtree(temporary)
temporary.mkdir()
subprocess.run(
    ["docker", "cp", f"{args.container}:{container_source}/.", str(temporary)],
    check=True,
)
hashes = file_hashes(temporary)
(temporary / "snapshot_sha256.json").write_text(
    json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.rename(destination)
print(f"Snapshot créé : {destination}")
print(f"Fichiers copiés : {len(hashes)}")
