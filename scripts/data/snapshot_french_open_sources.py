import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path.home() / "ivoireslm-storage"
SNAPSHOT = ROOT / "snapshots/french_open_2026-08-13_v0.1"
PARTIAL = SNAPSHOT.with_name(SNAPSHOT.name + ".partial")
WIKTIONARY_URL = (
    "https://kaikki.org/frwiktionary/Fran%C3%A7ais/"
    "kaikki.org-dictionary-Fran%C3%A7ais.jsonl.gz"
)
PYTHON_DOCS_URL = "https://github.com/python/python-docs-fr.git"
USER_AGENT = "IvoireSLM corpus builder/0.4 (https://github.com/gakale/ivoireslm)"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if SNAPSHOT.exists():
    manifest = SNAPSHOT / "snapshot_manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"Snapshot existant sans manifeste : {SNAPSHOT}")
    print(f"Snapshot déjà présent : {SNAPSHOT}")
    raise SystemExit(0)

if PARTIAL.exists():
    shutil.rmtree(PARTIAL)
PARTIAL.mkdir(parents=True)

dictionary = PARTIAL / "frwiktionary_kaikki_2026-08-11.jsonl.gz"
request = urllib.request.Request(WIKTIONARY_URL, headers={"User-Agent": USER_AGENT})
with urllib.request.urlopen(request, timeout=120) as response, dictionary.open("wb") as output:
    shutil.copyfileobj(response, output, length=1024 * 1024)
    dictionary_headers = {
        "content_length": response.headers.get("Content-Length"),
        "last_modified": response.headers.get("Last-Modified"),
        "etag": response.headers.get("ETag"),
    }

python_docs = PARTIAL / "python-docs-fr-3.14"
subprocess.run(
    [
        "git", "clone", "--depth", "1", "--single-branch", "--branch", "3.14",
        PYTHON_DOCS_URL, str(python_docs),
    ],
    check=True,
)
commit = subprocess.run(
    ["git", "-C", str(python_docs), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

manifest = {
    "snapshot_id": SNAPSHOT.name,
    "created_at": "2026-08-13",
    "sources": [
        {
            "source_id": "frwiktionary_kaikki_french",
            "url": WIKTIONARY_URL,
            "upstream_dump_date": "2026-08-04",
            "extraction_date": "2026-08-11",
            "local_file": dictionary.name,
            "bytes": dictionary.stat().st_size,
            "sha256": sha256_file(dictionary),
            "http_headers": dictionary_headers,
            "license": "CC BY-SA 4.0 and GFDL; reuser may comply with either, subject to per-page exceptions",
            "attribution": "French Wiktionary contributors; extraction by Wiktextract/kaikki.org",
        },
        {
            "source_id": "python_docs_fr_3_14",
            "url": PYTHON_DOCS_URL,
            "branch": "3.14",
            "commit": commit,
            "local_directory": python_docs.name,
            "translation_contribution_license": "CC0 1.0",
            "documentation_license": "Python Software Foundation License Version 2",
            "attribution": "Python Software Foundation and python-docs-fr contributors",
        },
    ],
}
(PARTIAL / "snapshot_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PARTIAL.rename(SNAPSHOT)
print(f"Snapshot créé : {SNAPSHOT}")
print(f"Wiktionnaire : {(SNAPSHOT / dictionary.name).stat().st_size:,} octets")
print(f"Documentation Python commit : {commit}")
