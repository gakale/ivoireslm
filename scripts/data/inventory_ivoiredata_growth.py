import json
import subprocess
from pathlib import Path


OUTPUT = Path.home() / "ivoireslm-storage/reports/ivoiredata_growth_inventory_v0.2.jsonl"
code = r'''
import collections,json,pathlib
import pyarrow.parquet as pq
root=pathlib.Path("/app/data_lake/domains")
for source in sorted(path for path in root.glob("*/*") if path.is_dir()):
    manifest=source/"manifest.json"
    metadata=json.loads(manifest.read_text()) if manifest.exists() else {}
    files=[path for path in source.rglob("*") if path.is_file()]
    parquet=[]
    for path in source.rglob("*.parquet"):
        if "datagouv_catalog" in str(path):
            continue
        try:
            item=pq.ParquetFile(path)
            parquet.append({"path":str(path.relative_to(source)),"rows":item.metadata.num_rows,"columns":item.schema.names})
        except Exception as error:
            parquet.append({"path":str(path.relative_to(source)),"error":str(error)})
    print(json.dumps({
        "source_id":source.name,"domain":source.parent.name,
        "rights_tier":metadata.get("rights_tier"),"source_url":metadata.get("source_url"),
        "bytes":sum(path.stat().st_size for path in files),"files":len(files),
        "extensions":dict(collections.Counter(path.suffix.lower() or "<none>" for path in files)),
        "business_parquet":parquet,
    },ensure_ascii=False))
'''
result = subprocess.run(
    ["docker", "exec", "ivoiredata-api-1", "python3", "-c", code],
    check=True,
    capture_output=True,
    text=True,
)
rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"Sources inventoriées : {len(rows)}")
print(f"Rapport : {OUTPUT}")
