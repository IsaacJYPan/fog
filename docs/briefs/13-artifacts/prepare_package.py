import hashlib
import json
import tarfile
import time
from pathlib import Path

work = Path("runs/brief13").resolve()
started = time.monotonic()
state = {"work_root": str(work), "work_dir": str(work), "steps": {}}
(work / "run.json").write_text(json.dumps(state, indent=2) + "\n")
package = work / "upload"
package.mkdir(exist_ok=True)
target = package / "replay-work.bin"
with tarfile.open(target.with_suffix(".partial"), "w:gz", compresslevel=1) as archive:
    for relative in (
        "run.json",
        "decisions/manifest.parquet",
        "decisions/quality_report.json",
        "decisions/shards",
    ):
        archive.add(work / relative, arcname=relative)
    for file in (work / "split").glob("*.parquet"):
        archive.add(file, arcname="split/" + file.name)
    archive.add(work / "split/shards", arcname="split/shards")
target.with_suffix(".partial").replace(target)
with target.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
result = {
    "filename": target.name,
    "bytes": target.stat().st_size,
    "sha256": digest,
    "wall_seconds": time.monotonic() - started,
    "source_commit": "7d17e1a",
    "contents": "fresh corrected decisions and filtered split, no raw logs or checkpoints",
    "command": "python runs/brief13/prepare_package.py",
}
Path("docs/briefs/13-artifacts/first-package.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
