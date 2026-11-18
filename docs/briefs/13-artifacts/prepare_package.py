"""Package corrected replay shards and splits for a private training job."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("runs/brief13"))
    parser.add_argument("--report", type=Path,
                        default=Path("docs/briefs/13-artifacts/first-package.json"))
    args = parser.parse_args()
    work = args.work.resolve()
    started = time.monotonic()
    state = {"work_root": str(work), "work_dir": str(work), "steps": {}}
    (work / "run.json").write_text(json.dumps(state, indent=2) + "\n")
    package = work / "upload"
    package.mkdir(exist_ok=True)
    target = package / "replay-work.bin"
    temporary = target.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for relative in ("run.json", "decisions/manifest.parquet", "decisions/quality_report.json",
                         "decisions/shards", "split/shards"):
            archive.add(work / relative, arcname=relative)
        for file in sorted((work / "split").glob("*.parquet")):
            archive.add(file, arcname="split/" + file.name)
    temporary.replace(target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "filename": target.name,
        "bytes": target.stat().st_size,
        "sha256": digest,
        "wall_seconds": time.monotonic() - started,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "contents": "corrected decisions and filtered split, no raw logs or checkpoints",
        "command": ["python", "docs/briefs/13-artifacts/prepare_package.py",
                    "--work", str(args.work), "--report", str(args.report)],
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
