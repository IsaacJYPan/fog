"""Read fresh quota and all owned job statuses, without exporting account identifiers.

Run with the official client's Python. A missing session is acceptable only for a
version-zero draft. Pull accelerator metadata for every active job: CPU jobs may
continue, while any active GPU job or unknown accelerator blocks submission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private-state", type=Path, required=True)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.private_state.parent.mkdir(parents=True, exist_ok=True)
    checked = {}

    def inventory():
        result = {}
        for page in range(1, 100):
            jobs = api.kernels_list(mine=True, page_size=100, page=page, sort_by="dateCreated")
            if not jobs:
                return result
            result.update({job.ref: job for job in jobs})
            if len(jobs) < 100:
                return result
        raise RuntimeError("job listing exceeded pagination bound")

    for _ in range(3):
        jobs = inventory()
        for ref, job in jobs.items():
            version = (job.current_version_number, str(job.last_run_time))
            if ref in checked and checked[ref]["version"] == version:
                continue
            for attempt in range(5):
                try:
                    status = str(api.kernels_status(ref).status).split(".")[-1]
                    break
                except Exception as error:
                    if "404" in str(error) and job.current_version_number == 0:
                        status = "VERSION_ZERO_NO_SESSION"
                        break
                    if "429" not in str(error) or attempt == 4:
                        raise
                    time.sleep(30)
            checked[ref] = {"version": version, "status": status}
            args.private_state.write_text(json.dumps(checked, indent=2) + "\n")
            time.sleep(2)
        fresh = inventory()
        if all(ref in checked and checked[ref]["version"] == (
            job.current_version_number, str(job.last_run_time)) for ref, job in fresh.items()
        ):
            break
    else:
        raise RuntimeError("job inventory kept changing; no stable account check")
    counts = Counter(checked[ref]["status"] for ref in fresh)
    terminal = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "VERSION_ZERO_NO_SESSION"}
    active = sum(n for status, n in counts.items() if status not in terminal)
    active_jobs = []
    for ref in fresh:
        status = checked[ref]["status"]
        if status in terminal:
            continue
        directory = args.private_state.parent / "job-metadata" / str(len(active_jobs))
        directory.mkdir(parents=True, exist_ok=True)
        api.kernels_pull(ref, str(directory), metadata=True, quiet=True)
        metadata = json.loads((directory / "kernel-metadata.json").read_text())
        active_jobs.append({"job_slug": ref.split("/")[-1], "status": status,
                            "enable_gpu": metadata.get("enable_gpu")})
    blocked = sum(job["enable_gpu"] is not False for job in active_jobs)
    quota = json.loads(subprocess.check_output(
        [sys.executable, "-m", "kaggle", "quota", "--format", "json"], text=True))
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": "kaggle quota --format json; official client list/status of every owned job",
              "owned_jobs": len(fresh), "status_counts": dict(counts),
              "active_or_unknown_jobs": active, "active_jobs": active_jobs,
              "blocking_gpu_or_unknown_jobs": blocked, "quota": quota}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if blocked:
        raise SystemExit("account has an active GPU/unknown job; do not submit")


if __name__ == "__main__":
    main()
