"""Recover the prepared private inputs without publishing an account or dataset name.

Run with the official Kaggle client installed and the owner's account configured.
Find the archive by filename and size, then verify the committed SHA-256 after download.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    expected = json.loads(Path(__file__).with_name("first-package.json").read_text())
    api = KaggleApi()
    api.authenticate()
    target = args.out / expected["filename"]
    seen = set()
    for page in range(1, 101):
        datasets = api.dataset_list(mine=True, page=page)
        if not datasets:
            break
        for dataset in datasets:
            if dataset.ref in seen:
                continue
            seen.add(dataset.ref)
            # Dataset totals include service metadata; the file size is authoritative.
            if dataset.total_bytes and dataset.total_bytes < expected["bytes"]:
                continue
            files = api.dataset_list_files(dataset.ref).dataset_files
            if not any(file.name == expected["filename"] and file.total_bytes == expected["bytes"]
                       for file in files):
                continue
            metadata_file = api.dataset_metadata(dataset.ref, str(args.out / "private-metadata"))
            metadata = json.loads(Path(metadata_file).read_text())["info"]
            assert metadata["isPrivate"] is True
            # The client prints a private dataset URL even with quiet=True.
            with redirect_stdout(io.StringIO()):
                api.dataset_download_file(dataset.ref, expected["filename"], path=str(args.out),
                                          force=True, quiet=True)
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected["sha256"]:
                target.unlink()
                continue
            (args.out / "private-input-ref.json").write_text(
                json.dumps({"ref": dataset.ref}, indent=2) + "\n")
            print(json.dumps({"private": True, "bytes": target.stat().st_size,
                              "sha256": digest, "download_verified": True}))
            return
    raise SystemExit("matching private input archive not found; rebuild with the recorded CPU commands")


if __name__ == "__main__":
    main()
