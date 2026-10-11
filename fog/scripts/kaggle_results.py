"""Fetch a pinned Kaggle run and prepare a local, inspectable training handoff."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/kaggle_results.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Several independent packages update the same latest pointer. Their staging
    # files must be distinct even though the last completed write wins.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix="." + path.name + "-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2) + "\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        for attempt in range(50):
            try:
                temporary.replace(path)
                break
            except PermissionError as error:
                # Windows can briefly deny replacement while another rename or
                # reader holds the target. Keep the complete prior JSON visible.
                if os.name != "nt" or error.winerror not in (5, 32, 33) or attempt == 49:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


def validate_members(archive: zipfile.ZipFile) -> None:
    """Validate the entire archive before writing any member, including on Windows."""
    seen = set()
    total = 0
    for entry in archive.infolist():
        name = entry.orig_filename
        parts = name.rstrip("/").split("/")
        if (not parts or name.startswith(("/", "\\")) or "\\" in name
                or any(p in {".", ".."} or ":" in p or p.endswith((".", " "))
                       or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", p)
                       for p in parts)
                or stat.S_ISLNK(entry.external_attr >> 16)):
            raise ValueError(f"Unsafe archive member: {name!r}")
        canonical = name.rstrip("/").casefold()
        if canonical in seen:
            raise ValueError(f"Duplicate archive member: {name!r}")
        seen.add(canonical)
        total += entry.file_size
    if total > 20 * 1024**3:
        raise ValueError("Uncompressed result exceeds the 20 GiB import limit")
    if "pipeline_state.json" not in archive.namelist():
        raise ValueError("Expected a FOG export with pipeline_state.json at its root")


def summarize(directory: Path, source: dict, display: Path | None = None) -> dict:
    display = display or directory
    def location(path: Path) -> str:
        return str((display / path.relative_to(directory)).resolve())
    state = json.loads((directory / "pipeline_state.json").read_text(encoding="utf-8"))
    checkpoints = []
    for path in sorted(directory.rglob("ppo_final.pt")):
        checkpoints.append({
            "path": location(path), "sha256": digest(path),
            "optimizer_state": location(path.with_suffix(".state.pt"))
            if path.with_suffix(".state.pt").is_file() else None,
        })
    metrics = []
    for path in sorted(directory.rglob("metrics.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        metrics.append({"path": location(path), "updates": len(rows),
                        "last": rows[-1] if rows else {}})
    audits = []
    for path in sorted((directory / "audits").rglob("summary.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        audits.append({"path": location(path), "passed": report.get("passed"),
                       "error_count": report.get("error_count"),
                       "games_completed": report.get("games_completed")})
    blocks = [{k: b.get(k) for k in ("index", "status", "train_wall_seconds", "error")}
              for b in state.get("blocks", [])]
    return {"source": source, "directory": str(display.resolve()),
            "pipeline_status": state.get("status"), "blocks": blocks,
            "checkpoints": checkpoints, "metrics": metrics, "audits": audits,
            "note": "Original configs retain Kaggle paths. Create a new local job before "
                    "continuing; do not overwrite the champion or reuse stale audit gates."}


def import_archive(archive_path: Path, output: Path, kernel: str, version: int) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel) or version < 1:
        raise ValueError("Expected owner/notebook and a positive version number")
    checksum = digest(archive_path)
    destination = output / kernel.replace("/", "--") / f"v{version}-{checksum[:12]}"
    source = {"kernel": kernel, "version": version, "archive_sha256": checksum,
              "archive": str(archive_path.resolve()),
              "notebook_url": f"https://www.kaggle.com/code/{kernel}"}
    if destination.exists():
        existing = json.loads((destination / "RESULTS.json").read_text(encoding="utf-8"))
        if existing["source"]["archive_sha256"] != checksum:
            raise ValueError("Existing result directory has different contents")
        print(f"Already imported: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory cleans only this newly created staging directory on failure.
    with tempfile.TemporaryDirectory(prefix=".import-", dir=destination.parent) as staging:
        stage = Path(staging) / "result"
        stage.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            validate_members(archive)
            archive.extractall(stage)
        report = summarize(stage, source, destination)
        write_json(stage / "RESULTS.json", report)
        write_handoff(stage, report, kernel, version)
        stage.rename(destination)
    write_json(output / "latest-import.json", {"directory": str(destination.resolve()),
                                               "report": str((destination / "RESULTS.json").resolve())})
    print(f"Imported: {destination}\nHandoff: {destination / 'HANDOFF.md'}")
    return destination


def write_handoff(directory: Path, report: dict, kernel: str, version: int) -> None:
    lines = [f"# Kaggle result: {kernel}, version {version}", "",
             f"Pipeline status: {report['pipeline_status']}", "", "## Checkpoints", ""]
    lines += [f"- `{c['path']}`" for c in report["checkpoints"]] or ["No final checkpoint."]
    lines += ["", "## Metrics", "", "```json", json.dumps(report["metrics"], indent=2),
              "```", "", report["note"], "", "See RESULTS.json for audit and source details."]
    (directory / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_output(url: str, path: Path, *, attempts: int = 3) -> None:
    """Resume interrupted transfers and publish only a complete response body.

    Output hashes and archive contents are verified by the collector afterward.
    Never include signed URLs in transfer errors or progress messages.
    """
    import requests

    if attempts < 1:
        raise ValueError("Download attempts must be positive")
    partial = path.with_name(path.name + ".part")
    offset, expected_size, etag = 0, None, None
    last_progress = time.monotonic()
    for attempt in range(attempts):
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if etag:
                headers["If-Range"] = etag
        try:
            with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(
                        f"Kaggle file download returned HTTP {response.status_code}"
                    )
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise RuntimeError("Kaggle output cannot resume a compressed HTTP body")
                if response.status_code == 200:
                    # A server may ignore Range or reject If-Range. Start a full
                    # replacement rather than appending duplicate/mixed bytes.
                    offset = 0
                    length = response.headers.get("Content-Length")
                    expected_size = int(length) if length is not None else None
                    etag = response.headers.get("ETag")
                    mode = "wb"
                else:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if not match:
                        raise RuntimeError("Kaggle output resume lacks a valid byte range")
                    start, end, total = map(int, match.groups())
                    if (start != offset or not start <= end < total
                            or (expected_size is not None and total != expected_size)):
                        raise RuntimeError("Kaggle output resume returned a different byte range")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != end - start + 1:
                        raise RuntimeError("Kaggle output resume length disagrees with its range")
                    current_etag = response.headers.get("ETag")
                    if etag and current_etag and etag != current_etag:
                        raise RuntimeError("Kaggle output changed during a resumed transfer")
                    expected_size, etag = total, current_etag or etag
                    mode = "ab" if offset else "wb"
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if expected_size is not None and offset > expected_size:
                            raise RuntimeError("Kaggle output exceeded its declared size")
                        if time.monotonic() - last_progress >= 30:
                            print(f"Output download: {offset:,} bytes received", flush=True)
                            last_progress = time.monotonic()
            if expected_size is None or offset == expected_size:
                partial.replace(path)
                return
        except requests.RequestException:
            # IncompleteRead can occur after several GB have already arrived.
            # Keep the bytes from this call and request only the missing suffix.
            offset = partial.stat().st_size if partial.exists() else 0
            if expected_size is not None and offset == expected_size:
                partial.replace(path)
                return
        if attempt + 1 < attempts:
            print(f"Output transfer retry {attempt + 2}/{attempts}: {offset:,} bytes retained",
                  flush=True)
    raise RuntimeError(
        f"Kaggle output transfer incomplete after {attempts} attempts; {offset:,} bytes retained"
    ) from None


def remote(kernel: str, version: int, destination: Path | None, filename: str,
           run_id: int | None) -> None:
    # Kaggle serves latest outputs; version_label returns 404 for these private
    # runs. Fail closed unless the signed file URL identifies the requested run.
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest,
    )
    api = KaggleApi()
    api.authenticate()
    owner, slug = kernel.split("/")
    with api.build_kaggle_client() as connection:
        service = connection.kernels.kernels_api_client
        if destination is None:
            request = ApiGetKernelSessionStatusRequest()
            request.user_name, request.kernel_slug = owner, slug
            print("Latest notebook session status (not a historical version lookup):")
            print(service.get_kernel_session_status(request))
            return
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        request.page_size = 200
        tokens = set()
        while True:
            response = service.list_kernel_session_output(request)
            for item in response.files or []:
                if item.file_name != filename:
                    continue
                url = urlsplit(item.url)
                match = re.match(r"^/kf/(\d+)/", url.path)
                if (url.scheme != "https" or url.hostname != "www.kaggleusercontent.com"
                        or not match or (run_id is not None and int(match[1]) != run_id)):
                    raise RuntimeError("Kaggle returned a different or unrecognized run. "
                                       "No file downloaded; use the saved version's Output tab.")
                destination.mkdir(parents=True, exist_ok=True)
                download_output(item.url, destination / filename)
                print(f"Downloaded {filename} from {kernel}, version {version}")
                write_json(destination / "download.json", {"run_id": int(match[1]), "kernel": kernel})
                return
            token = response.next_page_token
            if not token:
                raise FileNotFoundError("This version has no FOG export archive")
            if token in tokens:
                raise RuntimeError("Kaggle repeated an output page token")
            tokens.add(token)
            request.page_token = token


def client() -> Path:
    executable = ROOT / ".tools/kaggle" / ("Scripts/kaggle.exe" if os.name == "nt" else "bin/kaggle")
    if not executable.is_file():
        raise RuntimeError("Kaggle client missing. Run: fog-kaggle.cmd setup")
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["setup", "login", "status", "pull", "import", "_fetch", "_status"])
    parser.add_argument("--version", type=int)
    parser.add_argument("--kernel")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    kernel = args.kernel or config["kernel"]
    version = config["version"] if args.version is None else args.version
    run_id = args.run_id or config["run_id"]
    if (kernel != config["kernel"] or version != config["version"]) and not args.run_id:
        parser.error("For a different version supply --run-id from its scriptVersionId URL")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel) or version < 1:
        parser.error("Expected owner/notebook and a positive version")
    output = ROOT / "runs/kaggle"
    if args.command in {"_fetch", "_status"}:
        remote(kernel, version, args.archive if args.command == "_fetch" else None,
               config["archive_name"], run_id)
    elif args.command == "setup":
        env = ROOT / ".tools/kaggle"
        if not env.exists():
            subprocess.run(["uv", "venv", str(env), "--python", "3.12"], check=True)
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(["uv", "pip", "install", "--python", str(python), "kaggle==2.2.4"], check=True)
    elif args.command == "login":
        subprocess.run([str(client()), "auth", "login"], check=True)
    elif args.command == "status":
        subprocess.run([str(client().with_name("python.exe" if os.name == "nt" else "python")),
                        "-m", "fog.scripts.kaggle_results", "_status", "--kernel", kernel,
                        "--version", str(version), "--run-id", str(run_id)], check=True, cwd=ROOT)
    elif args.command == "import":
        if not args.archive:
            parser.error("import requires --archive")
        import_archive(args.archive, output, kernel, version)
    else:
        downloads = output / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".download-", dir=downloads) as temporary:
            subprocess.run([str(client().with_name("python.exe" if os.name == "nt" else "python")),
                            "-m", "fog.scripts.kaggle_results", "_fetch", "--kernel", kernel,
                            "--version", str(version), "--run-id", str(run_id),
                            "--archive", temporary], check=True, cwd=ROOT)
            archive = Path(temporary) / config["archive_name"]
            if not archive.is_file():
                raise FileNotFoundError("This version has no FOG output archive; check its run logs")
            saved = downloads / f"{kernel.replace('/', '--')}-v{version}-{digest(archive)[:12]}.zip"
            if not saved.exists():
                shutil.copy2(archive, saved)
        import_archive(saved, output, kernel, version)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print("Kaggle command did not complete. If authentication is required, run "
              "fog-kaggle.cmd login once, then retry.", file=sys.stderr)
        raise SystemExit(exc.returncode) from None
