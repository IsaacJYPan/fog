"""Build a portable Kaggle archive from one validated automated-training job."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from fog.config import load_config
from fog.scripts.automation_job import read_job, resolve_path, validate_job


def config_chain(path: Path, root: Path) -> set[Path]:
    paths = {path.resolve()}
    raw = read_job(path)
    parent = raw.get("extends")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute() and not parent_path.exists():
            sibling = path.parent / parent_path
            parent_path = sibling if sibling.exists() else root / parent_path
        paths.update(config_chain(parent_path, root))
    return paths


def bundle_files(job_path: Path, root: Path) -> set[Path]:
    job = read_job(job_path)
    validate_job(job, root)
    files: set[Path] = {job_path.resolve()}
    for relative in ("pyproject.toml", "README.md", "uv.lock", "KAGGLE_TRAINING.md"):
        path = root / relative
        if path.is_file():
            files.add(path.resolve())
    for directory in (
        root / "fog",
        root / "configs",
        root / "data" / "teams",
        root / "server" / "pokemon-showdown" / "dist",
    ):
        if not directory.exists():
            raise FileNotFoundError(f"Kaggle bundle input missing: {directory}")
        files.update(
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )

    base_path = resolve_path(root, job["base_config"])
    files.update(config_chain(base_path, root))
    config = load_config(base_path)
    referenced = [job["initial_checkpoint"]]
    referenced.extend(config.get("league", {}).get("past_ckpts", []) or [])
    replay_manifest = config.get("replay_policy", {}).get("manifest")
    if replay_manifest:
        referenced.append(replay_manifest)
    preflight = config.get("preflight", {}) or {}
    referenced.extend(
        value for key, value in preflight.items() if key in ("report", "replay_report") and value
    )
    evaluation_pool = job.get("evaluation", {}).get("pool")
    if evaluation_pool:
        referenced.append(evaluation_pool)
    for value in referenced:
        path = resolve_path(root, value).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Kaggle bundle reference missing: {path}")
        if path.is_file():
            files.add(path)
        else:
            files.update(child.resolve() for child in path.rglob("*") if child.is_file())

    if replay_manifest:
        import pyarrow.parquet as pq

        manifest_path = resolve_path(root, replay_manifest)
        table = pq.read_table(manifest_path, columns=["shard"])
        for value in table.column("shard").to_pylist():
            shard = resolve_path(root, value).resolve()
            if not shard.is_file():
                raise FileNotFoundError(f"replay shard missing: {shard}")
            files.add(shard)
    return files


def bundle_for_kaggle(job_path: Path, destination: Path, *, dry_run: bool = False) -> None:
    root = Path.cwd().resolve()
    files = bundle_files(job_path.resolve(), root)
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Kaggle bundle: {len(files):,} files, {total_bytes / (1024 ** 3):.2f} GiB")
    if dry_run:
        return
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "job": str(job_path),
        "files": len(files),
        "uncompressed_bytes": total_bytes,
    }
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            try:
                relative = path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"bundle input is outside the workspace: {path}") from exc
            archive.write(path, Path("fog") / relative)
        archive.writestr("fog/bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(destination)
