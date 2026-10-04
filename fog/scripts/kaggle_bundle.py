"""Build a portable Kaggle archive from one validated automated-training job."""

from __future__ import annotations

import io
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


def _bundle_inputs(job_path: Path, root: Path) -> tuple[set[Path], list[str]]:
    job = read_job(job_path)
    validate_job(job, root, check_files=False)
    files: set[Path] = {job_path.resolve()}
    missing: set[str] = set()

    def include(value: str | Path) -> None:
        path = resolve_path(root, str(value).replace("\\", "/")).resolve()
        relative = path.relative_to(root)
        if not path.exists():
            missing.add(relative.as_posix())
        elif path.is_file():
            files.add(path)
        else:
            children = [child.resolve() for child in path.rglob("*") if child.is_file()]
            children = [child for child in children if "__pycache__" not in child.parts]
            if not children:
                missing.add(relative.as_posix())
            files.update(children)

    for relative in (
        "pyproject.toml", "README.md", "uv.lock", "KAGGLE_TRAINING.md",
        "ARCHITECTURE.md",
    ):
        path = root / relative
        if path.is_file():
            files.add(path.resolve())
    for directory in (
        root / "fog",
        root / "configs",
        root / "notebooks",
        root / "docs",
        root / "data" / "teams",
        root / "server" / "pokemon-showdown" / "dist",
        root / "server" / "pokemon-showdown" / "sim",
        root / "server" / "pokemon-showdown" / "data",
        root / "server" / "pokemon-showdown" / "lib",
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
    for key, value in (job.get("overrides", {}) or {}).items():
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    referenced = [job["initial_checkpoint"]]
    # Explicit provenance/data artifacts beyond the manifests needed at training time.
    referenced.extend(job.get("bundle_inputs", []) or [])
    referenced.extend(job.get("initial_league_checkpoints", []) or [])
    referenced.extend([
        "server/pokemon-showdown/package.json",
        "server/pokemon-showdown/package-lock.json",
        "server/pokemon-showdown/pokemon-showdown",
    ])
    referenced.extend(config.get("league", {}).get("past_ckpts", []) or [])
    manifests = {config.get("replay_policy", {}).get("manifest")}
    manifests.update(config.get("bc", {}).get(key) for key in ("manifest", "holdout_manifest"))
    if config.get("env", {}).get("team_pool"):
        referenced.append(config["env"]["team_pool"])
    preflight = config.get("preflight", {}) or {}
    safety = job.get("safety", {}) or {}
    if safety.get("auto_refresh"):
        referenced.append("data/raw_replays")
        raw_games = [path for path in (root / "data/raw_replays").glob("*.json")
                     if not path.name.startswith("_")]
        if len(raw_games) < int(safety.get("games", 5)):
            missing.add("data/raw_replays")
        audit_path = resolve_path(root, safety["replay_audit_config"])
        files.update(config_chain(audit_path, root))
        audit = load_config(audit_path)
        if audit.get("bc", {}).get("init_ckpt"):
            referenced.append(audit["bc"]["init_ckpt"])
    else:
        referenced.extend(
            value for key, value in preflight.items() if key in ("report", "replay_report") and value
        )
    evaluation_pool = job.get("evaluation", {}).get("pool")
    if evaluation_pool:
        referenced.append(evaluation_pool)
    referenced.extend(value for value in manifests if value)
    for value in referenced:
        include(value)

    for replay_manifest in sorted(value for value in manifests if value):
        import pyarrow.parquet as pq

        manifest_path = resolve_path(root, replay_manifest)
        if not manifest_path.is_file():
            continue
        table = pq.read_table(manifest_path, columns=["shard"])
        for value in table.column("shard").to_pylist():
            include(value)
    # Account-local submission settings stay on the submitting machine.
    files.discard((root / "configs/kaggle_results.json").resolve())
    return files, sorted(missing)


def bundle_files(job_path: Path, root: Path) -> set[Path]:
    files, missing = _bundle_inputs(job_path, root)
    if missing:
        raise FileNotFoundError("Kaggle bundle inputs missing:\n" + "\n".join(missing))
    return files


def bundle_for_kaggle(
    job_path: Path, destination: Path, *, dry_run: bool = False, allow_missing_inputs: bool = False,
) -> None:
    root = Path.cwd().resolve()
    files, missing = _bundle_inputs(job_path.resolve(), root)
    if missing:
        print("Missing training inputs:\n" + "\n".join(missing))
        if not allow_missing_inputs:
            raise FileNotFoundError("restore the missing inputs or use --allow-missing-inputs for setup only")
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Kaggle bundle: {len(files):,} files, {total_bytes / (1024 ** 3):.2f} GiB")
    if dry_run:
        return
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "job": job_path.resolve().relative_to(root).as_posix(),
        "files": len(files),
        "uncompressed_bytes": total_bytes,
        "ready_for_training": not missing,
        "missing_inputs": missing,
    }
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            try:
                relative = path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"bundle input is outside the workspace: {path}") from exc
            name = (Path("fog") / relative).as_posix()
            if path.suffix == ".parquet":
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pq.read_table(path)
                if "shard" in table.column_names:
                    shards = [
                        resolve_path(root, value.replace("\\", "/")).resolve().relative_to(root).as_posix()
                        for value in table.column("shard").to_pylist()
                    ]
                    table = table.set_column(table.column_names.index("shard"), "shard", pa.array(shards))
                buffer = io.BytesIO()
                pq.write_table(table, buffer)
                archive.writestr(name, buffer.getvalue())
            else:
                archive.write(path, name)
        archive.writestr("fog/bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(destination)
