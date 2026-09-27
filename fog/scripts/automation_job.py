"""Portable job-file schema and path handling for automated training."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1


def read_job(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"job file must contain a mapping: {path}")
    return data


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def portable_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def detect_node() -> str | None:
    configured = os.environ.get("FOG_NODE_BINARY")
    if configured and Path(configured).is_file():
        return configured
    return shutil.which("node")


def validate_job(job: dict[str, Any], root: Path, *, check_files: bool = True) -> None:
    if int(job.get("version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"job.version must be {SCHEMA_VERSION}")
    required = ("name", "output_dir", "base_config", "initial_checkpoint")
    missing = [key for key in required if not job.get(key)]
    if missing:
        raise ValueError(f"job is missing: {', '.join(missing)}")
    for key in ("base_config", "initial_checkpoint"):
        path = resolve_path(root, job[key])
        if check_files and not path.is_file():
            raise FileNotFoundError(f"{key} not found: {path}")
    training = job.get("training", {})
    if int(training.get("blocks", 0)) <= 0:
        raise ValueError("training.blocks must be positive")
    if int(training.get("decisions_per_block", 0)) <= 0:
        raise ValueError("training.decisions_per_block must be positive")
    evaluation = job.get("evaluation", {})
    if evaluation.get("enabled", True):
        pool = resolve_path(root, evaluation.get("pool", ""))
        if check_files and not pool.exists():
            raise FileNotFoundError(f"evaluation pool not found: {pool}")
        games = int(evaluation.get("games", 64))
        if games <= 0 or games % 4:
            raise ValueError("evaluation.games must be a positive multiple of four")
        champion_games = int(evaluation.get("champion_games", 1024))
        if champion_games < 128 or champion_games % 4:
            raise ValueError("evaluation.champion_games must be at least 128 and a multiple of four")
        if not 0 < float(evaluation.get("promotion_family_alpha", 0.05)) < 1:
            raise ValueError("evaluation.promotion_family_alpha must be in (0,1)")
