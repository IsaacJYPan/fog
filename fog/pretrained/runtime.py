"""Explicit native import paths for frozen runtimes without affecting the collector."""

from __future__ import annotations

import os
from pathlib import Path


def interpreter_path(path: Path) -> str:
    """Make executable paths absolute without following a venv's Python symlink."""
    return str(path.expanduser().absolute())


def native_environment(root: Path):
    environment = os.environ.copy()
    # Setup probes import Metamon directly, before native_worker's own defaults.
    environment.setdefault("METAMON_CACHE_DIR", str((root / "runs/pretrained/cache").resolve()))
    source = root / "upstream"
    if source.exists():
        paths = [source / "amago", source / "metamon", source / "poke-env/src"]
        if not all(path.is_dir() for path in paths):
            raise ValueError("Frozen native source directories are incomplete")
        # Some Kaggle images expose editable distribution metadata without
        # executing their .pth import hooks. Sources are already hash verified.
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(root), *map(str, paths)] + ([existing] if existing else [])
        )
    return environment
