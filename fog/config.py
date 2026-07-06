"""Config loading + hashing.

Every run is reproducible from (resolved config + seed). Configs support single-level `extends` to a
base yaml; the resolved dict is hashed so runs are traceable to an exact configuration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a yaml config, resolving a single `extends:` chain relative to configs/."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent is not None:
        base = load_config(parent)
        cfg = _deep_merge(base, cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of a resolved config, used to tag runs and checkpoints."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _coerce(v: str) -> Any:
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply CLI `key.sub=value` overrides (dotted paths) onto a resolved config in place."""
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config_with_args(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    return apply_overrides(load_config(path), overrides)
