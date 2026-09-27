"""Load inherited YAML configs, validate overrides, and hash the resolved settings."""
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
    """Load a YAML config and recursively resolve its ``extends`` chain."""
    config = _load_config(path, ancestry=())
    from fog.config_schema import validate_config

    validate_config(config)
    return config


def _load_config(path: str | Path, ancestry: tuple[Path, ...]) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    resolved = path.resolve()
    if resolved in ancestry:
        cycle = " -> ".join(str(item) for item in (*ancestry, resolved))
        raise ValueError(f"config extends cycle: {cycle}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"config must contain a mapping: {path}")

    parent = cfg.pop("extends", None)
    if parent is not None:
        parent_path = Path(parent)
        sibling = path.parent / parent_path
        if not parent_path.is_absolute() and sibling.exists():
            parent_path = sibling
        base = _load_config(parent_path, ancestry=(*ancestry, resolved))
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
        key, separator, value = item.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"override must be a dotted key=value expression: {item!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            child = node.setdefault(p, {})
            if not isinstance(child, dict):
                raise TypeError(
                    f"cannot set {key!r}: {p!r} is already a non-mapping value"
                )
            node = child
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config_with_args(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = apply_overrides(load_config(path), overrides)
    from fog.config_schema import validate_config

    validate_config(config)
    return config
