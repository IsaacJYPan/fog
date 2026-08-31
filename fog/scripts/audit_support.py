"""Shared immutable-report helpers for direct and replay safety audits."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def parameter_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in module.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_digest_with_metadata(module: torch.nn.Module) -> str:
    """Digest tensor keys, dtype, shape, and bytes for cross-audit scope verification."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def git_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if revision:
            return revision
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        head = Path(".git/HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            return Path(".git", head[5:]).read_text(encoding="utf-8").strip()
        return head
    except OSError:
        return "unknown"


class AuditRecorder:
    """Append-only decision and issue recorder for non-mutating safety audits."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.issues: list[dict[str, Any]] = []

    def issue(self, record: dict, code: str, detail: str, *, severity: str = "error") -> None:
        issue = {
            "game": record.get("game"),
            "side": record.get("side"),
            "turn": record.get("turn"),
            "phase": record.get("phase"),
            "code": code,
            "severity": severity,
            "detail": detail,
        }
        record.setdefault("issues", []).append(issue)
        self.issues.append(issue)

    def check(
        self,
        record: dict,
        condition: bool,
        code: str,
        detail: str,
        *,
        severity: str = "error",
    ) -> None:
        if not bool(condition):
            self.issue(record, code, detail, severity=severity)

    def append(self, record: dict) -> None:
        self.records.append(record)

    def write_trace(self) -> None:
        with (self.out_dir / "decisions.jsonl").open("w", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(json.dumps(record, default=json_value, separators=(",", ":")) + "\n")
