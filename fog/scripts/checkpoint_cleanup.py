"""Conservatively prune unreferenced numbered model snapshots.

Final/best/latest models and explicitly referenced checkpoints are always protected.  By default
the newest two numbered snapshots in every checkpoint directory are also protected.  The command
is a dry run unless --execute is supplied.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NUMBERED = re.compile(r"^(ppo|bc_epoch|student_epoch)_(\d+)$")
PATH_REFERENCE = re.compile(r"runs[\\/][^\s\"']+?\.pt")
REFERENCE_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".cmd", ".ps1"}


def _normal(path: Path) -> str:
    return str(path.resolve()).lower()


def _referenced_checkpoints(root: Path) -> set[str]:
    references: set[str] = set()
    search_roots = [root / "configs", root / "runs"]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if root / "runs" / "checkpoint_cleanup" in path.parents:
                continue
            if not path.is_file() or path.suffix.lower() not in REFERENCE_SUFFIXES:
                continue
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in PATH_REFERENCE.findall(text):
                references.add(_normal(root / Path(match.replace("\\", "/"))))
    return references


def build_plan(root: Path, *, keep_recent: int = 2, min_age_hours: float = 1.0) -> dict[str, Any]:
    runs = (root / "runs").resolve()
    references = _referenced_checkpoints(root)
    groups: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
    protected: dict[str, str] = {}
    now = datetime.now(UTC).timestamp()

    for path in runs.rglob("*.pt"):
        match = NUMBERED.match(path.stem)
        if match is None:
            protected[str(path)] = "named final/best/special model"
            continue
        groups[path.parent].append((int(match.group(2)), path))

    candidates: list[dict[str, Any]] = []
    for entries in groups.values():
        entries.sort(key=lambda item: item[0])
        recent = {path for _, path in entries[-keep_recent:]} if keep_recent else set()
        for _, path in entries:
            reason = None
            if _normal(path) in references:
                reason = "referenced by config/report/pointer"
            elif path in recent:
                reason = f"newest {keep_recent} numbered snapshot(s) in run"
            # A non-positive window explicitly disables age protection.  This
            # also avoids treating a freshly-created file whose filesystem
            # timestamp is a few clock ticks ahead of ``now`` as protected
            # when callers request a zero-hour window.
            elif min_age_hours > 0 and (now - path.stat().st_mtime) / 3600 < min_age_hours:
                reason = f"newer than {min_age_hours:g} hour safety window"
            if reason:
                protected[str(path)] = reason
            else:
                candidates.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "modified_at": datetime.fromtimestamp(
                            path.stat().st_mtime, UTC
                        ).isoformat(),
                    }
                )

    return {
        "status": "planned",
        "generated_at": datetime.now(UTC).isoformat(),
        "workspace": str(root),
        "policy": {
            "numbered_patterns": ["ppo_N.pt", "bc_epoch_N.pt", "student_epoch_N.pt"],
            "keep_recent_per_checkpoint_directory": keep_recent,
            "minimum_age_hours": min_age_hours,
            "protect_all_named_models": True,
            "protect_references": True,
        },
        "candidate_count": len(candidates),
        "candidate_bytes": sum(item["bytes"] for item in candidates),
        "candidates": candidates,
        "protected_count": len(protected),
        "protected": protected,
    }


def execute_plan(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    runs = (root / "runs").resolve()
    removed = []
    for item in plan["candidates"]:
        path = Path(item["path"]).resolve()
        if runs not in path.parents:
            raise ValueError(f"refusing to delete outside runs: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        path.unlink()
        removed.append({"path": str(path), "bytes": size})
    plan.update(
        status="executed",
        executed_at=datetime.now(UTC).isoformat(),
        removed_count=len(removed),
        removed_bytes=sum(item["bytes"] for item in removed),
        removed=removed,
        recovery="irreversible local deletion; retained models and reports remain",
    )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--keep-recent", type=int, default=2)
    parser.add_argument("--min-age-hours", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("runs/checkpoint_cleanup/latest.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.keep_recent < 0:
        parser.error("--keep-recent must be non-negative")
    root = args.root.resolve()
    plan = build_plan(
        root, keep_recent=args.keep_recent, min_age_hours=args.min_age_hours
    )
    if args.execute:
        plan = execute_plan(plan, root)
    output = args.out if args.out.is_absolute() else root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    gib = plan["candidate_bytes"] / (1024**3)
    print(
        f"{plan['status']}: {plan['candidate_count']} candidate snapshots, "
        f"{gib:.2f} GiB; report={output}"
    )


if __name__ == "__main__":
    main()
