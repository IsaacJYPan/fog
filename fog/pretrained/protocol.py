"""Persistent limits for GPU allocations and the two final confirmation cohorts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import json
import hashlib
from pathlib import Path

PHASE_SECONDS = {"pilot": 7200, "main": 28800, "refinement": 28800}


def source_fingerprint(root: Path):
    files = {}
    for folder in ("fog", "integrations"):
        for path in sorted((root / folder).rglob("*")):
            if (
                path.is_file()
                and path.suffix in (".py", ".js", ".json")
                and "__pycache__" not in path.parts
            ):
                with path.open("rb") as stream:
                    files[path.relative_to(root).as_posix()] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def foulplay_fingerprint(source: Path):
    """Bind the opponent's code and cached inference priors, including untracked caches."""
    if not (source / "fp/search/standard_battles.py").is_file():
        raise ValueError("Require the Foul Play source used by the opponent")
    files = {}
    for path in sorted((source / "fp").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".json"):
            with path.open("rb") as stream:
                files[path.relative_to(source).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


@contextmanager
def registry(path: Path):
    """Fail closed on a concurrent or interrupted registry mutation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            "Registry is locked; inspect the existing operation before retrying"
        ) from error
    try:
        with handle:
            handle.write(datetime.now(UTC).isoformat())
        value = json.loads(path.read_text()) if path.exists() else {}
        yield value
        temporary = path.with_suffix(".pending")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        lock.unlink()


def reserve_gpu(path: Path, *, run_id: str, phase: str, condition: str, seconds: int, family: str):
    if phase not in PHASE_SECONDS or condition not in ("ordinary", "joint"):
        raise ValueError("Require a declared matched GPU phase and condition")
    if not 0 < seconds <= PHASE_SECONDS[phase] // 2:
        raise ValueError("Per-arm GPU allocation exceeds its phase limit")
    row = {"phase": phase, "condition": condition, "seconds": seconds, "family": family}
    with registry(path) as ledger:
        runs = ledger.setdefault("reservations", {})
        if run_id in runs:
            if runs[run_id] != row:
                raise ValueError("GPU reservation cannot change")
            return
        if any(
            old["phase"] == phase and old["condition"] == condition and old["family"] == family
            for old in runs.values()
        ):
            raise ValueError(
                "This matched arm already has a reservation; do not launch a duplicate"
            )
        if (
            sum(old["seconds"] for old in runs.values() if old["phase"] == phase) + seconds
            > PHASE_SECONDS[phase]
        ):
            raise ValueError("Aggregate GPU phase budget exceeded")
        if sum(old["seconds"] for old in runs.values()) + seconds > 18 * 3600:
            raise ValueError("Aggregate pretrained attempt exceeds 18 GPU hours")
        runs[run_id] = row
        ledger["accounting"] = (
            "Full requested wall time is charged, including failed or interrupted runs."
        )
        ledger["limit_seconds"] = 18 * 3600


def claim_confirmation(
    path: Path,
    *,
    attempt: int,
    candidate_sha256: str,
    pool_sha256: str,
    seed: int,
    games: int,
    sources_sha256="",
):
    if attempt not in (1, 2) or games < 1024 or games % 4:
        raise ValueError(
            "Only two predeclared confirmation attempts of at least 1024 balanced games"
        )
    row = {
        "candidate_sha256": candidate_sha256,
        "pool_sha256": pool_sha256,
        "seed": seed,
        "games": games,
        "sources_sha256": sources_sha256,
    }
    with registry(path) as ledger:
        attempts = ledger.setdefault("attempts", {})
        key = str(attempt)
        if any(old["pool_sha256"] != pool_sha256 for old in attempts.values()):
            raise ValueError("Final team pool cannot change between confirmation attempts")
        if key in attempts and attempts[key] != row:
            raise ValueError("Final cohort already spent on a different candidate or protocol")
        attempts[key] = row
        ledger["planned_attempts"] = 2
        ledger["alpha_per_attempt"] = 0.025


def claim_strong_evaluation(
    path: Path,
    *,
    attempt: int,
    finalist_sha256: str,
    evaluated_sha256: str,
    original_sha256: str,
    pool_sha256: str,
    seed: int,
    games: int,
    sources_sha256="",
    opponent_sha256="",
):
    if games < 256 or games % 4:
        raise ValueError("Strong-opponent evaluation requires at least 256 balanced games per arm")
    with registry(path) as ledger:
        primary = ledger.get("attempts", {}).get(str(attempt))
        if primary is None or primary["candidate_sha256"] != finalist_sha256:
            raise ValueError("Strong-opponent confirmation requires an already declared finalist")
        if primary["pool_sha256"] != pool_sha256:
            raise ValueError("Strong-opponent confirmation must use the reserved final pool")
        if primary["sources_sha256"] != sources_sha256:
            raise ValueError(
                "Confirmation sources cannot change between native and strong-opponent comparisons"
            )
        if evaluated_sha256 not in (finalist_sha256, original_sha256):
            raise ValueError(
                "Only the declared finalist and unchanged baseline may use this cohort"
            )
        row = {
            "seed": seed,
            "games": games,
            "original_sha256": original_sha256,
            "opponent_sha256": opponent_sha256,
        }
        strong = ledger.setdefault("strong", {})
        key = str(attempt)
        if key in strong and strong[key] != row:
            raise ValueError("Strong-opponent arms must use one immutable matched schedule")
        strong[key] = row
