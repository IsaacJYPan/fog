"""Mechanical preflight gate for long-running training jobs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def source_fingerprint(repo_root: str | Path = ".") -> str:
    """Hash every local source/input that can change observation or direct-Showdown mechanics."""
    root = Path(repo_root).resolve()
    files: set[Path] = set()
    suffixes = {".py", ".js", ".json"}
    for relative in ("fog", "data/teams"):
        base = root / relative
        if base.is_dir():
            files.update(
                path for path in base.rglob("*")
                if path.is_file() and path.suffix in suffixes | {".txt"}
            )
    config_root = root / "configs"
    if config_root.is_dir():
        files.update(config_root.rglob("*.yaml"))
    showdown = root / "server" / "pokemon-showdown"
    for relative in ("sim", "data", "lib"):
        base = showdown / relative
        if base.is_dir():
            files.update(
                path for path in base.rglob("*")
                if path.is_file() and path.suffix in {".ts", ".js", ".json"}
            )
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_file():
            files.add(path)

    digest = hashlib.sha256(b"fog-preflight-source-v1\0")
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def require_preflight(cfg: dict) -> dict | None:
    """Validate a no-training Showdown flight-recorder report when configured.

    Smoke tests and ordinary configs remain unaffected. Long recipes opt in with::

        preflight: {required: true, report: runs/preflight_audit/latest.json, min_games: 5}
    """
    gate = cfg.get("preflight", {}) or {}
    if not gate.get("required", False):
        return None
    path = Path(gate.get("report", "runs/preflight_audit/latest.json"))
    if not path.is_file():
        raise RuntimeError(
            f"training preflight is required but {path} does not exist; run "
            f"`python -m fog.scripts.preflight_audit --games {int(gate.get('min_games', 5))}`"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if not report.get("passed", False):
        problems.append("the recorder reported failed invariants")
    completed = int(report.get("games_completed", 0))
    minimum = int(gate.get("min_games", 5))
    if completed < minimum:
        problems.append(f"only {completed} games completed (need {minimum})")
    if report.get("backend") != "showdown_direct":
        problems.append("the report was not produced by authoritative direct Showdown")
    if int(report.get("optimizer_steps", -1)) != 0:
        problems.append("the report was not a zero-optimizer inspection run")
    minimum_dryruns = int(gate.get("min_replay_dryruns", 0))
    if int(report.get("replay_dryruns", 0)) < minimum_dryruns:
        problems.append(
            f"only {report.get('replay_dryruns', 0)} replay loss/backward dry-runs "
            f"(need {minimum_dryruns})"
        )
    minimum_targets = int(gate.get("min_replay_hidden_targets", 0))
    if int(report.get("replay_hidden_targets", 0)) < minimum_targets:
        problems.append(
            f"only {report.get('replay_hidden_targets', 0)} replay hidden targets "
            f"(need {minimum_targets})"
        )
    required_schema = int(gate.get("schema_version", 0))
    if int(report.get("schema_version", 0)) < required_schema:
        problems.append(
            f"schema_version={report.get('schema_version')} (need >= {required_schema})"
        )
    if gate.get("match_source", False):
        current = source_fingerprint()
        recorded = report.get("source_fingerprint")
        if recorded != current:
            problems.append("the audited source fingerprint does not match the current workspace")

    replay_report_path = gate.get("replay_report")
    if replay_report_path:
        replay_path = Path(replay_report_path)
        if not replay_path.is_file():
            problems.append(
                f"replay parser audit {replay_path} does not exist; run "
                "`python -m fog.scripts.replay_preflight_audit --games 5`"
            )
        else:
            replay_report = json.loads(replay_path.read_text(encoding="utf-8"))
            if not replay_report.get("passed", False):
                problems.append("the replay parser audit reported failed invariants")
            replay_games = int(replay_report.get("games_completed", 0))
            min_replay_games = int(gate.get("min_parsed_replay_games", 0))
            if replay_games < min_replay_games:
                problems.append(
                    f"only {replay_games} parsed replay games (need {min_replay_games})"
                )
            replay_dryruns = int(replay_report.get("perspective_dryruns", 0))
            min_parsed_dryruns = int(gate.get("min_parsed_replay_dryruns", 0))
            if replay_dryruns < min_parsed_dryruns:
                problems.append(
                    f"only {replay_dryruns} parsed replay loss/backward dry-runs "
                    f"(need {min_parsed_dryruns})"
                )
            replay_rows = int(replay_report.get("decision_rows", 0))
            min_replay_rows = int(gate.get("min_parsed_replay_rows", 0))
            if replay_rows < min_replay_rows:
                problems.append(
                    f"only {replay_rows} parsed replay decision rows (need {min_replay_rows})"
                )
            if replay_report.get("backend") != "public_showdown_replay":
                problems.append("the replay parser audit did not use public Showdown replays")
            if int(replay_report.get("optimizer_steps", -1)) != 0:
                problems.append("the replay parser audit was not a zero-optimizer inspection run")
            replay_schema = int(gate.get("replay_schema_version", 0))
            if int(replay_report.get("schema_version", 0)) < replay_schema:
                problems.append(
                    f"replay schema_version={replay_report.get('schema_version')} "
                    f"(need >= {replay_schema})"
                )
            if gate.get("match_source", False):
                replay_recorded = replay_report.get("source_fingerprint")
                if replay_recorded != source_fingerprint():
                    problems.append(
                        "the replay-audited source fingerprint does not match the current workspace"
                    )
    if problems:
        raise RuntimeError(f"training preflight gate rejected {path}: " + "; ".join(problems))
    return report
