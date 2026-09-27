"""Resumable, portable training jobs for local machines and Kaggle.

The job boundary is deliberately a complete training block.  A killed block can be
recovered from its newest saved model, while completed blocks and evaluations are never
repeated.  Training and evaluation use the direct in-memory Showdown worker; this driver
never starts or connects to the viewing server.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from fog.config import load_config
from fog.scripts.automation_job import SCHEMA_VERSION
from fog.scripts.automation_job import detect_node as _detect_node
from fog.scripts.automation_job import portable_path as _portable_path
from fog.scripts.automation_job import read_job as _read_yaml
from fog.scripts.automation_job import resolve_path as _resolve
from fog.scripts.automation_job import validate_job as _validate_job
from fog.scripts.kaggle_bundle import bundle_for_kaggle
from fog.scripts.kaggle_bundle import config_chain as _config_chain

__all__ = ["_config_chain", "bundle_for_kaggle"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_state(job: dict[str, Any], root: Path) -> tuple[Path, dict[str, Any]]:
    output = _resolve(root, job["output_dir"])
    state_path = output / "pipeline_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "job": job["name"],
            "status": "ready",
            "created_at": _now(),
            "updated_at": _now(),
            "initial_checkpoint": str(job["initial_checkpoint"]),
            "current_checkpoint": str(job["initial_checkpoint"]),
            "champion_checkpoint": str(job["initial_checkpoint"]),
            "stable_checkpoints": [str(job["initial_checkpoint"])],
            "league_checkpoints": list(
                job.get("initial_league_checkpoints", [str(job["initial_checkpoint"])])
            ),
            "blocks_since_promotion": 0,
            "confirmed_fallbacks": 0,
            "blocks": [],
        }
        _atomic_json(state_path, state)
    # Forward-compatible defaults for jobs created by older versions of the driver.
    initial = str(state.get("initial_checkpoint", job["initial_checkpoint"]))
    state.setdefault("champion_checkpoint", initial)
    state.setdefault("stable_checkpoints", [initial])
    state.setdefault(
        "league_checkpoints", list(job.get("initial_league_checkpoints", [initial]))
    )
    state.setdefault("blocks_since_promotion", 0)
    state.setdefault("confirmed_fallbacks", 0)
    return state_path, state


def _latest_weight_checkpoint(run_dir: Path) -> tuple[Path | None, int]:
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.exists():
        return None, 0
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints.glob("ppo_*.pt"):
        try:
            decisions = int(path.stem.removeprefix("ppo_"))
        except ValueError:
            continue
        candidates.append((decisions, path))
    if not candidates:
        return None, 0
    decisions, path = max(candidates)
    return path, decisions


def _promotion_design(job: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Freeze the number of looks and selection rules before their outcomes are observed."""
    evaluation = job.get("evaluation", {})
    design = {
        "looks": int(job["training"]["blocks"]),
        "family_alpha": float(evaluation.get("promotion_family_alpha", 0.05)),
        "games": int(evaluation.get("champion_games", 1024)),
        "seed": int(evaluation.get("seed", 10_000)),
        "seed_offset": int(evaluation.get("champion_seed_offset", 200_000)),
        "opponents": list(evaluation.get("opponents", ["simpleheuristics", "maxdamage"])),
        "minimum_score": float(evaluation.get("viewer_min_head_to_head", 0.53)),
        "maximum_regression": float(evaluation.get("viewer_max_benchmark_regression", 0.03)),
        "pool": str(evaluation.get("pool", "")),
        "pool_role": str(evaluation.get("promotion_pool_role", "validation")),
        "minimum_head_games": max(1000, int(evaluation.get("promotion_head_min_games", 1000))),
        "format": str(evaluation.get("format", "gen9ou")),
    }
    if design["looks"] < 1 or not 0 < design["family_alpha"] < 1:
        raise ValueError("promotion design needs positive looks and a family alpha in (0,1)")
    previous = state.setdefault("promotion_design", design)
    if previous != design:
        raise ValueError("promotion design changed on resume; start a new independently evaluated job")
    return design


def _deep_set(config: dict[str, Any], dotted: str, value: Any) -> None:
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _block_config(
    job: dict[str, Any], root: Path, run_dir: Path, checkpoint: Path, decisions: int, seed: int,
    *, profile: dict[str, Any] | None = None, state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_config(_resolve(root, job["base_config"]))
    training = job["training"]
    config["experiment"] = f"{job['name']}_{run_dir.name}"
    config["seed"] = seed
    config.setdefault("ppo", {}).update(
        total_decisions=decisions,
        batch_decisions=int(training.get("batch_decisions", 24_000)),
        minibatch_size=int(training.get("minibatch_size", 1536)),
        epochs=int(training.get("epochs", 1)),
        lr=float(training.get("lr", 3e-5)),
        clip=float(training.get("clip", 0.05)),
        init_ckpt=str(checkpoint),
    )
    config.setdefault("league", {})["snapshot_every_decisions"] = int(
        training.get("checkpoint_every_decisions", 24_000)
    )
    config.setdefault("log", {}).update(
        dir=str(run_dir),
        checkpoint_every_decisions=int(training.get("checkpoint_every_decisions", 24_000)),
        tensorboard=bool(training.get("tensorboard", True)),
    )
    for key, value in (job.get("overrides", {}) or {}).items():
        _deep_set(config, key, value)
    for key, value in ((profile or {}).get("overrides", {}) or {}).items():
        _deep_set(config, key, value)
    if state is not None:
        # Only checkpoints produced by this job belong here.  This avoids accidentally loading an
        # incompatible historical architecture while still preserving strategy diversity across
        # blocks.  The current checkpoint is supplied separately as league.latest_ckpt by PPO.
        limit = int((job.get("retention", {}) or {}).get("league_checkpoints", 8))
        history = [
            value for value in state.get("league_checkpoints", [])
            if _resolve(root, value).is_file()
        ]
        # On the first block a single checkpoint legitimately fills both current and historical
        # roles.  Once alternatives exist, remove the duplicate current entry.
        if len(history) > 1:
            history = [value for value in history if _resolve(root, value) != checkpoint]
        if history:
            configured = list(config.setdefault("league", {}).get("past_ckpts", []))
            merged = list(dict.fromkeys(configured + history))
            config["league"]["past_ckpts"] = merged[-limit:]
    # A portable environment always uses the direct worker's normal `node` lookup.  A user can
    # still override it explicitly with FOG_NODE_BINARY.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        config.setdefault("env", {})["node_binary"] = "node"
        config["env"]["showdown_root"] = str(root / "server" / "pokemon-showdown")
    return config


def _run_process(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        stdout.write(f"\n[{_now()}] $ {' '.join(command)}\n")
        stdout.flush()
        environment = os.environ.copy()
        if not environment.get("FOG_NODE_BINARY"):
            node = _detect_node()
            if node:
                environment["FOG_NODE_BINARY"] = node
        completed = subprocess.run(
            command, stdout=stdout, stderr=stderr, env=environment, check=False
        )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def _require_current_safety_reports(config: dict[str, Any]) -> None:
    from fog.training.preflight import require_preflight

    require_preflight(config)


def _refresh_safety_reports(
    job: dict[str, Any], root: Path, config: dict[str, Any], log_dir: Path
) -> bool:
    """Refresh stale zero-update safety reports, then verify the new reports.

    The source fingerprint intentionally changes when Python or training configuration changes.
    Long jobs cross that boundary surprisingly often, so each block checks it explicitly.  An
    opted-in refresh runs both no-optimizer audits; it never weakens or bypasses the gate.
    """
    try:
        _require_current_safety_reports(config)
        return False
    except (OSError, RuntimeError, ValueError):
        safety = job.get("safety", {}) or {}
        if not safety.get("auto_refresh", False):
            raise

    preflight = config.get("preflight", {}) or {}
    direct_report = _resolve(root, preflight.get("report", ""))
    replay_report = _resolve(root, preflight.get("replay_report", ""))
    if not direct_report.name or not replay_report.name:
        raise ValueError("automatic safety refresh requires preflight.report and replay_report")
    replay_config_value = (job.get("safety", {}) or {}).get("replay_audit_config")
    if not replay_config_value:
        raise ValueError("safety.replay_audit_config is required when auto_refresh is enabled")
    replay_config = _resolve(root, replay_config_value)
    if not replay_config.is_file():
        raise FileNotFoundError(f"replay safety config not found: {replay_config}")

    games = int((job.get("safety", {}) or {}).get("games", 5))
    device = str((job.get("safety", {}) or {}).get("device", "cuda"))
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = (
        (
            [
                sys.executable,
                "-m",
                "fog.scripts.preflight_audit",
                "--games",
                str(games),
                "--device",
                device,
                "--out",
                str(direct_report.parent),
            ],
            "direct",
        ),
        (
            [
                sys.executable,
                "-m",
                "fog.scripts.replay_preflight_audit",
                "--games",
                str(games),
                "--config",
                str(replay_config),
                "--device",
                device,
                "--out",
                str(replay_report.parent),
            ],
            "replay",
        ),
    )
    for command, name in commands:
        _run_process(command, log_dir / f"{name}_stdout.log", log_dir / f"{name}_stderr.log")
    _require_current_safety_reports(config)
    return True


def _evaluation_command(
    job: dict[str, Any], config_path: Path, parent: Path, candidate: Path, out: Path, seed: int,
    *, settings: dict[str, Any] | None = None,
) -> list[str]:
    evaluation = settings or job.get("evaluation", {})
    command = [
        sys.executable,
        "-m",
        "fog.eval.showdown_direct_selection",
        "--pool",
        str(evaluation["pool"]),
        "--config",
        str(config_path),
        "--checkpoint",
        f"parent={parent}",
        "--checkpoint",
        f"candidate={candidate}",
    ]
    for opponent in evaluation.get("opponents", ["simpleheuristics", "maxdamage"]):
        command += ["--opponent", str(opponent).lower()]
    if evaluation.get("head_to_head", True):
        command.append("--head-to-head")
    command += [
        "--games",
        str(int(evaluation.get("games", 64))),
        "--seed",
        str(seed),
        "--format",
        str(evaluation.get("format", "gen9ou")),
        "--device",
        str(evaluation.get("device", "cuda")),
        "--progress-every",
        str(int(evaluation.get("progress_every", 4))),
        "--max-turns",
        str(int(evaluation.get("max_turns", 120))),
        "--action-stats",
        "--resume",
        "--out",
        str(out),
    ]
    return command


def summarize_evaluation(
    report: dict[str, Any], settings: dict[str, Any] | None = None
) -> dict[str, Any]:
    from dataclasses import asdict

    from fog.eval.stats import paired_block_interval

    gate = settings or {}
    rows = report.get("rows", [])
    summary: dict[str, Any] = {"status": report.get("status"), "rows": {}}
    for row in rows:
        wins, losses = int(row["a_wins"]), int(row["b_wins"])
        draws, truncated = int(row["draws"]), int(row.get("truncated", 0))
        finished = wins + losses + draws - truncated
        scores = row.get("game_scores", [])
        requested = row.get("requested_games")
        complete = bool(
            requested is not None and requested > 0 and requested % 4 == 0
            and row.get("completed_games") == requested and row["failures"] == 0
            and truncated == 0 and finished == requested and len(scores) == requested
            and all(score in (0.0, 0.5, 1.0) for score in scores)
            and scores.count(1.0) == wins and scores.count(0.0) == losses
            and scores.count(0.5) == draws
        )
        summary["rows"][row["label"]] = {
            "wins": wins,
            "losses": losses,
            "draws": draws - truncated,
            "truncated": truncated,
            "requested_games": requested,
            "completed_games": row.get("completed_games"),
            "finished_games": finished,
            "failures": int(row["failures"]),
            "win_rate": (wins + 0.5 * (draws - truncated)) / finished if finished > 0 else None,
            "game_scores": scores,
            "complete_evidence": complete,
        }
    head = summary["rows"].get("parent vs candidate")
    summary["candidate_head_to_head_win_rate"] = (
        1.0 - head["win_rate"] if head is not None and head["win_rate"] is not None else None
    )
    deltas = {}
    for label, parent_row in summary["rows"].items():
        if not label.startswith("parent vs ") or label == "parent vs candidate":
            continue
        opponent = label.removeprefix("parent vs ")
        candidate_row = summary["rows"].get(f"candidate vs {opponent}")
        if (candidate_row is not None and parent_row["win_rate"] is not None
                and candidate_row["win_rate"] is not None):
            deltas[opponent] = candidate_row["win_rate"] - parent_row["win_rate"]
    summary["benchmark_win_rate_deltas"] = deltas
    minimum_head = float(gate.get("viewer_min_head_to_head", 0.53))
    maximum_regression = float(gate.get("viewer_max_benchmark_regression", 0.03))
    required = {"parent vs candidate"}
    for opponent in gate.get("opponents", []):
        required.update((f"parent vs {opponent}", f"candidate vs {opponent}"))
    complete = bool(
        report.get("status") == "complete" and required <= summary["rows"].keys()
        and all(row["complete_evidence"] for row in summary["rows"].values())
    )
    summary["complete_evidence"] = complete
    summary["candidate_score_interval"] = None
    summary["benchmark_delta_intervals"] = {}
    # One family includes head-to-head and every paired anchor comparison.
    alpha = float(gate.get("promotion_alpha", 0.05)) / (1 + len(deltas))
    minimum_games = max(128, int(gate.get("promotion_min_games", 128)))
    enough = complete and all(row["finished_games"] >= minimum_games
                              for row in summary["rows"].values())
    if enough:
        summary["candidate_score_interval"] = asdict(paired_block_interval(
            [1.0 - value for value in head["game_scores"]], alpha=alpha,
        ))
        for opponent in deltas:
            parent_scores = summary["rows"][f"parent vs {opponent}"]["game_scores"]
            candidate_scores = summary["rows"][f"candidate vs {opponent}"]["game_scores"]
            if len(parent_scores) != len(candidate_scores):
                enough = False
                break
            summary["benchmark_delta_intervals"][opponent] = asdict(paired_block_interval(
                [candidate - parent for candidate, parent in zip(candidate_scores, parent_scores)],
                alpha=alpha, bounds=(-1.0, 1.0),
            ))
    summary["promotion_uncertainty_method"] = "paired four-game block percentile bootstrap"
    summary["promotion_alpha_per_comparison"] = alpha
    interval = summary["candidate_score_interval"]
    summary["recommend_viewer_update"] = bool(
        gate.get("promotion_pool_role") == "test"
        and head is not None
        and head["finished_games"] >= max(1000, int(gate.get("promotion_head_min_games", 1000)))
        and enough and interval is not None and interval["low"] > 0.5
        and summary["candidate_head_to_head_win_rate"] is not None
        and summary["candidate_head_to_head_win_rate"] >= minimum_head
        and all(delta >= -maximum_regression for delta in deltas.values())
        and all(value["low"] >= -maximum_regression
                for value in summary["benchmark_delta_intervals"].values())
    )
    summary["promotion_scope"] = gate.get("promotion_pool_role", "validation")
    deterioration_head = float(gate.get("deterioration_head_to_head_below", 0.47))
    deterioration_regression = float(gate.get("deterioration_benchmark_regression", 0.03))
    summary["deteriorated"] = bool(
        any(row["failures"] > 0 for row in summary["rows"].values())
        or (
            summary["candidate_head_to_head_win_rate"] is not None
            and summary["candidate_head_to_head_win_rate"] < deterioration_head
        )
        or any(delta < -deterioration_regression for delta in deltas.values())
    )
    # Ordinary 64-game variance is diagnostic, not a reason to throw away learning.  A rollback
    # signal requires both a decisive head-to-head loss and a material anchor regression.  The
    # caller must repeat this test on an independent seed before acting on it.
    collapse_head = float(gate.get("collapse_head_to_head_below", 0.42))
    collapse_regression = float(gate.get("collapse_benchmark_regression", 0.05))
    head_collapse = bool(
        head is not None and head["truncated"] == 0
        and head["finished_games"] >= int(gate.get("games", 64))
        and
        summary["candidate_head_to_head_win_rate"] is not None
        and summary["candidate_head_to_head_win_rate"] < collapse_head
    )
    anchor_collapse = any(delta < -collapse_regression for delta in deltas.values())
    technical_failure = any(row["failures"] > 0 for row in summary["rows"].values())
    summary["collapse_signal"] = bool(technical_failure or (head_collapse and anchor_collapse))
    return summary


def _adaptive_training_profile(job: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Return explicit, reproducible adaptations for stagnation or confirmed collapse."""
    training = job["training"]
    adaptation = job.get("adaptation", {}) or {}
    profile: dict[str, Any] = {
        "name": "normal",
        "decisions": int(training["decisions_per_block"]),
        "lr": float(training.get("lr", 3e-5)),
        "overrides": {},
    }
    stagnant = int(state.get("blocks_since_promotion", 0))
    diversity_after = int(adaptation.get("diversity_after_nonpromotions", 3))
    if stagnant >= diversity_after:
        profile["name"] = "diversity"
        profile["overrides"].update(adaptation.get("diversity_overrides", {}) or {})

    uses = state.get("fallback_uses", {}) or {}
    repeated = max((int(value) for value in uses.values()), default=0)
    if repeated >= 1:
        profile["name"] = "recovery"
        profile["lr"] *= float(adaptation.get("fallback_lr_scale", 0.5))
        profile["overrides"].update(adaptation.get("recovery_overrides", {}) or {})
    if repeated >= 2:
        profile["name"] = "extended_recovery"
        profile["decisions"] = round(
            profile["decisions"] * float(adaptation.get("repeated_fallback_block_scale", 2.0))
        )
    profile["overrides"]["ppo.lr"] = profile["lr"]
    return profile


def _record_training_result(
    state: dict[str, Any], parent: str, candidate: str, deteriorated: bool,
    *, confirmed_collapse: bool = False, backup_limit: int = 2,
    maximum_fallback_uses: int = 2,
) -> tuple[str, bool]:
    """Advance the learner unless two-seed evidence confirmed a real collapse."""
    backups = list(state.setdefault("stable_checkpoints", []))
    uses = state.setdefault("fallback_uses", {})
    paused = False
    if confirmed_collapse:
        if parent not in backups:
            backups.append(parent)
        candidates = [parent] + [item for item in reversed(backups) if item != parent]
        selected = next(
            (item for item in candidates if int(uses.get(item, 0)) < maximum_fallback_uses),
            candidates[0],
        )
        uses[selected] = int(uses.get(selected, 0)) + 1
        state["confirmed_fallbacks"] = int(state.get("confirmed_fallbacks", 0)) + 1
        if uses[selected] > maximum_fallback_uses:
            paused = True
    else:
        if parent != candidate and parent not in backups:
            backups.append(parent)
        selected = candidate
        if candidate not in backups:
            backups.append(candidate)
    state["stable_checkpoints"] = backups[-backup_limit:] if backup_limit else []
    # Keep the old field readable for existing status tooling.
    state["fallback_checkpoints"] = list(state["stable_checkpoints"])
    state["fallback_uses"] = uses
    state["last_result_weak"] = bool(deteriorated)
    state["last_collapse_confirmed"] = bool(confirmed_collapse)
    return selected, paused


def _prune_numbered_checkpoints(run_dir: Path, keep: int = 2) -> dict[str, Any]:
    """Remove old numeric PPO snapshots after a block has been evaluated."""
    checkpoint_dir = run_dir / "checkpoints"
    numbered: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("ppo_*.pt"):
        try:
            numbered.append((int(path.stem.removeprefix("ppo_")), path))
        except ValueError:
            continue
    numbered.sort()
    removed = []
    for _, path in numbered[:-keep] if keep else numbered:
        size = path.stat().st_size
        path.unlink()
        removed.append({"path": str(path), "bytes": size})
    report = {
        "kept_numbered": [str(path) for _, path in (numbered[-keep:] if keep else [])],
        "removed": removed,
        "removed_bytes": sum(item["bytes"] for item in removed),
    }
    if checkpoint_dir.exists():
        _atomic_json(run_dir / "checkpoint_retention.json", report)
    return report


def _write_human_status(output: Path, state: dict[str, Any]) -> None:
    lines = [
        f"Job: {state['job']}",
        f"Status: {state['status']}",
        f"Learner: {state['current_checkpoint']}",
        f"Champion: {state.get('champion_checkpoint', state['current_checkpoint'])}",
        f"Completed blocks: {sum(b.get('status') == 'complete' for b in state['blocks'])}",
        f"Blocks since champion promotion: {state.get('blocks_since_promotion', 0)}",
        f"Updated: {state['updated_at']}",
    ]
    (output / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_job(job_path: Path, *, dry_run: bool = False, max_blocks: int | None = None) -> None:
    root = Path.cwd().resolve()
    job = _read_yaml(job_path)
    _validate_job(job, root)
    state_path, state = _load_state(job, root)
    promotion_design = _promotion_design(job, state)
    _atomic_json(state_path, state)
    output = state_path.parent
    training = job["training"]
    total_blocks = int(training["blocks"])
    if max_blocks is not None:
        total_blocks = min(total_blocks, int(max_blocks))
    seed0 = int(training.get("seed", 0))
    current = _resolve(root, state["current_checkpoint"])
    if state.get("status") == "paused_for_large_evaluation":
        print("Training is paused: the same fallback model has been needed repeatedly.")
        print("Run a larger evaluation before resuming or changing the training design.")
        return
    completed = {int(item["index"]) for item in state["blocks"] if item.get("status") == "complete"}

    for index in range(1, total_blocks + 1):
        if index in completed:
            continue
        profile = _adaptive_training_profile(job, state)
        block_root = output / f"block_{index:03d}"
        final = block_root / "checkpoints" / "ppo_final.pt"
        target = int(profile["decisions"])
        latest, seen = _latest_weight_checkpoint(block_root)
        run_dir = block_root
        start = current
        remaining = target
        recovered_from = None
        if final.is_file():
            candidate = final
        else:
            if latest is not None and seen < target:
                # PPO currently stores model weights but not Adam moments.  Recovery is honest
                # about that boundary and never overwrites the partial block.
                recovered_from = str(latest)
                start = latest
                remaining = target - seen
                run_dir = block_root / f"recovery_after_{seen}"
                final = run_dir / "checkpoints" / "ppo_final.pt"
            config = _block_config(
                job, root, run_dir, start, remaining, seed0 + index,
                profile=profile, state=state,
            )
            config_path = run_dir / "config.resolved.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [sys.executable, "-m", "fog.training.ppo", "--config", str(config_path)]
            if dry_run:
                print(" ".join(command))
                return
            record = {
                "index": index,
                "status": "training",
                "started_at": _now(),
                "parent_checkpoint": _portable_path(root, current),
                "run_dir": str(run_dir),
                "recovered_from": recovered_from,
                "training_profile": profile,
            }
            state["blocks"] = [b for b in state["blocks"] if int(b["index"]) != index] + [record]
            state.update(status="training", updated_at=_now())
            _atomic_json(state_path, state)
            _write_human_status(output, state)
            started = time.monotonic()
            try:
                refreshed = _refresh_safety_reports(
                    job, root, config, output / "safety_refreshes" / f"before_block_{index:03d}"
                )
                record["safety_reports_refreshed"] = refreshed
                _run_process(command, run_dir / "pipeline_stdout.log", run_dir / "pipeline_stderr.log")
            except Exception as exc:
                record.update(status="failed", error=str(exc), finished_at=_now())
                state.update(status="failed", updated_at=_now())
                _atomic_json(state_path, state)
                _write_human_status(output, state)
                raise
            record["train_wall_seconds"] = time.monotonic() - started
            if not final.is_file():
                raise FileNotFoundError(f"training exited without final checkpoint: {final}")
            candidate = final

        record = next((b for b in state["blocks"] if int(b["index"]) == index), None)
        if record is None:
            record = {
                "index": index,
                "parent_checkpoint": _portable_path(root, current),
                "run_dir": _portable_path(root, run_dir),
            }
            state["blocks"].append(record)
        record.update(status="evaluating", candidate_checkpoint=str(candidate))
        state.update(status="evaluating", updated_at=_now())
        _atomic_json(state_path, state)
        _write_human_status(output, state)

        evaluation = job.get("evaluation", {})
        deteriorated = False
        collapse_confirmed = False
        if evaluation.get("enabled", True):
            eval_dir = block_root / "evaluation"
            result_path = eval_dir / "results.json"
            config_path = run_dir / "config.resolved.yaml"
            command = _evaluation_command(
                job, config_path, current, candidate, result_path,
                int(evaluation.get("seed", 10_000)) + index,
            )
            if dry_run:
                print(" ".join(command))
                return
            _run_process(command, eval_dir / "stdout.log", eval_dir / "stderr.log")
            report = json.loads(result_path.read_text(encoding="utf-8"))
            if report.get("status") != "complete":
                raise RuntimeError(f"evaluation did not complete: {result_path}")
            record["evaluation"] = summarize_evaluation(report, evaluation)
            deteriorated = bool(record["evaluation"]["deteriorated"])
            if record["evaluation"].get("collapse_signal", False):
                confirmation = copy.deepcopy(evaluation)
                confirmation["games"] = int(evaluation.get("confirmation_games", 128))
                confirm_dir = block_root / "collapse_confirmation"
                confirm_result = confirm_dir / "results.json"
                confirm_command = _evaluation_command(
                    job, config_path, current, candidate, confirm_result,
                    int(evaluation.get("seed", 10_000))
                    + int(evaluation.get("confirmation_seed_offset", 100_000)) + index,
                    settings=confirmation,
                )
                _run_process(
                    confirm_command,
                    confirm_dir / "stdout.log",
                    confirm_dir / "stderr.log",
                )
                confirm_report = json.loads(confirm_result.read_text(encoding="utf-8"))
                if confirm_report.get("status") != "complete":
                    raise RuntimeError(f"collapse confirmation did not complete: {confirm_result}")
                record["collapse_confirmation"] = summarize_evaluation(
                    confirm_report, confirmation
                )
                collapse_confirmed = bool(
                    record["evaluation"].get("collapse_signal")
                    and record["collapse_confirmation"].get("collapse_signal")
                )

        retention = job.get("retention", {}) or {}
        selected, paused = _record_training_result(
            state,
            _portable_path(root, current),
            _portable_path(root, candidate),
            deteriorated,
            confirmed_collapse=collapse_confirmed,
            backup_limit=int(retention.get("backup_checkpoints", 2)),
            maximum_fallback_uses=int(retention.get("maximum_fallback_uses", 2)),
        )
        record["deteriorated"] = deteriorated
        record["collapse_confirmed"] = collapse_confirmed
        if selected != _portable_path(root, candidate):
            record["rolled_back_to"] = selected

        # Champion selection is independent from learner continuation.  When the learner has
        # moved through neutral blocks, compare the candidate to the protected champion rather
        # than silently treating the previous learner as the best known model.
        promoted = False
        champion = _resolve(root, state["champion_checkpoint"])
        if selected == _portable_path(root, candidate) and evaluation.get("enabled", True):
            # Always independent of the diagnostic screen, including champion == parent.
            champion_dir = block_root / "champion_evaluation"
            champion_result = champion_dir / "results.json"
            champion_settings = copy.deepcopy(evaluation)
            champion_settings["games"] = int(evaluation.get("champion_games", 1024))
            champion_settings["promotion_alpha"] = (
                promotion_design["family_alpha"] / promotion_design["looks"]
            )
            champion_command = _evaluation_command(
                job, config_path, champion, candidate, champion_result,
                int(evaluation.get("seed", 10_000))
                + int(evaluation.get("champion_seed_offset", 200_000)) + index,
                settings=champion_settings,
            )
            _run_process(champion_command, champion_dir / "stdout.log", champion_dir / "stderr.log")
            champion_report = json.loads(champion_result.read_text(encoding="utf-8"))
            if champion_report.get("status") != "complete":
                raise RuntimeError(f"champion evaluation did not complete: {champion_result}")
            champion_summary = summarize_evaluation(champion_report, champion_settings)
            record["champion_evaluation"] = champion_summary
            promoted = bool(champion_summary.get("recommend_viewer_update", False))
            if promoted:
                state["champion_checkpoint"] = _portable_path(root, candidate)
                state["recommended_viewer_checkpoint"] = _portable_path(root, candidate)
                state["blocks_since_promotion"] = 0
            else:
                state["blocks_since_promotion"] = int(
                    state.get("blocks_since_promotion", 0)
                ) + 1
        record["champion_promoted"] = promoted

        current = _resolve(root, selected)
        if selected == _portable_path(root, candidate):
            league_history = list(state.setdefault("league_checkpoints", []))
            if selected not in league_history:
                league_history.append(selected)
            league_limit = int(retention.get("league_checkpoints", 8))
            state["league_checkpoints"] = league_history[-league_limit:]
        if retention.get("prune_numbered_after_evaluation", True):
            record["checkpoint_retention"] = _prune_numbered_checkpoints(
                run_dir, int(retention.get("keep_numbered_per_block", 2))
            )
        record.update(status="complete", finished_at=_now())
        pause_after = int((job.get("adaptation", {}) or {}).get("pause_after_nonpromotions", 9))
        stagnation_pause = int(state.get("blocks_since_promotion", 0)) >= pause_after
        state.update(
            status=(
                "paused_for_large_evaluation"
                if paused or stagnation_pause
                else ("ready" if index < total_blocks else "complete")
            ),
            current_checkpoint=_portable_path(root, current),
            updated_at=_now(),
        )
        _atomic_json(state_path, state)
        _write_human_status(output, state)
        if paused or stagnation_pause:
            state["larger_evaluation_games_recommended"] = int(
                retention.get("larger_evaluation_games", 256)
            )
            state["pause_reason"] = (
                "repeated confirmed fallback"
                if paused
                else "no champion improvement across the configured number of blocks"
            )
            _atomic_json(state_path, state)
            _write_human_status(output, state)
            return


def doctor(job_path: Path) -> int:
    root = Path.cwd().resolve()
    job = _read_yaml(job_path)
    checks: list[tuple[str, bool, str]] = []
    try:
        _validate_job(job, root)
        checks.append(("job", True, "configuration and referenced files are present"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        checks.append(("job", False, str(exc)))
    checks.append(("python", sys.version_info >= (3, 11), sys.version.split()[0]))
    node = _detect_node()
    checks.append(("node", bool(node), str(node or "not found")))
    showdown = root / "server" / "pokemon-showdown" / "dist" / "sim" / "index.js"
    checks.append(("showdown", showdown.is_file(), str(showdown)))
    try:
        import torch

        checks.append(("cuda", torch.cuda.is_available(), str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "not available"))
    except (ImportError, OSError, RuntimeError) as exc:
        checks.append(("cuda", False, str(exc)))
    try:
        from fog.training.preflight import require_preflight

        training = job["training"]
        config = _block_config(
            job,
            root,
            _resolve(root, job["output_dir"]) / "doctor",
            _resolve(root, job["initial_checkpoint"]),
            int(training["decisions_per_block"]),
            int(training.get("seed", 0)),
        )
        require_preflight(config)
        checks.append(("safety reports", True, "current source fingerprint matches"))
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        checks.append(("safety reports", False, str(exc)))
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def export_job(job_path: Path, destination: Path) -> Path:
    root = Path.cwd().resolve()
    job = _read_yaml(job_path)
    _validate_job(job, root)
    output = _resolve(root, job["output_dir"])
    if not output.exists():
        raise FileNotFoundError(f"job output does not exist: {output}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(destination.with_suffix("")), "zip", root_dir=output)
    print(archive)
    return Path(archive)


def show_status(job_path: Path) -> None:
    root = Path.cwd().resolve()
    job = _read_yaml(job_path)
    state_path, state = _load_state(job, root)
    _write_human_status(state_path.parent, state)
    print((state_path.parent / "STATUS.txt").read_text(encoding="utf-8"), end="")
    for block in sorted(state["blocks"], key=lambda item: int(item["index"])):
        print(f"Block {block['index']}: {block['status']}")
        evaluation = block.get("evaluation", {}).get("rows", {})
        for label, row in evaluation.items():
            print(f"  {label}: {row['wins']}-{row['losses']} draws={row['draws']} failures={row['failures']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume training and automatic evaluations")
    run.add_argument("--job", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--max-blocks", type=int)
    status = sub.add_parser("status", help="print a concise job status")
    status.add_argument("--job", type=Path, required=True)
    check = sub.add_parser("doctor", help="validate a local or Kaggle environment")
    check.add_argument("--job", type=Path, required=True)
    export = sub.add_parser("export", help="zip all state, logs, reports, and checkpoints")
    export.add_argument("--job", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    bundle = sub.add_parser("bundle-kaggle", help="build one uploadable Kaggle input zip")
    bundle.add_argument("--job", type=Path, required=True)
    bundle.add_argument("--out", type=Path, required=True)
    bundle.add_argument("--dry-run", action="store_true")
    bundle.add_argument("--allow-missing-inputs", action="store_true",
                        help="create a setup bundle with an explicit missing-input inventory")
    args = parser.parse_args()
    try:
        if args.command == "run":
            run_job(args.job, dry_run=args.dry_run, max_blocks=args.max_blocks)
        elif args.command == "status":
            show_status(args.job)
        elif args.command == "doctor":
            raise SystemExit(doctor(args.job))
        elif args.command == "export":
            export_job(args.job, args.out)
        elif args.command == "bundle-kaggle":
            bundle_for_kaggle(args.job, args.out, dry_run=args.dry_run,
                              allow_missing_inputs=args.allow_missing_inputs)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
