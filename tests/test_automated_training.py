from __future__ import annotations

import json
from pathlib import Path

from fog.scripts.automated_training import (
    _adaptive_training_profile,
    _config_chain,
    _latest_weight_checkpoint,
    _load_state,
    _prune_numbered_checkpoints,
    _record_training_result,
    _refresh_safety_reports,
    summarize_evaluation,
)
from fog.scripts.automation_job import read_job


def test_active_typed_league_uses_portable_checkpoint_artifacts():
    root = Path(__file__).resolve().parents[1]
    job = read_job(root / "configs" / "automated_gen9ou_typed_league.yaml")
    checkpoints = [job["initial_checkpoint"], *job["initial_league_checkpoints"]]

    assert all(Path(path).parts[:2] == ("artifacts", "checkpoints") for path in checkpoints)
    assert all((root / path).is_file() for path in checkpoints)


def test_latest_weight_checkpoint_ignores_final_and_non_numeric(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name in ("ppo_24000.pt", "ppo_48000.pt", "ppo_final.pt", "ppo_bad.pt"):
        (checkpoints / name).write_bytes(b"x")
    path, decisions = _latest_weight_checkpoint(tmp_path)
    assert path == checkpoints / "ppo_48000.pt"
    assert decisions == 48_000


def test_new_state_is_stable_and_human_readable(tmp_path: Path):
    job = {
        "name": "test",
        "output_dir": str(tmp_path / "run"),
        "initial_checkpoint": "model.pt",
    }
    state_path, state = _load_state(job, tmp_path)
    assert state_path.is_file()
    assert state["status"] == "ready"
    assert state["current_checkpoint"] == "model.pt"
    assert json.loads(state_path.read_text())["job"] == "test"


def test_evaluation_summary_flips_parent_head_to_head():
    report = {
        "status": "complete",
        "rows": [
            {
                "label": "parent vs candidate",
                "a_wins": 30,
                "b_wins": 34,
                "draws": 0,
                "failures": 0,
                "a_win_rate_decisive": 30 / 64,
            }
        ],
    }
    summary = summarize_evaluation(report)
    assert summary["candidate_head_to_head_win_rate"] == 34 / 64
    assert summary["recommend_viewer_update"] is False  # aggregate-only screen is not promotion evidence


def test_viewer_recommendation_rejects_benchmark_regression():
    def row(label: str, wins: int, losses: int):
        return {
            "label": label,
            "a_wins": wins,
            "b_wins": losses,
            "draws": 0,
            "failures": 0,
            "a_win_rate_decisive": wins / (wins + losses),
        }

    report = {
        "status": "complete",
        "rows": [
            row("parent vs candidate", 29, 35),
            row("parent vs maxdamage", 55, 9),
            row("candidate vs maxdamage", 50, 14),
        ],
    }
    summary = summarize_evaluation(report)
    assert summary["candidate_head_to_head_win_rate"] == 35 / 64
    assert summary["recommend_viewer_update"] is False


def test_config_chain_includes_all_parents(tmp_path: Path):
    base = tmp_path / "base.yaml"
    middle = tmp_path / "middle.yaml"
    leaf = tmp_path / "leaf.yaml"
    base.write_text("seed: 1\n", encoding="utf-8")
    middle.write_text(f"extends: {base}\n", encoding="utf-8")
    leaf.write_text(f"extends: {middle}\n", encoding="utf-8")
    assert _config_chain(leaf, tmp_path) == {base.resolve(), middle.resolve(), leaf.resolve()}


def test_weak_results_advance_but_confirmed_collapse_falls_back():
    state = {}
    selected, paused = _record_training_result(state, "good.pt", "weak1.pt", True)
    assert (selected, paused) == ("weak1.pt", False)
    selected, paused = _record_training_result(
        state, "weak1.pt", "weak2.pt", True, confirmed_collapse=True
    )
    assert (selected, paused) == ("weak1.pt", False)


def test_more_than_two_uses_of_same_fallback_pauses():
    state = {"stable_checkpoints": ["good.pt"]}
    for index in range(2):
        selected, paused = _record_training_result(
            state,
            "good.pt",
            f"bad{index}.pt",
            True,
            confirmed_collapse=True,
            backup_limit=1,
        )
        assert (selected, paused) == ("good.pt", False)
    selected, paused = _record_training_result(
        state,
        "good.pt",
        "bad2.pt",
        True,
        confirmed_collapse=True,
        backup_limit=1,
    )
    assert (selected, paused) == ("good.pt", True)


def test_collapse_requires_head_to_head_and_anchor_regression():
    def row(label: str, wins: int, losses: int):
        return {
            "label": label,
            "a_wins": wins,
            "b_wins": losses,
            "draws": 0,
            "failures": 0,
            "a_win_rate_decisive": wins / (wins + losses),
        }

    report = {
        "status": "complete",
        "rows": [
            row("parent vs candidate", 40, 24),
            row("parent vs maxdamage", 48, 16),
            row("candidate vs maxdamage", 42, 22),
        ],
    }
    summary = summarize_evaluation(report)
    assert summary["candidate_head_to_head_win_rate"] == 24 / 64
    assert summary["collapse_signal"] is True

    report["rows"][-1] = row("candidate vs maxdamage", 47, 17)
    assert summarize_evaluation(report)["collapse_signal"] is False


def test_adaptive_profile_diversifies_and_extends_after_repeated_fallback():
    job = {
        "training": {"decisions_per_block": 240_000, "lr": 3e-5},
        "adaptation": {
            "diversity_after_nonpromotions": 3,
            "fallback_lr_scale": 0.5,
            "repeated_fallback_block_scale": 2.0,
            "diversity_overrides": {"league.past_frac": 0.55},
        },
    }
    profile = _adaptive_training_profile(
        job,
        {"blocks_since_promotion": 3, "fallback_uses": {"good.pt": 2}},
    )
    assert profile["name"] == "extended_recovery"
    assert profile["decisions"] == 480_000
    assert profile["lr"] == 1.5e-5
    assert profile["overrides"]["league.past_frac"] == 0.55


def test_checkpoint_retention_keeps_two_numeric_and_named_final(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name in ("ppo_1.pt", "ppo_2.pt", "ppo_3.pt", "ppo_final.pt"):
        (checkpoint_dir / name).write_bytes(b"model")

    report = _prune_numbered_checkpoints(tmp_path, keep=2)

    assert not (checkpoint_dir / "ppo_1.pt").exists()
    assert (checkpoint_dir / "ppo_2.pt").exists()
    assert (checkpoint_dir / "ppo_3.pt").exists()
    assert (checkpoint_dir / "ppo_final.pt").exists()
    assert report["removed_bytes"] == 5


def test_stale_safety_reports_are_refreshed_and_rechecked(tmp_path: Path, monkeypatch):
    replay_config = tmp_path / "replay.yaml"
    replay_config.write_text("seed: 1\n", encoding="utf-8")
    direct = tmp_path / "direct" / "summary.json"
    replay = tmp_path / "replay" / "summary.json"
    calls = []
    checks = iter([RuntimeError("stale"), None])

    def fake_require(_config):
        result = next(checks)
        if result:
            raise result

    def fake_run(command, stdout, stderr):
        calls.append((command, stdout, stderr))

    monkeypatch.setattr("fog.scripts.automated_training._require_current_safety_reports", fake_require)
    monkeypatch.setattr("fog.scripts.automated_training._run_process", fake_run)
    refreshed = _refresh_safety_reports(
        {
            "safety": {
                "auto_refresh": True,
                "games": 5,
                "device": "cpu",
                "replay_audit_config": str(replay_config),
            }
        },
        tmp_path,
        {"preflight": {"report": str(direct), "replay_report": str(replay)}},
        tmp_path / "logs",
    )

    assert refreshed is True
    assert len(calls) == 2
    assert "fog.scripts.preflight_audit" in calls[0][0]
    assert "fog.scripts.replay_preflight_audit" in calls[1][0]
