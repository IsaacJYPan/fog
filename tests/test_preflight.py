"""Training gate contracts for the mechanical flight recorder."""
from __future__ import annotations

import json

import pytest

from fog.training.preflight import require_preflight, source_fingerprint


def test_preflight_gate_is_opt_in():
    assert require_preflight({}) is None


@pytest.mark.parametrize("path", [
    "fog/encoding/observation.py", "fog/env/showdown_worker.js",
    "fog/models/belief.py", "fog/training/losses.py",
    "fog/constants/generated/moves.json", "fog/scripts/parse_replays.py",
    "fog/scripts/preflight_audit.py", "fog/scripts/replay_preflight_audit.py",
    "server/pokemon-showdown/sim/battle.ts", "server/pokemon-showdown/package-lock.json",
    "uv.lock",
])
def test_fingerprint_tracks_audited_code_and_mechanics(tmp_path, path):
    source = tmp_path / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("before", encoding="utf-8")
    before = source_fingerprint(tmp_path)
    source.write_text("after", encoding="utf-8")
    assert source_fingerprint(tmp_path) != before


@pytest.mark.parametrize("path", [
    "README.md", "configs/archive/old.yaml", "configs/another_experiment.yaml",
    "data/teams/unused_team.txt", "fog/scripts/kaggle_results.py", "fog/eval/gauntlet.py",
])
def test_fingerprint_ignores_unrelated_inputs_and_tools(tmp_path, path):
    before = source_fingerprint(tmp_path)
    source = tmp_path / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("unrelated edit", encoding="utf-8")
    assert source_fingerprint(tmp_path) == before


def test_preflight_gate_rejects_missing_or_failed_report(tmp_path):
    path = tmp_path / "latest.json"
    cfg = {"preflight": {"required": True, "report": str(path), "min_games": 5}}
    with pytest.raises(RuntimeError, match="does not exist"):
        require_preflight(cfg)

    path.write_text(json.dumps({
        "passed": False, "games_completed": 5, "backend": "showdown_direct",
        "optimizer_steps": 0,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed invariants"):
        require_preflight(cfg)


def test_preflight_gate_accepts_five_game_no_optimizer_showdown_report(tmp_path):
    path = tmp_path / "latest.json"
    report = {
        "passed": True, "games_completed": 5, "backend": "showdown_direct",
        "optimizer_steps": 0,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    cfg = {"preflight": {"required": True, "report": str(path), "min_games": 5}}

    assert require_preflight(cfg) == report


def test_preflight_gate_rejects_stale_schema(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "passed": True,
        "games_completed": 5,
        "backend": "showdown_direct",
        "optimizer_steps": 0,
    }), encoding="utf-8")
    cfg = {
        "preflight": {
            "required": True,
            "report": str(path),
            "min_games": 5,
            "schema_version": 2,
        }
    }
    with pytest.raises(RuntimeError, match="schema_version=1"):
        require_preflight(cfg)


def test_preflight_gate_rejects_changed_source(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "source_fingerprint": "stale",
        "passed": True,
        "games_completed": 5,
        "backend": "showdown_direct",
        "optimizer_steps": 0,
    }), encoding="utf-8")
    cfg = {
        "preflight": {
            "required": True,
            "report": str(path),
            "min_games": 5,
            "schema_version": 2,
            "match_source": True,
        }
    }
    with pytest.raises(RuntimeError, match="source fingerprint"):
        require_preflight(cfg)


def test_preflight_gate_requires_public_replay_audit_when_configured(tmp_path):
    direct = tmp_path / "direct.json"
    replay = tmp_path / "replay.json"
    direct_report = {
        "passed": True,
        "games_completed": 5,
        "backend": "showdown_direct",
        "optimizer_steps": 0,
    }
    direct.write_text(json.dumps(direct_report), encoding="utf-8")
    cfg = {
        "preflight": {
            "required": True,
            "report": str(direct),
            "min_games": 5,
            "replay_report": str(replay),
            "min_parsed_replay_games": 5,
            "min_parsed_replay_dryruns": 10,
            "min_parsed_replay_rows": 100,
            "replay_schema_version": 1,
        }
    }
    with pytest.raises(RuntimeError, match="does not exist"):
        require_preflight(cfg)

    replay.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "backend": "public_showdown_replay",
                "games_completed": 5,
                "perspective_dryruns": 10,
                "decision_rows": 100,
                "optimizer_steps": 0,
            }
        ),
        encoding="utf-8",
    )
    assert require_preflight(cfg) == direct_report
