"""Training gate contracts for the mechanical flight recorder."""
from __future__ import annotations

import json

import pytest

from fog.training.preflight import require_preflight


def test_preflight_gate_is_opt_in():
    assert require_preflight({}) is None


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
