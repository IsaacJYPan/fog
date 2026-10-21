from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from fog.train import TrainerKind, TrainingSession, main


@pytest.mark.parametrize(
    "config,kind,backend",
    [
        ({"ppo": {}}, None, "ppo"),
        ({"bc": {}}, None, "bc"),
        ({"trainer": "bc", "ppo": {}, "bc": {}}, None, "bc"),
        ({"trainer": "ppo", "ppo": {}, "bc": {}}, "bc", "bc"),
    ],
)
def test_training_session_dispatches_selected_backend(monkeypatch, config, kind, backend):
    seen = []
    result = {"step": 12}

    def train(config):
        seen.append(config)
        return result

    monkeypatch.setitem(sys.modules, f"fog.training.{backend}", SimpleNamespace(train=train))

    assert TrainingSession(config, kind=kind).run() is result
    assert seen == [config]


@pytest.mark.parametrize(
    "config,message",
    [({}, "cannot infer trainer"), ({"ppo": {}, "bc": {}}, "both ppo and bc")],
)
def test_training_session_requires_explicit_choice(config, message):
    session = TrainingSession(config)
    with pytest.raises(ValueError, match=message):
        session.resolved_kind()

    session.kind = TrainerKind.BEHAVIOR_CLONING
    assert session.resolved_kind() is TrainerKind.BEHAVIOR_CLONING


def test_training_session_loads_overrides(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("trainer: ppo\nppo:\n  lr: 0.001\n", encoding="utf-8")

    session = TrainingSession.from_file(config, overrides=["ppo.lr=0.0002"])

    assert session.source == config
    assert session.resolved_kind() is TrainerKind.PPO
    assert session.config["ppo"]["lr"] == pytest.approx(0.0002)


def test_cli_passes_trainer_and_overrides(monkeypatch, tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("trainer: ppo\nppo:\n  lr: 0.001\nbc: {}\n", encoding="utf-8")
    sessions = []
    monkeypatch.setattr(TrainingSession, "run", lambda session: sessions.append(session))
    argv = ["fog-train-run", "--config", str(config), "--trainer", "bc", "--set", "ppo.lr=0.0002"]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert len(sessions) == 1
    assert sessions[0].resolved_kind() is TrainerKind.BEHAVIOR_CLONING
    assert sessions[0].config["ppo"]["lr"] == pytest.approx(0.0002)
