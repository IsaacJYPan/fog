from __future__ import annotations

import pytest

from fog.train import BehaviorCloningTrainer, PPOTrainer, TrainerKind, TrainingSession


def test_training_session_infers_backend_from_config_section():
    assert isinstance(TrainingSession({"ppo": {}}).build_trainer(), PPOTrainer)
    assert isinstance(TrainingSession({"bc": {}}).build_trainer(), BehaviorCloningTrainer)


def test_training_session_requires_explicit_choice_for_ambiguous_config():
    session = TrainingSession({"ppo": {}, "bc": {}})
    with pytest.raises(ValueError, match="both ppo and bc"):
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
