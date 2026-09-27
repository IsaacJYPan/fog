from __future__ import annotations

from pathlib import Path

import pytest

from fog.config import load_config
from fog.models.legacy_native_teacher import LegacyNativeTeacher

LEGACY_CHECKPOINT = Path("runs/set_ar_native_evs_200k/checkpoints/ppo_final.pt")


@pytest.mark.skipif(
    not LEGACY_CHECKPOINT.is_file(),
    reason="requires the historical native-EV teacher checkpoint (not distributed with the repo)",
)
def test_legacy_teacher_loads_native_checkpoint_without_missing_tensors():
    cfg = load_config("configs/set_ar_train.yaml")
    teacher = LegacyNativeTeacher.from_checkpoint(cfg, str(LEGACY_CHECKPOINT))
    assert teacher.backbone.hist.hidden_size == 256
    assert teacher.policy_value.policy[0].in_features == 516


def test_policy_distillation_tap_is_opt_in():
    from fog.env.player import FogPlayer

    player = FogPlayer()
    assert player.decision_observer is None
    player.decision_observer = lambda *_: None
    assert callable(player.decision_observer)
