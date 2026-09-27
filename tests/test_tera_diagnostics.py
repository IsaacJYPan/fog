from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fog.env.player import FogPlayer, _sample_action
from fog.eval.action_diagnostics import new_action_stats, record_action


@pytest.mark.parametrize("bonus, expected", [(0., 0), (.25, 0), (1.5, 4)])
def test_fog_player_actual_greedy_path_applies_legal_tera_bonus(monkeypatch, bonus, expected):
    from fog.encoding.observation import Observation

    scores = torch.full((1, 14), -torch.inf)
    scores[0, 0], scores[0, 4] = 0., -1.

    class Agent(torch.nn.Module):
        def forward(self, batch):
            return SimpleNamespace(log_probs=scores.log_softmax(-1))

    obs = SimpleNamespace(validate=lambda: None, active_idx=np.array([0, 6]),
                          legality_mask=torch.isfinite(scores[0]).numpy())
    player = FogPlayer(Agent(), temperature=0, tera_logit_bonus=bonus)
    player._player_ref = object()
    monkeypatch.setattr(player, "_battle_state", lambda b: ("g", None,
        SimpleNamespace(record_private_team=lambda b: None), set()))
    monkeypatch.setattr(player, "_feed_events", lambda *args: None)
    monkeypatch.setattr(Observation, "from_battle", lambda *a, **kw: obs)
    monkeypatch.setattr("fog.encoding.batch.observation_to_batch", lambda *a: {})
    monkeypatch.setattr("fog.encoding.events.record_history_decision", lambda *a: None)
    monkeypatch.setattr("fog.env.player.map_action_to_order", lambda b, a, p: a)
    assert player.act(SimpleNamespace()) == expected
    scores[0, 4] = -torch.inf
    assert player.act(SimpleNamespace()) == 0
    assert _sample_action(scores[0], temperature=0, tera_logit_bonus=100) == 0


def test_action_stats_record_game_level_tera_once_and_raw_margins():
    scores = torch.full((14,), -torch.inf)
    scores[0], scores[4] = 0, -1
    legal = torch.isfinite(scores).numpy()
    stats = new_action_stats()
    battle = SimpleNamespace(battle_tag="one", turn=4)
    record_action(stats, battle, legal, scores.log_softmax(-1), 4, policy_diagnostics=True)
    battle.turn = 5
    record_action(stats, battle, legal, scores.log_softmax(-1), 0)
    assert stats["games_with_tera"] == 1
    assert stats["first_tera_turns"] == [4]
    assert stats["raw_tera_diagnostics"]["paired_margins"][0] == [-1]
    assert stats["raw_tera_diagnostics"]["tera_probability_t1"][0] == pytest.approx(1/(1+np.e))
