from types import SimpleNamespace

import pytest
import torch

from fog.pretrained.native_worker import NativePlayer
from fog.pretrained.values import policy_value


def player_with_proposal():
    player = NativePlayer(SimpleNamespace(model=torch.nn.Linear(1, 1)))
    snapshot = {
        "hidden": torch.tensor([1.]), "space": {"history": [1]},
        "rl2": torch.zeros(1, 1, 14), "previous_state": {"turn": 0}, "time": 0,
        "generator": torch.Generator().manual_seed(7),
    }
    player.battles["root"] = dict(
        snapshot, battle={"public_events": ["turn 1"]}, choices={0: "move 1", 4: "switch 2"},
        pending_stage="proposed", pending_hidden=torch.tensor([2.]), pending_state={"turn": 1},
        decision_snapshot={key: value.clone() if isinstance(value, torch.Tensor) else value
                           for key, value in snapshot.items()},
    )
    return player


def test_counterfactual_siblings_isolate_history_cache_and_sampling_state():
    player = player_with_proposal()
    root = player.battles["root"]
    player.update({"op": "fork", "battle_tag": "a", "source_tag": "root", "choice": "move 1"})
    player.update({"op": "fork", "battle_tag": "b", "source_tag": "root", "choice": "switch 2"})
    a, b = player.battles["a"], player.battles["b"]
    assert a["time"] == b["time"] == 1 and root["time"] == 0
    assert a["rl2"][0, 0, 1] == 1 and b["rl2"][0, 0, 5] == 1
    assert a["rl2"].sum() == b["rl2"].sum() == 1 and root["rl2"].sum() == 0
    a["hidden"][0] = 99
    a["space"]["history"].append(2)
    a["battle"]["public_events"].append("only branch a")
    a["previous_state"]["turn"] = 99
    assert b["hidden"][0] == root["pending_hidden"][0] == 2 and root["hidden"][0] == 1
    assert b["space"]["history"] == root["space"]["history"] == [1]
    assert b["battle"]["public_events"] == root["battle"]["public_events"] == ["turn 1"]
    assert b["previous_state"]["turn"] == root["pending_state"]["turn"] == 1
    torch.rand(20, generator=a["generator"])
    assert torch.equal(b["generator"].get_state(), root["generator"].get_state())
    player.update({"op": "close", "battle_tag": "a"})
    assert set(player.battles) == {"root", "b"}


def test_execution_commit_records_the_override_and_rejects_double_execution():
    player = player_with_proposal()
    result = player.update({"op": "commit", "battle_tag": "root", "choice": "/choose switch 2"})
    record = player.battles["root"]
    assert result["native_action"] == 4 and record["time"] == 1
    assert record["rl2"][0, 0, 5] == 1 and record["hidden"][0] == 2
    with pytest.raises(ValueError, match="pending execution"):
        player.update({"op": "commit", "battle_tag": "root", "choice": "switch 2"})
    with pytest.raises(ValueError, match="uncommitted"):
        player.update({"op": "fork", "battle_tag": "a", "source_tag": "root", "choice": "move 1"})


def test_invalid_fork_does_not_publish_a_partial_record_or_change_the_parent():
    player = player_with_proposal()
    with pytest.raises(ValueError, match="does not match"):
        player.update({"op": "fork", "battle_tag": "bad", "source_tag": "root", "choice": "move 3"})
    assert list(player.battles) == ["root"] and player.battles["root"]["time"] == 0
    with pytest.raises(ValueError, match="already exists"):
        player.update({"op": "fork", "battle_tag": "root", "source_tag": "root", "choice": "move 1"})


def test_idle_opponent_fork_preserves_its_time_without_fabricating_a_choice():
    player = player_with_proposal()
    with pytest.raises(ValueError, match="cannot omit"):
        player.update({"op": "fork", "battle_tag": "idle", "source_tag": "root"})
    player.update({"op": "commit", "battle_tag": "root", "choice": "move 1"})
    response = player.update({"op": "fork", "battle_tag": "idle", "source_tag": "root"})
    assert response["native_action"] is None
    assert player.battles["idle"]["time"] == player.battles["root"]["time"] == 1
    assert player.battles["idle"]["hidden"].data_ptr() != player.battles["root"]["hidden"].data_ptr()


def test_counterfactual_api_rejects_hidden_labels_before_cloning():
    player = player_with_proposal()
    with pytest.raises(ValueError, match="Complete-team labels"):
        player.update({"op": "fork", "battle_tag": "bad", "source_tag": "root", "choice": "move 1",
                       "nested": [{"simulator_state": {"secret": "team"}}]})
    assert list(player.battles) == ["root"]


def test_rollback_restores_time_cache_and_removes_pending_branch_state():
    player = player_with_proposal()
    player.update({"op": "commit", "battle_tag": "root", "choice": "move 1"})
    player.update({"op": "rollback", "battle_tag": "root"})
    record = player.battles["root"]
    assert record["time"] == 0 and record["hidden"][0] == 1 and record["rl2"].sum() == 0
    assert "pending_stage" not in record and "pending_hidden" not in record


class KnownCritics:
    def __call__(self, latent, actions):
        # Different gamma and ensemble axes catch a swapped reduction; values
        # exceeding one demonstrate that these are raw returns, not outcomes.
        indices = (actions * torch.arange(13)).sum(-1)
        raw = 10 * indices[:, :, :, None, :] + torch.tensor([50., 90.])
        raw = raw + torch.tensor([-20., 20.])[None, None, None, :, None]
        return raw[..., None]

    def bin_dist_to_raw_vals(self, distribution):
        return distribution


def test_native_leaf_value_uses_legal_policy_last_gamma_and_raw_return_units():
    model = SimpleNamespace(gammas=torch.tensor([.5, .9]), critics=KnownCritics(), reward_multiplier=10.)
    probabilities = torch.zeros(2, 3, 13)
    probabilities[..., 0] = .25
    probabilities[..., 7] = .75
    readout = policy_value(model, torch.zeros(2, 3, 8), probabilities)
    torch.testing.assert_close(readout.value, torch.full((2, 3), 14.25))
    torch.testing.assert_close(readout.action_values[0, 0], 9 + torch.arange(13).float())
    torch.testing.assert_close(readout.critic_dispersion, torch.full((2, 3, 13), 2.))
    assert readout.gamma == pytest.approx(.9)
    assert readout.reward_multiplier == 10.


@pytest.mark.parametrize("invalid", [float("nan"), -1., .5])
def test_invalid_native_leaf_probabilities_fail_explicitly(invalid):
    model = SimpleNamespace(gammas=torch.tensor([.5, .9]), critics=KnownCritics(), reward_multiplier=10.)
    probabilities = torch.zeros(1, 1, 13)
    probabilities[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="normalized legal policy"):
        policy_value(model, torch.zeros(1, 1, 8), probabilities)


@pytest.mark.parametrize("invalid", [float("nan"), 0., -10.])
def test_invalid_reward_scale_cannot_silently_change_search_units(invalid):
    model = SimpleNamespace(gammas=torch.tensor([.5, .9]), critics=KnownCritics(), reward_multiplier=invalid)
    probabilities = torch.zeros(1, 1, 13)
    probabilities[0, 0, 0] = 1.
    with pytest.raises(ValueError, match="reward multiplier"):
        policy_value(model, torch.zeros(1, 1, 8), probabilities)
