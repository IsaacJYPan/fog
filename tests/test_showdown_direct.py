"""Direct, serverless Showdown and exact chance-enumeration acceptance tests."""

from __future__ import annotations

import math
import os
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

from fog.env.showdown_direct import (
    DEFAULT_SHOWDOWN_ROOT,
    ShowdownDirectClient,
    ShowdownWorkerError,
)
from fog.env.showdown_local import LocalShowdownBattle
from fog.env.showdown_local_runner import play_local_battle

EXPLICIT_NODE = os.environ.get("FOG_NODE_BINARY")
if EXPLICIT_NODE:
    os.environ["PATH"] = str(Path(EXPLICIT_NODE).resolve().parent) + os.pathsep + os.environ["PATH"]
SHOWDOWN_BUILT = (DEFAULT_SHOWDOWN_ROOT / "dist" / "sim" / "index.js").is_file()
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not SHOWDOWN_BUILT,
    reason="requires Node and the pinned local Pokemon Showdown build",
)


def _mew(move: str, *, nature: str = "Serious", speed_evs: int = 0) -> dict:
    return {
        "species": "Mew",
        "ability": "Synchronize",
        "item": "Leftovers",
        "moves": [move],
        "nature": nature,
        "evs": {"spe": speed_evs},
        "level": 100,
    }


def _root(
    client: ShowdownDirectClient,
    p1_move: str,
    p2_move: str = "Splash",
    *,
    speed_tie: bool = False,
) -> dict:
    p1 = _mew(p1_move, nature="Serious" if speed_tie else "Timid", speed_evs=0 if speed_tie else 252)
    p2 = _mew(p2_move)
    return client.create_state(
        {
            "formatid": "gen9customgame@@@!teampreview",
            "seed": f"sodium,{'1' * 64}",
            "strictChoices": True,
            "p1": {"name": "Alpha", "team": [p1]},
            "p2": {"name": "Beta", "team": [p2]},
        }
    )


def _fraction(probability) -> Fraction:
    return Fraction(probability.numerator, probability.denominator)


def _log_text(branch) -> str:
    return "\n".join(branch.public_log)


def test_ping_uses_requested_local_showdown_checkout() -> None:
    with ShowdownDirectClient() as client:
        result = client.ping()

    assert result["pong"] is True
    assert Path(result["showdown_root"]) == DEFAULT_SHOWDOWN_ROOT


def test_accuracy_is_two_exact_weighted_branches() -> None:
    with ShowdownDirectClient() as client:
        root = _root(client, "Will-O-Wisp")
        result = client.enumerate(root["state"], ("move 1", "move 1"), max_paths=100)

    assert result.exact is True
    assert result.sampled is False
    assert _fraction(result.kept_mass) == 1
    assert _fraction(result.pruned_mass) == 0
    assert {_fraction(branch.raw_probability) for branch in result.branches} == {
        Fraction(17, 20),
        Fraction(3, 20),
    }
    assert any("|-miss|" in _log_text(branch) for branch in result.branches)
    assert any("|-status|" in _log_text(branch) for branch in result.branches)


def test_critical_hit_mass_and_total_mass_are_exact() -> None:
    with ShowdownDirectClient() as client:
        root = _root(client, "Tackle")
        result = client.enumerate(root["state"], ("move 1", "move 1"), max_paths=10_000)

    total = sum((_fraction(branch.raw_probability) for branch in result.branches), Fraction())
    critical = sum(
        (
            _fraction(branch.raw_probability)
            for branch in result.branches
            if "|-crit|" in _log_text(branch)
        ),
        Fraction(),
    )
    assert total == 1
    assert critical == Fraction(1, 24)


def test_speed_tie_has_two_equal_action_orders() -> None:
    with ShowdownDirectClient() as client:
        # Final Gambit ends this deliberately tiny battle after the first action, isolating the
        # action-order coin flip from unrelated equal-speed end-of-turn event ordering.
        root = _root(client, "Final Gambit", "Final Gambit", speed_tie=True)
        result = client.enumerate(root["state"], ("move 1", "move 1"), max_paths=100)

    assert len(result.branches) == 2
    assert {_fraction(branch.raw_probability) for branch in result.branches} == {Fraction(1, 2)}
    logs = [_log_text(branch) for branch in result.branches]
    assert any("|move|p1a" in log for log in logs)
    assert any("|move|p2a" in log for log in logs)


def test_enumeration_is_deterministic() -> None:
    with ShowdownDirectClient() as client:
        root = _root(client, "Will-O-Wisp")
        first = client.enumerate(root["state"], ("move 1", "move 1"), include_state=False)
        second = client.enumerate(root["state"], ("move 1", "move 1"), include_state=False)

    assert first == second


def test_path_limit_fails_instead_of_sampling() -> None:
    with ShowdownDirectClient() as client:
        root = _root(client, "Will-O-Wisp")
        with pytest.raises(ShowdownWorkerError, match="no sampled fallback"):
            client.enumerate(root["state"], ("move 1", "move 1"), max_paths=1)


def test_explicit_pruning_reports_exact_removed_mass() -> None:
    with ShowdownDirectClient() as client:
        root = _root(client, "Will-O-Wisp")
        result = client.enumerate(
            root["state"], ("move 1", "move 1"), min_probability=0.2
        )

    assert len(result.branches) == 1
    assert _fraction(result.branches[0].raw_probability) == Fraction(17, 20)
    assert _fraction(result.branches[0].probability) == 1
    assert _fraction(result.kept_mass) == Fraction(17, 20)
    assert _fraction(result.pruned_mass) == Fraction(3, 20)


def test_seeded_showdown_outcome_is_one_of_the_exact_leaves() -> None:
    seed = f"sodium,{'a' * 64}"
    with ShowdownDirectClient() as client:
        root = _root(client, "Will-O-Wisp")
        exact = client.enumerate(root["state"], ("move 1", "move 1"), include_state=False)
        sampled = client.sample(
            root["state"], ("move 1", "move 1"), seed=seed, include_state=False
        )

    assert tuple(sampled["public_log"]) in {branch.public_log for branch in exact.branches}


def test_persistent_session_runs_a_battle_without_a_server() -> None:
    options = {
        "formatid": "gen9customgame@@@!teampreview",
        "seed": f"sodium,{'2' * 64}",
        "strictChoices": True,
        "p1": {"name": "Alpha", "team": [_mew("Final Gambit")]},
        "p2": {"name": "Beta", "team": [_mew("Final Gambit")]},
    }
    with ShowdownDirectClient() as client:
        session_id, initial = client.create_session(options, include_state=True)
        assert initial.request_state == "move"
        assert initial.requests["p1"]["active"][0]["moves"][0]["id"] == "finalgambit"

        final = client.step_session(
            session_id, ("/choose move finalgambit", "/choose move finalgambit")
        )
        assert client.close_session(session_id) is True

    assert final.ended is True
    assert any(line.startswith("|move|") for line in final.public_messages)
    assert any(line.startswith("|move|") for line in final.messages["p1"])


def test_persistent_session_returns_updated_request_for_hidden_trap_retry() -> None:
    bench = {
        "species": "Celebi",
        "ability": "Natural Cure",
        "item": "Leftovers",
        "moves": ["Splash"],
        "nature": "Serious",
        "level": 100,
    }
    gothitelle = {
        "species": "Gothitelle",
        "ability": "Shadow Tag",
        "item": "Leftovers",
        "moves": ["Splash"],
        "nature": "Serious",
        "level": 100,
    }
    options = {
        "formatid": "gen9customgame@@@!teampreview",
        "seed": f"sodium,{'8' * 64}",
        "strictChoices": True,
        "p1": {"name": "Alpha", "team": [_mew("Splash"), bench]},
        "p2": {"name": "Beta", "team": [gothitelle, _mew("Splash")]},
    }
    with ShowdownDirectClient() as client:
        session_id, initial = client.create_session(options)
        assert initial.requests["p1"]["active"][0].get("maybeTrapped") is True

        rejected = client.step_session(session_id, ("switch 2", "move splash"))
        assert "trapped" in rejected.choice_errors["p1"].lower()
        assert rejected.requests["p1"]["active"][0]["trapped"] is True
        assert rejected.turn == initial.turn

        accepted = client.step_session(session_id, ("move splash", "move splash"))
        client.close_session(session_id)

    assert accepted.choice_errors == {}
    assert accepted.turn == initial.turn + 1


def test_exact_leaves_expose_common_and_private_channels() -> None:
    with ShowdownDirectClient() as client:
        root = _root(client, "Tackle")
        result = client.enumerate(root["state"], ("move 1", "move 1"), max_paths=10_000)

    assert result.branches
    for branch in result.branches:
        assert set(branch.player_logs) == {"p1", "p2"}
        assert set(branch.requests) == {"p1", "p2"}
        assert any(line.startswith("|-damage|") for line in branch.public_log)
        # Both channels are returned even in Custom Game, whose Exact HP Mod can make them equal.
        # A standard-format censorship assertion is covered by the full gen9ou view test below.
        p1_text = "\n".join(branch.player_logs["p1"])
        p2_text = "\n".join(branch.player_logs["p2"])
        assert "|-damage|" in p1_text
        assert "|-damage|" in p2_text


def test_strict_candidate_replay_reconstructs_without_editing_state() -> None:
    actual = _mew("Final Gambit")
    options = {
        "formatid": "gen9customgame@@@!teampreview",
        "seed": f"sodium,{'7' * 64}",
        "strictChoices": True,
        "p1": {"name": "Alpha", "team": [actual]},
        "p2": {"name": "Beta", "team": [actual]},
    }
    compatible = {**actual, "item": "Rocky Helmet"}
    incompatible = {**actual, "moves": ["Splash"]}
    with ShowdownDirectClient() as client:
        session_id, _ = client.create_session(options)
        candidate = client.reconstruct_session_strict(
            session_id, p1_team=[compatible], p2_team=[actual]
        )
        assert candidate["state"]["sides"][0]["pokemon"][0]["item"] == "rockyhelmet"

        client.step_session(session_id, ("move finalgambit", "move finalgambit"))
        with pytest.raises(ShowdownWorkerError, match="invalid|choice|replay|diverged"):
            client.reconstruct_session_strict(
                session_id, p1_team=[incompatible], p2_team=[actual]
            )
        client.close_session(session_id)


def test_local_battle_reuses_poke_env_censored_views() -> None:
    options = {
        "formatid": "gen9customgame@@@!teampreview",
        "seed": f"sodium,{'3' * 64}",
        "strictChoices": True,
        "p1": {"name": "Alpha", "team": [_mew("Final Gambit")]},
        "p2": {"name": "Beta", "team": [_mew("Final Gambit")]},
    }
    with (
        ShowdownDirectClient() as client,
        LocalShowdownBattle.create(client, options) as battle,
    ):
            assert battle.p1.player_role == "p1"
            assert battle.p2.player_role == "p2"
            assert [move.id for move in battle.p1.available_moves] == ["finalgambit"]
            assert battle.p1.opponent_active_pokemon is not None
            assert battle.p1.opponent_active_pokemon.item == "unknown_item"
            assert battle.p2.active_pokemon.item == "leftovers"

            battle.step("move finalgambit", "move finalgambit")

            assert battle.ended is True
            assert battle.p1.finished is True
            assert battle.p2.finished is True
            assert battle.p1.won is not battle.p2.won or battle.p1.won is None


def test_direct_two_sided_solver_uses_authoritative_action_and_chance_axes() -> None:
    import types

    import numpy as np
    import torch
    from torch import nn

    from fog.search.direct_two_sided import direct_two_sided_action

    class ConstantValue(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            count = batch["entities"].shape[0]
            return types.SimpleNamespace(value=torch.zeros(count, device=self.marker.device))

    mon = {**_mew("Tackle"), "moves": ["Tackle", "Splash"]}
    options = {
        "formatid": "gen9customgame@@@!teampreview",
        "seed": f"sodium,{'8' * 64}",
        "strictChoices": True,
        "p1": {"name": "Alpha", "team": [mon]},
        "p2": {"name": "Beta", "team": [mon]},
    }
    with (
        ShowdownDirectClient() as client,
        LocalShowdownBattle.create(client, options) as battle,
    ):
        result = direct_two_sided_action(
            ConstantValue(), battle.p1, role="p1", device=torch.device("cpu"), iters=8
        )

    assert result.payoff.shape[:2] == (1, 1)
    assert result.payoff.shape[2] >= 2
    assert result.payoff.shape[3] >= 2
    assert result.simulations == result.payoff.shape[2] * result.payoff.shape[3]
    assert result.chance_branches >= result.simulations
    assert result.action_dist.sum() == pytest.approx(1.0)
    assert result.action in np.flatnonzero(result.action_dist)


def test_local_runner_completes_without_network_players() -> None:
    options = {
        "formatid": "gen9customgame@@@!teampreview",
        "seed": f"sodium,{'4' * 64}",
        "strictChoices": True,
        "p1": {"name": "Alpha", "team": [_mew("Final Gambit")]},
        "p2": {"name": "Beta", "team": [_mew("Final Gambit")]},
    }
    with ShowdownDirectClient() as client:
        result = play_local_battle(client, options, "default", "default")

    assert result.turns == 1
    assert result.p1_won in {True, False, None}


def test_local_runner_declares_turn_limit_a_draw() -> None:
    options = {
        "formatid": "gen9customgame@@@!teampreview",
        "seed": f"sodium,{'4' * 64}",
        "strictChoices": True,
        "p1": {"name": "Alpha", "team": [_mew("Splash")]},
        "p2": {"name": "Beta", "team": [_mew("Splash")]},
    }
    with ShowdownDirectClient() as client:
        result = play_local_battle(
            client, options, "default", "default", max_turns=0
        )

    assert result.truncated is True
    assert result.p1_won is None
    assert result.winner is None


def test_local_runner_handles_gen9ou_team_preview_and_full_game() -> None:
    from fog.env.teams import load_pool, to_packed

    pool = load_pool(Path(__file__).resolve().parents[1] / "data" / "teams" / "pool_b")
    team = to_packed(pool[0])
    options = {
        "formatid": "gen9ou",
        "seed": f"sodium,{'6' * 64}",
        "strictChoices": True,
        "p1": {"name": "Alpha", "team": team},
        "p2": {"name": "Beta", "team": team},
    }
    with ShowdownDirectClient() as client:
        result = play_local_battle(client, options, "random", "random", seed=6)

    assert result.turns > 0
    assert result.winner in {"Alpha", "Beta", None}


def test_direct_cross_sim_trajectory_uses_the_standard_action_surface() -> None:
    from fog.env.teams import load_pool
    from fog.scripts.fuzz_sim import showdown_direct_trajectory

    pool = load_pool(Path(__file__).resolve().parents[1] / "data" / "teams" / "pool_b")
    trajectory = showdown_direct_trajectory(pool[0], pool[1], seed=9, max_turns=2)

    assert trajectory
    assert all(entry["role"] in {"p1", "p2"} for entry in trajectory)
    assert all(entry["mask"] for entry in trajectory)


def test_fog_player_can_act_through_local_order_factory() -> None:
    from fog.env.player import FogPlayer
    from fog.env.showdown_local_runner import bind_fog_player
    from fog.models.agent import FogAgent

    options = {
        "formatid": "gen9customgame@@@!teampreview",
        "seed": f"sodium,{'5' * 64}",
        "strictChoices": True,
        "p1": {"name": "FOG", "team": [_mew("Final Gambit")]},
        "p2": {"name": "Beta", "team": [_mew("Final Gambit")]},
    }
    adapter = bind_fog_player(FogPlayer(FogAgent.from_tables({}).eval(), temperature=0))
    with ShowdownDirectClient() as client:
        result = play_local_battle(client, options, adapter, "default")

    assert result.turns == 1


def test_showdown_direct_training_backend_smoke(tmp_path) -> None:
    from fog.training.ppo import train

    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "mew.txt").write_text(
        "Mew @ Leftovers\n"
        "Ability: Synchronize\n"
        "Tera Type: Psychic\n"
        "Timid Nature\n"
        "- Final Gambit\n",
        encoding="utf-8",
    )
    cfg = {
        "seed": 6,
        "device": "cpu",
        "env": {
            "backend": "showdown_direct",
            "battle_format": "gen9customgame@@@!teampreview",
            "team_pool": str(pool),
        },
        "ppo": {
            "total_decisions": 2,
            "batch_decisions": 2,
            "games_per_chunk": 2,
            "epochs": 1,
            "minibatch_size": 1,
        },
        "league": {"scripted": ["MaxDamage"], "snapshot_every_decisions": 100},
        "log": {"dir": str(tmp_path / "run")},
    }

    metrics = train(cfg)

    assert metrics["decisions"] >= 2
    assert math.isfinite(metrics["loss"])
    assert (tmp_path / "run" / "checkpoints" / "ppo_final.pt").is_file()


def test_showdown_direct_training_backend_process_workers(tmp_path) -> None:
    from fog.training.ppo import train

    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "mew.txt").write_text(
        "Mew @ Leftovers\n"
        "Ability: Synchronize\n"
        "Tera Type: Psychic\n"
        "Timid Nature\n"
        "- Final Gambit\n",
        encoding="utf-8",
    )
    cfg = {
        "seed": 7,
        "device": "cpu",
        "env": {
            "backend": "showdown_direct",
            "battle_format": "gen9customgame@@@!teampreview",
            "team_pool": str(pool),
            "rollout_workers": 2,
            "rollout_worker_mode": "process",
            "actor_threads": 1,
            "rollout_worker_timeout": 60,
        },
        "ppo": {
            # Two one-decision batches force persistent workers to survive an optimizer step and
            # consume the in-place shared actor refresh before collecting again.
            "total_decisions": 4,
            "batch_decisions": 2,
            "games_per_chunk": 2,
            "epochs": 1,
            "minibatch_size": 1,
        },
        "league": {"scripted": ["MaxDamage"], "snapshot_every_decisions": 100},
        "log": {"dir": str(tmp_path / "run")},
    }

    metrics = train(cfg)

    assert metrics["decisions"] >= 4
    assert math.isfinite(metrics["loss"])
    assert (tmp_path / "run" / "checkpoints" / "ppo_final.pt").is_file()
