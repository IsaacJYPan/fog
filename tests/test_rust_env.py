"""Tests for the poke-engine adapter. Skipped when poke-engine isn't installed."""
from __future__ import annotations

from pathlib import Path

import pytest

from fog.env.rust_env import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

POOL = Path(__file__).resolve().parents[1] / "data" / "teams" / "pool_b"


def _paste(name: str) -> str:
    return (POOL / name).read_text(encoding="utf-8")


def test_paste_to_engine_side_has_tera():
    from fog.env.rust_env import paste_to_engine_side

    side = paste_to_engine_side(_paste("team1_balance.txt"))
    assert len(side.pokemon) == 6
    mon = side.pokemon[0]
    assert mon.id == "greattusk"
    assert str(mon.tera_type) != ""          # gen-9 feature present (load-bearing check)
    assert len(mon.moves) == 4


def test_proxy_encodes_through_shared_encoder():
    import poke_engine as pe

    from fog.encoding.observation import Observation
    from fog.env.rust_env import RustBattleProxy, paste_to_engine_side

    state = pe.State(side_one=paste_to_engine_side(_paste("team1_balance.txt")),
                     side_two=paste_to_engine_side(_paste("team2_offense.txt")))
    proxy = RustBattleProxy(state)
    obs = Observation.from_battle(proxy)
    obs.validate()
    assert obs.entities[:6].any() and obs.entities[6:].any()
    assert obs.legality_mask[:4].any() or obs.legality_mask[8:13].any()


def test_action_mapping_and_roundtrip():
    from fog.env.rust_env import smoke_roundtrip

    out = smoke_roundtrip(_paste("team1_balance.txt"), _paste("team2_offense.txt"))
    assert out["obs_ok"] and out["n_instruction_sets"] >= 1


def test_fuzz_trajectory_runs():
    from fog.scripts.fuzz_sim import diff_trajectories, engine_trajectory

    traj = engine_trajectory(_paste("team1_balance.txt"), _paste("team3_bulky.txt"),
                             seed=7, max_turns=20)
    assert len(traj) > 0
    # same seed -> identical trajectory (determinism is what makes the cross-sim diff meaningful)
    traj2 = engine_trajectory(_paste("team1_balance.txt"), _paste("team3_bulky.txt"),
                              seed=7, max_turns=20)
    assert diff_trajectories(traj, traj2) == {}
    # perturbations are detected and CLASSIFIED by cause
    import copy

    traj3 = copy.deepcopy(traj)
    pid, hp, status = traj3[0]["side"][0]
    traj3[0]["side"][0] = (pid, max(0.0, hp - 0.5), status)
    d = diff_trajectories(traj, traj3)
    assert d["cause"] == "hp" and d["turn"] == traj[0]["turn"] and d["mon"] == pid

    traj4 = copy.deepcopy(traj)
    traj4[0]["mask"] = tuple(list(traj4[0]["mask"])[:-1]) or (13,)
    assert diff_trajectories(traj, traj4)["cause"] == "mask"
    assert diff_trajectories(traj, traj[:-1])["cause"] == "length"
