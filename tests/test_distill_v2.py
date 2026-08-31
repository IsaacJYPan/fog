"""distill v2 reading (fog/training/distill.py, PROPOSAL_IMPL_PLAN Phase I.1): pb_log rows
convert to the v1 column contract, the action-conditioned own posterior backs the features,
cap-truncated games are dropped, and a real v2 shard distills end-to-end."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE


def _duck_obs(n_actions: int = 14):
    return SimpleNamespace(
        entities=np.zeros((3, 4), dtype=np.float32),
        entity_ids=np.zeros(3, dtype=np.int64),
        field=np.zeros(5, dtype=np.float32),
        history=np.zeros((2, 7), dtype=np.float32),
        active_idx=np.array([0, 6], dtype=np.int64),
        legality_mask=np.ones(n_actions, dtype=bool),
        opp_revealed={"tokens": np.full((6, 15), -1, dtype=np.int64), "masks": {}},
    )


def test_v2_row_columns_action_conditioned_posterior():
    """P(k | a) ∝ (Σ_j rho[k, j]) · σ1[k, a] — the two-sided analogue of v1's executed-label
    posterior; σ2 marginalizes out because its rows sum to 1. col = -1 falls back to the root
    margin (older logs)."""
    from fog.training.distill import _v2_row_columns

    row = {
        "obs": _duck_obs(),
        "rho": np.array([[0.3, 0.1], [0.4, 0.2]]),          # margins [0.4, 0.6]
        "sigma1": np.array([[0.5, 0.5], [0.25, 0.75]]),
        "action_label_col": 0,
        "true_index": 0,
        "action_dist": np.full(14, 1 / 14, dtype=np.float32),
        "action": 3, "turn": 2, "outcome": 1.0,
    }
    cols = _v2_row_columns(row)
    # post ∝ [0.4*0.5, 0.6*0.25] = [0.2, 0.15] -> [4/7, 3/7]; feature[1] = mass on true (4/7)
    assert cols["posterior_features"][1] == pytest.approx(4 / 7)
    assert cols["posterior_features"][2] == pytest.approx(4 / 7)          # top-1 mass
    assert cols["search_dist"].shape == (14,) and cols["outcome"] == 1.0

    row["action_label_col"] = -1
    cols = _v2_row_columns(row)
    assert cols["posterior_features"][1] == pytest.approx(0.4), "fallback: root margin"


@pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")
def test_v2_log_distills_end_to_end(tmp_path):
    """A real 2-game pb_log shard loads through load_v2_decisions (game re-keying, truncated
    drop) and runs a short distill() — format-blind training on the v2 arm's logs (I.1)."""
    import torch

    from fog.env.engine_battle import EngineBattle
    from fog.models.agent import FogAgent
    from fog.search.pb_log import play_and_log_pb_games
    from fog.training.distill import distill, load_v2_decisions

    attacker = """Hydreigon @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Dark Pulse
- Flash Cannon
- Fire Blast
"""
    weak2 = """Great Tusk @ Heavy-Duty Boots
Level: 5
Ability: Protosynthesis
EVs: 4 HP
Serious Nature
- Headlong Rush
- Rapid Spin
- Knock Off
- Ice Spinner

Kingambit @ Black Glasses
Level: 5
Ability: Supreme Overlord
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance
"""

    class StubOwn:
        def states(self, battle, role):
            t = battle.truth_tokens(role)
            return SimpleNamespace(tokens=np.stack([t, t]),
                                   weights=np.array([0.5, 0.5]), true_index=0, k=2)

    gen_agent = FogAgent.from_tables({})
    proto = EngineBattle(attacker, weak2, seed=0, battle_tag="dv2-proto")
    opp_pool = np.stack([proto.truth_tokens("p2")])

    def factory(game_id, seed):
        return EngineBattle(attacker, weak2, seed=seed, battle_tag=f"dv2-{game_id}")

    out = str(tmp_path / "shard_00.pt")
    rows = play_and_log_pb_games(
        gen_agent, factory, StubOwn(), opp_pool, n_games=3, out_path=out, seed=5,
        max_decisions=20, search_kwargs={"iters": 16, "n_queried": 2})
    assert all("action_label_col" in r for r in rows)

    # mark game 2's rows truncated in the saved shard: the loader must drop exactly those
    saved = torch.load(out, weights_only=False)
    n_g2 = sum(1 for r in saved["rows"] if r["game_id"] == 2)
    assert n_g2 > 0
    for r in saved["rows"]:
        if r["game_id"] == 2:
            r["truncated"] = True
    torch.save(saved, out)

    data = load_v2_decisions(tmp_path)
    assert data["outcome"].shape[0] == len(rows) - n_g2
    assert set(np.unique(data["game"])) == {0, 1}
    assert data["search_dist"].shape[1] == 14
    np.testing.assert_allclose(data["search_dist"].sum(axis=1), 1.0, rtol=1e-5)
    assert np.isfinite(data["posterior_features"]).all()

    train_agent = FogAgent.from_tables({"model": {"use_posterior_features": True}})
    report = distill(train_agent, data, epochs=2, batch_size=8, holdout_frac=0.5, seed=0)
    assert report["n_decisions"] == data["outcome"].shape[0]
    assert np.isfinite(report["holdout_after"]["kl"])
    assert "kl_by_turn" in report["holdout_after"]
