"""PBValueNet outcome trainer (fog/training/pb_value_train.py, PROPOSAL_IMPL_PLAN Phase G3):
dataset build from v2 shards (j*=-1 and truncated rows dropped, belief variants collected),
one-labeled-cell training runs end-to-end on a real log, and from_tables sizes fit real tokens."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

ATTACKER = """Hydreigon @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Dark Pulse
- Flash Cannon
- Fire Blast
"""

WEAK2 = """Great Tusk @ Heavy-Duty Boots
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


@pytest.fixture(scope="module")
def v2_shard(tmp_path_factory):
    from fog.env.engine_battle import EngineBattle
    from fog.models.agent import FogAgent
    from fog.search.pb_log import play_and_log_pb_games

    class StubOwn:
        def states(self, battle, role):
            t = battle.truth_tokens(role)
            return SimpleNamespace(tokens=np.stack([t, t]),
                                   weights=np.array([0.5, 0.5]), true_index=0, k=2)

    agent = FogAgent.from_tables({})
    proto = EngineBattle(ATTACKER, WEAK2, seed=0, battle_tag="pbvt-proto")
    opp_pool = np.stack([proto.truth_tokens("p2")])

    def factory(game_id, seed):
        return EngineBattle(ATTACKER, WEAK2, seed=seed, battle_tag=f"pbvt-{game_id}")

    out_dir = tmp_path_factory.mktemp("pbvt")
    play_and_log_pb_games(
        agent, factory, StubOwn(), opp_pool, n_games=3,
        out_path=str(out_dir / "shard_00.pt"), seed=5,
        max_decisions=20, search_kwargs={"iters": 16, "n_queried": 2})
    return out_dir, agent


def test_dataset_build_filters_and_belief_variants(v2_shard):
    import torch

    from fog.training.pb_value_train import build_pb_value_rows

    out_dir, _ = v2_shard
    rows = build_pb_value_rows(out_dir)
    assert rows, "the log yields trainable rows"
    for r in rows:
        assert r["own_tokens"].shape[0] > r["k_star"] >= 0
        assert r["opp_tokens"].shape[0] > r["j_star"] >= 0
        assert r["outcome"] in (-1.0, 0.0, 1.0)
        assert len(r["beliefs"]) >= 1, "root rho is always present"
        for b in r["beliefs"]:
            assert b.shape == (r["own_tokens"].shape[0], r["opp_tokens"].shape[0])
            assert b.sum() > 0
    # G1 coverage actually flowed through: some row carries more than the root belief
    assert any(len(r["beliefs"]) > 1 for r in rows)

    # j* = -1 rows are dropped
    shard = out_dir / "shard_00.pt"
    saved = torch.load(shard, weights_only=False)
    for row in saved["rows"]:
        row["opp_true_index"] = -1
    torch.save(saved, out_dir / "shard_01.pt")
    rows2 = build_pb_value_rows(out_dir)
    assert len(rows2) == len(rows), "the j*=-1 duplicate shard contributes nothing"
    (out_dir / "shard_01.pt").unlink()


def test_trainer_end_to_end_one_labeled_cell(v2_shard):
    from fog.models.pb_value import PBValueNet
    from fog.training.pb_value_train import build_pb_value_rows, train_pb_value

    out_dir, agent = v2_shard
    rows = build_pb_value_rows(out_dir)
    net = PBValueNet.from_tables()          # real-table sizes: real tokens must not hit <UNK>
    report = train_pb_value(agent, net, rows, epochs=2, batch_size=8,
                            holdout_frac=0.34, seed=0)
    assert report["n_rows"] == len(rows)
    assert np.isfinite(report["holdout_mse_after"])
    assert np.isfinite(report["train_loss_last"])
    assert report["belief_variants_mean"] >= 1.0
    assert report["holdout_mse_predict_mean_baseline"] >= 0.0
