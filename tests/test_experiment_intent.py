"""Experiment-INTENT tests (audit 2026-07-05): each asserts that a milestone's mechanism actually
fires, not just that components satisfy their contracts. Every test here is a regression test for a
defect that shipped with a green suite:

  * POOL-A belief supervision silently disabled (species-multiset truth match always ambiguous)
  * M2 set-oracle a training no-op (truth never in the live batch or the frags)
  * watchdog false-halting fresh L-runs (baselines from the control's FINAL tail)
  * zero-mirror / skipped leakage measurements fabricating ΔI spikes
  * clipped-ΔI estimator with no telescoping control
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

DATA = Path(__file__).resolve().parents[1] / "data" / "teams"


def _pool(name: str) -> list[str]:
    from fog.env.teams import load_pool

    return load_pool(DATA / name)


# --------------------------------------------------------------------- truth plumbing (M1.5 intent)
def test_recording_teambuilder_tracks_yielded_paste():
    from fog.env.teams import make_pool_teambuilder

    teams = _pool("pool_a")
    tb = make_pool_teambuilder(teams, seed=3)
    packed = tb.yield_team()
    assert tb.last_paste in teams
    assert tb.last_index is not None and tb._packed[tb.last_index] == packed
    tb.yield_team()
    assert tb.history and len(tb.history) == 2


def test_pool_truth_species_match_is_ambiguous_on_pool_a():
    """The original defect: all POOL-A teams share the same 6 species, so species-multiset matching
    returns None for EVERY battle — belief supervision must not depend on it."""
    from fog.constants.tables import TABLES
    from fog.env.team_truth import PoolTruth

    pt = PoolTruth(_pool("pool_a"), TABLES)
    all_species = pt._truth[0][:, 0].tolist()
    assert pt.match(all_species) is None  # fully revealed species still ambiguous by design


def test_pool_truth_match_views_disambiguates_pool_a():
    from fog.constants.tables import TABLES
    from fog.encoding.reveals import SlotView
    from fog.env.team_truth import PoolTruth

    pt = PoolTruth(_pool("pool_a"), TABLES)
    truth = pt._truth[0]
    views = []
    for row in truth:
        views.append(
            SlotView(
                species=int(row[0]),
                item=int(row[1]) if row[1] >= 0 else None,
                ability=int(row[2]) if row[2] >= 0 else None,
                moves=[int(m) for m in row[3:7] if m >= 0],
                tera=int(row[8]) if row[8] >= 0 else None,
            )
        )
    got = pt.match_views(views)
    assert got is not None and np.array_equal(got, truth)


def test_pool_truth_match_views_respects_no_item():
    """<NO_ITEM> after Knock Off must not disqualify the true team (sheet keeps the original item)."""
    from fog.constants.schema import NO_ITEM
    from fog.constants.tables import TABLES
    from fog.encoding.reveals import SlotView
    from fog.env.team_truth import PoolTruth

    pt = PoolTruth(_pool("pool_a"), TABLES)
    truth = pt._truth[2]
    no_item = TABLES.item_index(NO_ITEM)
    views = [
        SlotView(species=int(row[0]), item=no_item,
                 moves=[int(m) for m in row[3:7] if m >= 0])
        for row in truth
    ]
    got = pt.match_views(views)
    # moves still disambiguate; the knocked-off item must not exclude the true team
    assert got is not None and np.array_equal(got, truth)


def test_hidden_info_oxygen_check():
    """POOL-A (shared species) conceals; pool_b (unique species multisets) does not — the leakage
    trainer must refuse the latter."""
    from fog.constants.tables import TABLES
    from fog.env.team_truth import PoolTruth

    assert PoolTruth(_pool("pool_a"), TABLES).hidden_info_warning() is None
    assert PoolTruth(_pool("pool_b"), TABLES).hidden_info_warning() is not None


# ------------------------------------------------------------------------- M2 set-oracle intent
class _Mon:
    def __init__(self):
        self.species = "garchomp"
        self.item = "leftovers"
        self.ability = "roughskin"
        self.moves = {"earthquake": object(), "dragonclaw": object()}
        self.tera_type = None
        self.base_stats = {"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102}
        self.types = ["dragon", "ground"]
        self.weight = 95.0
        self.current_hp_fraction = 1.0
        self.fainted = False
        self.status = None
        self.status_counter = 0
        self.boosts = {}
        self.effects = {}
        self.terastallized = False


class _Battle:
    def __init__(self, tag="oracle-battle"):
        self.battle_tag = tag
        self.player_role = "p1"
        self.turn = 1
        self.observations = {}
        self.team = {f"p1: Mon{i}": _Mon() for i in range(6)}
        self.opponent_team = {}
        self.teampreview_opponent_team = []
        self.active_pokemon = None
        self.opponent_active_pokemon = None
        self.weather = {}
        self.fields = {}
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.force_switch = False
        self.won = True


class _FakePlayer:
    def create_order(self, target, **kwargs):
        return ("order", target, kwargs)

    def choose_default_move(self):
        return ("default",)


def test_oracle_truth_reaches_frag_and_update_forward():
    """The M2 defect: acting with truth but updating without (or with no truth at all) silently
    trains an N-agent. The frag must carry opp_truth_tokens so ppo_update's forward sees exactly
    what the acting forward saw; -1 rows must be present when the provider had no answer."""
    from fog.env.player import FogPlayer
    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer

    torch.manual_seed(0)
    agent = FogAgent.from_tables({"model": {"oracle": "set"}})
    truth = np.full((6, 9), -1, dtype=np.int64)
    truth[:, 0] = np.arange(6)

    p = FogPlayer(agent, temperature=0.0)
    p._player_ref = _FakePlayer()
    buf = RolloutBuffer()
    p.attach_buffer(buf)
    p.record_mirror = False  # not under test here
    p.oracle_truth_provider = lambda b: truth
    p.act(_Battle("oracle-a"))
    p.oracle_truth_provider = lambda b: None  # provider has no answer -> -1 fill, same structure
    p.act(_Battle("oracle-a"))

    frags = buf._games[p._game_ids["oracle-a"]].frags
    assert np.array_equal(frags[0]["opp_truth_tokens"], truth)
    assert (frags[1]["opp_truth_tokens"] == -1).all()

    for gid in list(p._game_ids.values()):
        buf.end_game(gid, outcome=1.0)
    batch = buf.to_batch()
    assert batch["opp_truth_tokens"].shape == (2, 6, 9)
    out = agent(batch)  # the update-time forward consumes the truth without special-casing
    assert out.belief is not None


def test_apply_set_oracle_merges_unknown_truth_back_to_revealed():
    from fog.models.oracle import apply_set_oracle

    revealed = {"tokens": torch.arange(2 * 54, dtype=torch.long).reshape(2, 6, 9) % 7, "masks": {}}
    truth = torch.full((2, 6, 9), -1, dtype=torch.long)
    truth[0, 0, 0] = 5
    out = apply_set_oracle(revealed, truth)
    assert out["tokens"][0, 0, 0] == 5                       # truth wins where known
    assert (out["tokens"][1] == revealed["tokens"][1]).all() # no truth -> exactly the revealed input


def test_train_raises_on_action_oracle_and_unpooled_set_oracle():
    from fog.training.ppo import train

    with pytest.raises(NotImplementedError):
        train({"model": {"oracle": "action"}, "env": {"battle_format": "gen9randombattle"}})
    with pytest.raises(ValueError):
        train({"model": {"oracle": "set"}, "env": {"battle_format": "gen9randombattle"}})


# ------------------------------------------------------------------------ watchdog baseline intent
def test_watchdog_baselines_are_progress_matched(tmp_path):
    """The false-halt defect: a fresh L-run's first update (low win-rate, high entropy) satisfied
    both diversity-spike conditions against the control's FINAL tail."""
    from fog.training.leakage import DegenerateSolutionWatchdog

    csv = tmp_path / "metrics.csv"
    rows = ["step,win_rate,move_entropy,avg_game_length"]
    control = [
        (100, 0.30, 2.0), (200, 0.40, 1.8), (3000, 0.60, 1.2), (4000, 0.62, 1.15),
        (5000, 0.65, 1.1), (6000, 0.70, 1.0), (7000, 0.70, 1.0), (8000, 0.71, 0.95),
    ]
    for step, wr, ent in control:
        rows.append(f"{step},{wr},{ent},30.0")
    csv.write_text("\n".join(rows), encoding="utf-8")

    wd = DegenerateSolutionWatchdog.from_control_run(tmp_path)
    early_metrics = {"win_rate": 0.32, "move_entropy": 2.05, "avg_game_length": 31.0}

    # legacy behavior (final tail) would flag a healthy first update
    assert wd.check_all(early_metrics)["move_diversity_spike"] is True
    # progress-matched baselines do not
    wd.update_to(150)
    assert wd.check_all(early_metrics)["move_diversity_spike"] is False
    # and a genuine late crash+spike still fires at matched progress
    wd.update_to(8000)
    assert wd.check_all({"win_rate": 0.45, "move_entropy": 1.6, "avg_game_length": 31.0})[
        "move_diversity_spike"
    ] is True


# ------------------------------------------------------------------- leakage estimator intent (§7)
def _leak_frag():
    return {"obs": np.zeros(2, dtype=np.float32)}


def test_leak_ce_gaps_fill_without_fabricating_delta():
    """A skipped measurement (None) treated as CE=0 fabricated a full-CE reveal spike."""
    from fog.training.rollout import RolloutBuffer

    buf = RolloutBuffer(gamma=1.0, lam=1.0, beta=1.0, clip_deltas=True)
    g = buf.start_game()
    for ce in (2.0, None, 1.5):
        buf.add_decision(g, _leak_frag(), action=0, log_prob=-1.0, value=0.0, leak_ce=ce)
    buf.end_game(g, outcome=0.0, final_leak_ce=1.0)
    batch = buf.to_batch()
    # filled ce = [2.0, 2.0, 1.5]; deltas = [0, 0.5, 0.5]; rewards = -deltas (beta=1, outcome 0)
    # gamma=lam=1, values=0 -> returns[t] = sum of future rewards
    assert np.allclose(batch["returns"].numpy(), [-1.0, -1.0, -0.5])


def test_unclipped_deltas_telescope_to_net_ce_drop():
    from fog.training.rollout import RolloutBuffer

    ces = [3.0, 3.4, 2.1]          # non-monotone: a clipped estimator would overcharge
    final = 1.5
    buf = RolloutBuffer(gamma=1.0, lam=1.0, beta=1.0, clip_deltas=False)
    g = buf.start_game()
    for ce in ces:
        buf.add_decision(g, _leak_frag(), action=0, log_prob=-1.0, value=0.0, leak_ce=ce)
    buf.end_game(g, outcome=0.0, final_leak_ce=final)
    batch = buf.to_batch()
    # total penalty = -(sum of signed deltas) = -(ce_first - ce_final): the telescoping property
    assert np.isclose(float(batch["returns"][0]), -(ces[0] - final))

    buf2 = RolloutBuffer(gamma=1.0, lam=1.0, beta=1.0, clip_deltas=True)
    g2 = buf2.start_game()
    for ce in ces:
        buf2.add_decision(g2, _leak_frag(), action=0, log_prob=-1.0, value=0.0, leak_ce=ce)
    buf2.end_game(g2, outcome=0.0, final_leak_ce=final)
    clipped_total = -float(buf2.to_batch()["returns"][0])
    assert clipped_total > (ces[0] - final)  # the clipped form overcharges on noisy CE


def test_zero_mirror_yields_no_measurement():
    """Structure-preserving zero mirror (all labels -1) must produce leak_ce=None, not CE=0."""
    from types import SimpleNamespace

    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
    from fog.env.player import FogPlayer, _zero_mirror

    obs = SimpleNamespace(
        entities=np.zeros((12, ENTITY_RAW_DIM), dtype=np.float32),
        entity_ids=np.zeros((12, N_ID_FIELDS), dtype=np.int64),
        field=np.zeros(DIMS.field_dim, dtype=np.float32),
        history=np.zeros((1, DIMS.turn_dim), dtype=np.float32),
    )
    p = FogPlayer(agent=None)
    p.phi_hat = object()  # non-None sentinel; must never be reached
    assert p._leakage_ce(_zero_mirror(obs), obs, device=None) is None


# ----------------------------------------------------------------------------- paired eval intent
def test_paired_teambuilders_swap_sides_each_pair():
    from fog.env.teams import make_paired_teambuilders

    teams = _pool("pool_a")
    tb_a, tb_b = make_paired_teambuilders(teams, n_games=6, seed=9)
    a_idx, b_idx = [], []
    for _ in range(6):
        tb_a.yield_team()
        tb_b.yield_team()
        a_idx.append(tb_a.last_index)
        b_idx.append(tb_b.last_index)
    for k in range(0, 6, 2):
        assert a_idx[k] == b_idx[k + 1] and a_idx[k + 1] == b_idx[k]
