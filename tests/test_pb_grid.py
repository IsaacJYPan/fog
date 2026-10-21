"""Public-belief chance grids: branch probabilities, action axes, availability, and censored leaves.

Duplicate candidates must produce identical branch tables. One-sided decisions,
seed independence, and value assembly are checked on engine fixtures.
"""
from __future__ import annotations

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

DEFENDER = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave
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


def _battle(seed: int = 0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(ATTACKER, DEFENDER, seed=seed, battle_tag=f"pbgrid-test-{seed}")


def _truth_grid(battle, k_dup: int, j_dup: int):
    own = np.stack([battle.truth_tokens("p1")] * k_dup)
    opp = np.stack([battle.truth_tokens("p2")] * j_dup)
    return own, opp


@pytest.fixture(scope="module")
def grid_2x2():
    from fog.search.pb_grid import build_pb_grid

    b = _battle()
    own, opp = _truth_grid(b, 2, 2)
    return build_pb_grid(b, "p1", own, opp, seed=11)


def test_shapes_labels_and_budget_accounting(grid_2x2):
    from fog.search.pb_grid import public_label

    g = grid_2x2
    n_k, n_a = g.avail1.shape
    n_j, n_b = g.avail2.shape
    assert (n_k, n_j) == (2, 2)
    assert g.branch_prob.shape[:4] == (2, 2, n_a, n_b)
    # every label round-trips through the local action map
    b = _battle()
    for k in range(n_k):
        for a in np.flatnonzero(g.avail1[k]):
            assert public_label(b, "p1", int(g.local_action1[k, a])) == g.labels1[a]
    for j in range(n_j):
        for bcol in np.flatnonzero(g.avail2[j]):
            assert public_label(b, "p2", int(g.local_action2[j, bcol])) == g.labels2[bcol]
    # budget: one commit per available (k, j, a, b)
    want = sum(int(g.avail1[k].sum()) * int(g.avail2[j].sum())
               for k in range(n_k) for j in range(n_j))
    assert g.sims_run == want
    assert not g.failed.any()


def test_branch_probabilities_sum_to_one_on_available_cells(grid_2x2):
    g = grid_2x2
    cell_mass = g.branch_prob.sum(axis=-1)                          # [K, J, A, B]
    available = (g.avail1[:, None, :, None] & g.avail2[None, :, None, :]) & ~g.failed
    np.testing.assert_allclose(cell_mass[available], 1.0, atol=1e-9)
    np.testing.assert_allclose(cell_mass[~available], 0.0)
    assert (g.pruned_mass[available] <= 0.2 + 1e-12).all()
    # exactly one of terminal / leaf per existing branch
    exists = g.branch_prob > 0
    is_terminal = ~np.isnan(g.terminal_value)
    has_leaf = g.leaf_index >= 0
    assert ((is_terminal ^ has_leaf) == exists).all()


def test_identical_candidate_worlds_produce_identical_tables(grid_2x2):
    """Both own-worlds and both opponent-worlds are the same team here, so every derived table
    must be invariant across them: chance enumeration is deterministic given the state, and the
    fresh per-fork RNG seeds must not leak into branch structure."""
    g = grid_2x2
    np.testing.assert_array_equal(g.avail1[0], g.avail1[1])
    np.testing.assert_array_equal(g.avail2[0], g.avail2[1])
    np.testing.assert_allclose(g.branch_prob[0], g.branch_prob[1])
    np.testing.assert_allclose(g.branch_prob[:, 0], g.branch_prob[:, 1])
    np.testing.assert_array_equal(np.isnan(g.terminal_value[0]), np.isnan(g.terminal_value[1]))


def test_leaves_are_censored_observations(grid_2x2):
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import L

    g = grid_2x2
    assert len(g.leaf_obs) > 0
    assert len(g.leaf_battles) == len(g.leaf_obs)
    assert max(int(g.leaf_index.max()), -1) == len(g.leaf_obs) - 1
    obs = g.leaf_obs[0]
    tokens = np.asarray(obs.opp_revealed["tokens"])
    assert (tokens == -1).any(), \
        "a one-turn leaf must still hide most opponent attributes (fog semantics, 19.12)"
    own_known = obs.entities[:6, L.epistemic:L.epistemic + DIMS.epistemic_flags]
    assert not own_known.all(), "the value leaf must not retain the acting side's private sheet"
    assert obs.legality_mask.all(), "private legality is replaced by a constant public placeholder"
    assert g.leaf_evidence.shape == (len(g.leaf_obs), 2, 2)
    for k, j, a, b, m in np.argwhere(g.leaf_index >= 0):
        leaf = g.leaf_index[k, j, a, b, m]
        assert g.leaf_evidence[leaf, k, j], "a leaf must retain the pair that generated it"


def test_value_assembly_contract(grid_2x2):
    """u-tensor assembly: branch-weighted mix of terminal payoffs and stubbed leaf
    values, no NaN on available cells, bounded by the payoff range."""
    g = grid_2x2
    v_leaf = np.full(len(g.leaf_obs), 0.5)
    per_branch = np.where(g.leaf_index >= 0, v_leaf[np.maximum(g.leaf_index, 0)],
                          np.where(np.isnan(g.terminal_value), 0.0, g.terminal_value))
    u = (g.branch_prob * per_branch).sum(axis=-1)
    available = (g.avail1[:, None, :, None] & g.avail2[None, :, None, :]) & ~g.failed
    assert np.isfinite(u[available]).all()
    assert (np.abs(u[available]) <= 1.0 + 1e-9).all()


def test_grid_rejects_decisions_we_do_not_own():
    """A one-sided state the OPPONENT owns (only p2 pending) is not ours to solve — F2 accepts our
    own one-sided decisions but the symmetric opponent-only case stays an error."""
    from fog.env.engine_battle import EngineBattle
    from fog.search.pb_grid import build_pb_grid
    from fog.search.pb_leaves import commit_pair_branches

    b = EngineBattle(ATTACKER, WEAK2, seed=0, battle_tag="pbgrid-onesided")
    draco = next(i for i, m in enumerate(b.active("p1").moves)
                 if str(m.id).lower().replace(" ", "") == "dracometeor")
    outcomes, _ = commit_pair_branches(b, "p1", draco,
                                       int(np.flatnonzero(b.action_mask("p2"))[0]))
    leaf = next(o.leaf for o in outcomes if o.leaf is not None
                and tuple(o.leaf.pending_sides()) == ("p2",))
    own, opp = _truth_grid(leaf, 1, 1)
    with pytest.raises(ValueError, match="do not own"):
        build_pb_grid(leaf, "p1", own, opp)


# A three-Pokemon level-5 team: any Specs hit knocks out the lead, leaving 2 live bench -> our
# forced replacement is a MULTI-choice one-sided decision (F2), not a trivial single switch.
WEAK3 = WEAK2 + """
Dragonite @ Heavy-Duty Boots
Level: 5
Ability: Multiscale
EVs: 4 HP
Serious Nature
- Extreme Speed
- Earthquake
- Dragon Dance
- Roost
"""


def _p1_replacement_leaf(seed: int = 1):
    """A leaf where OUR lead (p1) has just fainted to the opponent's Draco Meteor and we owe a
    forced replacement with two live bench Pokemon (pending == {p1})."""
    from fog.env.engine_battle import EngineBattle
    from fog.search.pb_leaves import commit_pair_branches

    b = EngineBattle(WEAK3, ATTACKER, seed=seed, battle_tag=f"pbgrid-repl-{seed}")
    draco = next(i for i, m in enumerate(b.active("p2").moves)
                 if str(m.id).lower().replace(" ", "") == "dracometeor")
    outcomes, _ = commit_pair_branches(
        b, "p1", int(np.flatnonzero(b.action_mask("p1"))[0]), draco)
    return next(o.leaf for o in outcomes if o.leaf is not None
                and tuple(o.leaf.pending_sides()) == ("p1",))


def test_one_sided_grid_uses_a_synthetic_observe_column():
    """F2: our forced replacement builds a grid with a single synthetic opponent label, avail2 all
    True at B=1, no real opponent action index, and every cell committed one-sidedly."""
    from fog.search.pb_grid import OBSERVE_LABEL, build_pb_grid

    leaf = _p1_replacement_leaf()
    own, opp = _truth_grid(leaf, 2, 3)
    g = build_pb_grid(leaf, "p1", own, opp, seed=4)

    assert g.labels2 == [OBSERVE_LABEL]
    assert g.avail2.shape == (3, 1) and g.avail2.all()
    assert (g.local_action2 == -1).all(), "the observe column carries no real opponent action"
    # our labels are the live switches; every own-world can play each (species-based, k-invariant)
    assert all(lab.startswith("s:") for lab in g.labels1)
    assert g.avail1.shape == (2, len(g.labels1)) and g.avail1.all()
    # budget: one commit per (own-world k, our-label a, opp-world j), with B == 1
    assert g.sims_run == 2 * len(g.labels1) * 3
    # branch bookkeeping identical to the two-sided path
    cell_mass = g.branch_prob.sum(axis=-1)
    available = (g.avail1[:, None, :, None] & g.avail2[None, :, None, :]) & ~g.failed
    np.testing.assert_allclose(cell_mass[available], 1.0, atol=1e-9)
    assert not g.failed.any()
    # leaves stay censored (19.12)
    assert len(g.leaf_obs) > 0
    assert (np.asarray(g.leaf_obs[0].opp_revealed["tokens"]) == -1).any()


def test_one_sided_grid_handles_a_pivot_force_switch():
    """A mid-turn U-turn force-switch (not a faint) is also one-sided and forks correctly — the
    plan's explicit check that fork_for_own_world works off a decision boundary."""
    from fog.env.engine_battle import EngineBattle
    from fog.search.pb_grid import OBSERVE_LABEL, build_pb_grid
    from fog.search.pb_leaves import commit_pair_branches

    pivot_p1 = """Scizor @ Heavy-Duty Boots
Level: 50
Ability: Technician
EVs: 252 HP
Adamant Nature
- U-turn
- Bullet Punch
- Knock Off
- Roost

""" + WEAK3
    pivot_p2 = """Blissey @ Leftovers
Level: 50
Ability: Natural Cure
EVs: 252 HP
Calm Nature
- Seismic Toss
- Soft-Boiled
- Toxic
- Protect
"""
    b = EngineBattle(pivot_p1, pivot_p2, seed=0, battle_tag="pbgrid-pivot")
    uturn = next(i for i, m in enumerate(b.active("p1").moves)
                 if str(m.id).lower().replace(" ", "") == "uturn")
    softb = next(i for i, m in enumerate(b.active("p2").moves)
                 if str(m.id).lower().replace(" ", "") == "softboiled")
    outcomes, _ = commit_pair_branches(b, "p1", uturn, softb)
    leaf = next(o.leaf for o in outcomes if o.leaf is not None
                and tuple(o.leaf.pending_sides()) == ("p1",))
    own, opp = _truth_grid(leaf, 1, 2)
    g = build_pb_grid(leaf, "p1", own, opp, seed=1)
    assert g.labels2 == [OBSERVE_LABEL]
    assert all(lab.startswith("s:") for lab in g.labels1) and len(g.labels1) >= 1
    assert g.sims_run > 0 and not g.failed.any()


def test_double_ko_takes_the_two_sided_path():
    """A simultaneous double faint (both sides owe a replacement) is a NORMAL two-sided decision,
    not a one-sided one: both label spaces are real switches, never the synthetic observe column."""
    from fog.env.engine_battle import EngineBattle
    from fog.search.pb_grid import OBSERVE_LABEL, build_pb_grid
    from fog.search.pb_leaves import commit_pair_branches

    dko_p1 = """Drifblim @ Life Orb
Level: 5
Ability: Aftermath
EVs: 4 HP
Serious Nature
- Explosion
- Shadow Ball
- Tackle
- Protect

Kingambit @ Black Glasses
Level: 5
Ability: Supreme Overlord
EVs: 252 HP
Adamant Nature
- Sucker Punch
- Iron Head
- Kowtow Cleave
- Swords Dance
"""
    dko_p2 = """Cloyster @ Life Orb
Level: 5
Ability: Skill Link
EVs: 4 HP
Serious Nature
- Icicle Spear
- Rock Blast
- Tackle
- Protect

Kingambit @ Black Glasses
Level: 5
Ability: Supreme Overlord
EVs: 252 HP
Adamant Nature
- Sucker Punch
- Iron Head
- Kowtow Cleave
- Swords Dance
"""
    b = EngineBattle(dko_p1, dko_p2, seed=0, battle_tag="pbgrid-dko")
    expl = next(i for i, m in enumerate(b.active("p1").moves)
                if str(m.id).lower().replace(" ", "") == "explosion")
    tackle = next(i for i, m in enumerate(b.active("p2").moves)
                  if str(m.id).lower().replace(" ", "") == "tackle")
    outcomes, _ = commit_pair_branches(b, "p1", expl, tackle)
    leaf = next(o.leaf for o in outcomes if o.leaf is not None
                and tuple(o.leaf.pending_sides()) == ("p1", "p2"))
    own, opp = _truth_grid(leaf, 2, 2)
    g = build_pb_grid(leaf, "p1", own, opp, seed=1)
    assert OBSERVE_LABEL not in g.labels2, "a double KO is two-sided; the opponent really chooses"
    assert all(lab.startswith("s:") for lab in g.labels2)
    assert (g.local_action2[g.avail2] >= 0).all(), "real opponent replacement actions committed"


def test_grid_deterministic_given_seed():
    from fog.search.pb_grid import build_pb_grid

    tables = []
    for _ in range(2):
        b = _battle(seed=4)
        own, opp = _truth_grid(b, 1, 1)
        g = build_pb_grid(b, "p1", own, opp, seed=9)
        tables.append((g.branch_prob.copy(), g.pruned_mass.copy(), g.sims_run))
    np.testing.assert_array_equal(tables[0][0], tables[1][0])
    np.testing.assert_array_equal(tables[0][1], tables[1][1])
    assert tables[0][2] == tables[1][2]
