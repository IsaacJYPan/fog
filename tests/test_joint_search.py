"""Joint sampled counterfactual search (SEARCH_PLAN §4): the pure regret-matching core (the
bluff test IS the method's defining property), posterior math, the posterior-features head input,
own-world forks, own-state sources, and end-to-end engine decisions."""
from __future__ import annotations

import numpy as np
import pytest

from fog.search.joint import (
    _floor_mix,
    action_posteriors,
    joint_regret_matching,
    posterior_features_np,
)

# ---------------------------------------------------------------------------------- pure math
def test_action_posteriors_hand_computed():
    prior = np.array([0.25, 0.75])
    pi = np.array([[0.8, 0.2], [0.4, 0.6]])
    post = action_posteriors(prior, pi)
    # post[k, a] = prior[k] pi[k,a] / sum_k'
    a0 = np.array([0.25 * 0.8, 0.75 * 0.4])
    a1 = np.array([0.25 * 0.2, 0.75 * 0.6])
    np.testing.assert_allclose(post[:, 0], a0 / a0.sum())
    np.testing.assert_allclose(post[:, 1], a1 / a1.sum())
    # a column nobody plays is all-zero, not NaN
    post = action_posteriors(prior, np.array([[1.0, 0.0], [1.0, 0.0]]))
    np.testing.assert_allclose(post[:, 1], 0.0)


def test_floor_mix_keeps_distribution():
    avail = np.array([[True, True, False], [True, True, True]])
    pi = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out = _floor_mix(pi, avail, floor=0.05)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert (out[avail] >= 0.05 - 1e-12).all()
    assert (out[~avail] == 0).all()


def test_posterior_features_shapes_and_extremes():
    from fog.models.heads import N_POSTERIOR_FEATURES

    uniform = posterior_features_np(np.array([0.25] * 4), k_index=1)
    assert uniform.shape == (N_POSTERIOR_FEATURES,)
    assert uniform[0] == pytest.approx(1.0)          # normalized entropy of uniform
    assert uniform[1] == pytest.approx(0.25)         # mass on this state
    assert uniform[3] == pytest.approx(1.0)          # all 4 states effective
    point = posterior_features_np(np.array([0.0, 1.0, 0.0, 0.0]), k_index=1)
    assert point[0] == pytest.approx(0.0)
    assert point[1] == pytest.approx(1.0)
    assert point[2] == pytest.approx(1.0)


def _bluff_game(coupled: bool):
    """Two states (Strong, Weak) x two public actions (attack, wait). Attacking is good for S
    only while it doesn't identify S; a W attack ('bluff') is profitable only while unexpected.
    q depends on the induced posterior — the coupling channel. Analytic equilibrium (y = S attack
    prob ≈ 1): W attacks x with x/(x+y) = 0.1/0.4 = 1/4 -> x = 1/3, post(S|attack) = 3/4."""
    prior = np.array([0.5, 0.5])
    avail = np.ones((2, 2), dtype=bool)
    frozen = action_posteriors(prior, np.full((2, 2), 0.5))  # what 'independent' search assumes

    def values_fn(post):
        p = post if coupled else frozen
        return np.array([
            [0.8 - 1.0 * p[0, 0], 0.0],      # S: attack value drops as attack identifies S
            [0.1 - 0.4 * p[1, 0], 0.0],      # W: bluff value, gone once expected
        ])

    return values_fn, prior, avail


def test_joint_finds_bluff_independent_does_not():
    """THE defining property (SEARCH_PLAN §4.3 test (a)): with the belief coupling live, the weak
    state mixes in bluffs (protecting the strong state's attack); with values frozen at the prior
    (exactly what K independent searches compute), the bluff never appears."""
    floor = 0.02
    pi0 = np.full((2, 2), 0.5)

    fn, prior, avail = _bluff_game(coupled=False)
    pi_ind, _ = joint_regret_matching(fn, prior, pi0, avail, iters=200, floor=floor)
    assert pi_ind[0, 0] > 0.9            # S always attacks (0.8 - 0.5 > 0)
    assert pi_ind[1, 0] < 2 * floor      # W never bluffs (0.1 - 0.2 < 0)

    fn, prior, avail = _bluff_game(coupled=True)
    pi_joint, post = joint_regret_matching(fn, prior, pi0, avail, iters=200, floor=floor)
    np.testing.assert_allclose(pi_joint.sum(axis=1), 1.0)
    assert pi_joint[1, 0] > 0.25, "weak state must bluff when the coupling is live (analytic 1/3)"
    assert pi_joint[0, 0] > 0.9, "the bluff PROTECTS attacking: strong state keeps it"
    # the posterior after 'attack' must NOT identify S (analytic 3/4)
    assert 0.65 < post[0, 0] < 0.85


def test_joint_regret_matching_deterministic():
    fn, prior, avail = _bluff_game(coupled=True)
    pi0 = np.full((2, 2), 0.5)
    a = joint_regret_matching(fn, prior, pi0, avail, iters=50, floor=0.02)
    b = joint_regret_matching(fn, prior, pi0, avail, iters=50, floor=0.02)
    np.testing.assert_array_equal(a[0], b[0])


# --------------------------------------------------------------------------- head + agent wiring
def test_policy_value_head_posterior_input():
    import torch

    from fog.models.heads import PolicyValueHead

    torch.manual_seed(0)
    head = PolicyValueHead(state_dim=16, belief_dim=8, posterior_dim=4)
    s = torch.randn(3, 16)
    b = torch.randn(3, 8)
    mask = torch.ones(3, 14, dtype=torch.bool)
    lp_none, v_none = head(s, mask, b)                                    # absent -> zeros
    lp_zero, v_zero = head(s, mask, b, posterior_features=torch.zeros(3, 4))
    torch.testing.assert_close(v_none, v_zero)
    torch.testing.assert_close(lp_none, lp_zero)
    _, v_feat = head(s, mask, b, posterior_features=torch.ones(3, 4))
    assert not torch.allclose(v_none, v_feat), "posterior features must reach the value"


def test_agent_flag_wires_posterior_dim():
    from fog.models.agent import FogAgent
    from fog.models.heads import N_POSTERIOR_FEATURES

    assert FogAgent({}).policy_value.posterior_dim == 0  # default OFF: checkpoints unchanged
    on = FogAgent({"model": {"use_posterior_features": True}})
    assert on.policy_value.posterior_dim == N_POSTERIOR_FEATURES
    on.assert_within_budget()


# ------------------------------------------------------------------------------- engine-backed
from fog.env.engine_battle import HAVE_ENGINE  # noqa: E402

engine = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

P_ME = """Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Knock Off

Kingambit @ Black Glasses
Ability: Supreme Overlord
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance
"""

# same species (shared-species pool, POOL-A style), different hidden sets
P_ME_VARIANT = """Great Tusk @ Leftovers
Ability: Protosynthesis
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Stealth Rock

Kingambit @ Air Balloon
Ability: Supreme Overlord
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Substitute
"""

P_OPP = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave

Hydreigon @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Dark Pulse
- Flash Cannon
- U-turn
"""


@pytest.fixture(scope="module")
def plain_agent():
    from fog.models.agent import FogAgent

    a = FogAgent.from_tables({})
    a.eval()
    return a


@pytest.fixture(scope="module")
def posterior_agent():
    from fog.models.agent import FogAgent

    a = FogAgent.from_tables({"model": {"use_posterior_features": True}})
    a.eval()
    return a


def _battle(seed=0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(P_ME, P_OPP, seed=seed, battle_tag=f"joint-test-{seed}")


def _perturb_own(battle):
    """p1 truth with slot 0's item and one hidden move swapped (a possible-me that isn't me)."""
    from fog.constants.tables import TABLES

    world = battle.truth_tokens("p1")
    ids = TABLES.items["ids"]
    new_item = next(i for i, s in enumerate(ids) if not s.startswith("<") and i != int(world[0, 1]))
    learnable = np.flatnonzero(TABLES.learnset_mask(int(world[0, 0])))
    current = {int(m) for m in world[0, 3:7]}
    new_move = next(int(m) for m in learnable if int(m) not in current)
    world[0, 1] = new_item
    world[0, 6] = new_move
    return world, ids[new_item], TABLES.moves["ids"][new_move]


@engine
def test_fork_for_own_world_swaps_my_hidden_keeps_public():
    battle = _battle()
    world, item_id, move_id = _perturb_own(battle)
    fork = battle.fork_for_own_world("p1", world, seed=3)

    mine, forked = battle.side("p1").pokemon[0], fork.side("p1").pokemon[0]
    assert str(forked.id) == str(mine.id)
    assert str(forked.item) == item_id != str(mine.item)
    assert move_id in {str(m.id) for m in forked.moves}
    assert abs(mine.hp / mine.maxhp - forked.hp / forked.maxhp) < 0.01  # HP is public as a fraction
    # the OPPONENT side is untouched engine truth
    opp, forked_opp = battle.side("p2").pokemon[0], fork.side("p2").pokemon[0]
    assert str(forked_opp.item) == str(opp.item) and forked_opp.hp == opp.hp


@engine
def test_fork_two_sided():
    battle = _battle(seed=1)
    own, own_item, _ = _perturb_own(battle)
    opp = battle.truth_tokens("p2")
    from fog.constants.tables import TABLES

    ids = TABLES.items["ids"]
    opp[0, 1] = next(i for i, s in enumerate(ids) if not s.startswith("<") and i != int(opp[0, 1]))
    fork = battle.fork_for_own_world("p1", own, opp_world_tokens=opp, seed=5)
    assert str(fork.side("p1").pokemon[0].item) == own_item
    assert str(fork.side("p2").pokemon[0].item) == ids[int(opp[0, 1])]


@engine
def test_enumerated_pool_states_filters_on_reveals():
    from fog.search.own_states import EnumeratedPoolStates

    pool = [P_ME, P_ME_VARIANT]
    battle = _battle(seed=2)
    source = EnumeratedPoolStates(pool, battle.tables)

    own = source.states(battle, "p1")
    assert own.k == 2, "nothing set-distinguishing revealed yet: both pool teams consistent"
    np.testing.assert_allclose(own.weights, [0.5, 0.5])
    np.testing.assert_array_equal(own.tokens[own.true_index], battle.truth_tokens("p1"))

    # revealing Knock Off (not on the variant's sheet) must eliminate the variant
    battle._record_reveal("p1", "greattusk", move="knockoff")
    own = source.states(battle, "p1")
    assert own.k == 1
    np.testing.assert_array_equal(own.tokens[0], battle.truth_tokens("p1"))
    assert own.true_index == 0


@engine
def test_sampled_own_states_includes_truth_and_is_seeded(plain_agent):
    from fog.search.own_states import SampledOwnStates

    battle = _battle(seed=3)
    truth = battle.truth_tokens("p1")

    a = SampledOwnStates(plain_agent, k=4, oversample=2, seed=7).states(battle, "p1")
    assert a.tokens.shape[1:] == (6, 9) and 1 <= a.k <= 4
    assert a.weights.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(a.tokens[a.true_index], truth)

    b = SampledOwnStates(plain_agent, k=4, oversample=2, seed=7).states(battle, "p1")
    np.testing.assert_array_equal(a.tokens, b.tokens)


@engine
@pytest.mark.parametrize("agent_fixture", ["posterior_agent", "plain_agent"])
def test_joint_search_action_end_to_end(agent_fixture, request):
    """Legal decision, live battle untouched, distillation-ready distribution, seed-deterministic —
    with the coupling consumer ON (posterior_agent) and OFF (plain_agent: degenerate independent
    mode, must still run; the contrast case is pinned by the pure-math bluff test)."""
    from fog.search.joint import joint_search_action
    from fog.search.own_states import EnumeratedPoolStates

    agent = request.getfixturevalue(agent_fixture)
    battle = _battle(seed=4)
    pre = battle.state.to_string()
    source = EnumeratedPoolStates([P_ME, P_ME_VARIANT], battle.tables)

    res = joint_search_action(agent, battle, "p1", own_source=source, iters=4, seed=9)
    assert battle.action_mask("p1")[res.action], "joint search chose an illegal action"
    assert battle.state.to_string() == pre, "search must not mutate the live battle"
    assert res.action_dist.shape == (14,) and res.action_dist.sum() == pytest.approx(1.0)
    assert res.policies.shape[0] == 2 and res.posterior.shape == res.policies.shape
    np.testing.assert_allclose(res.policies.sum(axis=1), 1.0)
    assert res.failures == 0, "clean pool position should simulate without failures"

    res2 = joint_search_action(agent, battle, "p1", own_source=source, iters=4, seed=9)
    assert res2.action == res.action
    np.testing.assert_array_equal(res2.action_dist, res.action_dist)


@engine
def test_public_labels_differ_across_worlds(plain_agent):
    """The label space must be PUBLIC actions: the two pool variants share revealed moves but
    differ in one hidden move each, so the union label set exceeds any single world's set."""
    from fog.search.joint import joint_search_action
    from fog.search.own_states import EnumeratedPoolStates

    agent = plain_agent
    battle = _battle(seed=5)
    source = EnumeratedPoolStates([P_ME, P_ME_VARIANT], battle.tables)
    res = joint_search_action(agent, battle, "p1", own_source=source, iters=2, seed=11)
    move_labels = {lab for lab in res.labels if lab.startswith("m:")}
    assert any("knockoff" in lab for lab in move_labels)
    assert any("stealthrock" in lab for lab in move_labels), (
        "the variant's hidden move must appear in the public-label union"
    )
