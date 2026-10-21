"""EngineBattle.step_branches: chance outcomes enumerated as
weighted forks instead of one sampled rollout. Pins: probability accounting, parent immutability,
pruning rules, fresh per-fork random streams, sampled-step consistency (a plain step() lands on
one of the enumerated branches), and determinism."""
from __future__ import annotations

import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

# Hydreigon (faster) Dark Pulse vs Tyranitar Thunder Wave at full HP enumerates exactly the
# Gen 9 also gives Thunder Wave 90% accuracy, so the crit x flinch grid splits the branches in
# which Tyranitar acts into hit/miss outcomes.
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


def _battle(seed: int = 0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(ATTACKER, DEFENDER, seed=seed, battle_tag=f"branch-test-{seed}")


def _move(battle, role: str, move_id: str) -> str:
    active = battle.active(role)
    for i, m in enumerate(active.moves):
        if str(m.id).lower().replace(" ", "") == move_id:
            return battle.engine_action(role, i)
    raise AssertionError(f"{move_id} not on {active.id}")


def _signature(battle) -> tuple:
    """Observable state fingerprint: per-mon hp/status both sides + turn/finished/winner."""
    sig = [battle.turn, battle.finished, battle.winner]
    for side in (battle.state.side_one, battle.state.side_two):
        for p in side.pokemon:
            sig.append((int(p.hp), str(p.status)))
    return tuple(sig)


MOVES = {"p1": "darkpulse", "p2": "thunderwave"}


def _step_args(b):
    return {r: _move(b, r, mid) for r, mid in MOVES.items()}


def test_branches_enumerate_with_probabilities_summing_to_one():
    b = _battle()
    branches, pruned = b.step_branches(_step_args(b))
    assert len(branches) == 6                    # crit x flinch x conditional Thunder Wave hit
    assert pruned == 0.0
    assert sum(p for p, _ in branches) == pytest.approx(1.0)
    probs = sorted((p for p, _ in branches), reverse=True)
    assert probs[0] == pytest.approx((23 / 24) * 0.8 * 0.9)
    assert probs[1] == pytest.approx((23 / 24) * 0.2)


def test_parent_battle_is_not_advanced():
    b = _battle()
    before = _signature(b)
    b.step_branches(_step_args(b))
    assert _signature(b) == before


def test_sampled_step_lands_on_an_enumerated_branch():
    """step() (the sampling path) must produce a state identical to one of the enumerated
    branches: same pre-state, same instruction sets, one selection."""
    a, b = _battle(seed=7), _battle(seed=7)
    branches, _ = a.step_branches(_step_args(a))
    b.step(_step_args(b))
    assert _signature(b) in {_signature(f) for _, f in branches}


def test_pruning_keeps_top_and_renormalizes():
    b = _battle()
    branches, pruned = b.step_branches(_step_args(b), min_prob=0.10)
    assert len(branches) == 2
    assert pruned == pytest.approx(1 - ((23 / 24) * 0.8 * 0.9 + (23 / 24) * 0.2))
    assert sum(p for p, _ in branches) == pytest.approx(1.0)
    # the most likely branch is always kept, even under an absurd threshold
    only, pruned = b.step_branches(_step_args(b), min_prob=1.0)
    assert len(only) == 1
    assert only[0][0] == pytest.approx(1.0)
    assert pruned == pytest.approx(1 - (23 / 24) * 0.8 * 0.9)


def test_max_branches_cap_keeps_the_most_likely():
    b = _battle()
    branches, pruned = b.step_branches(_step_args(b), max_branches=2)
    assert len(branches) == 2
    raw = sorted((p * (1.0 - pruned) for p, _ in branches), reverse=True)
    assert raw[0] == pytest.approx((23 / 24) * 0.8 * 0.9)
    assert raw[1] == pytest.approx((23 / 24) * 0.2)


def test_forks_draw_fresh_independent_random_futures():
    """Clones have independent random state."""
    b = _battle()
    branches, _ = b.step_branches(_step_args(b))
    rngs = [f.rng for _, f in branches]
    assert all(r is not b.rng for r in rngs)
    draws = [r.random() for r in rngs]
    assert len(set(draws)) == len(draws), "sibling forks must not replay one luck script"


def test_step_branches_deterministic_given_parent_seed():
    sigs = []
    for _ in range(2):
        b = _battle(seed=13)
        branches, _ = b.step_branches(_step_args(b))
        sigs.append(tuple((round(p, 12), _signature(f)) for p, f in branches))
    assert sigs[0] == sigs[1]


def test_branch_forks_are_steppable():
    """A branch fork is a live battle: it can be stepped onward (the future grid builder will
    walk forced replacements this way, stopping at the next pending decision)."""
    b = _battle()
    branches, _ = b.step_branches(_step_args(b))
    _, fork = branches[0]
    assert not fork.finished
    fork.step(_step_args(fork))                  # one more full turn on the fork
    assert _signature(fork) != _signature(b)
