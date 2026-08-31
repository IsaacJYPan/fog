"""commit_pair_branches (fog/search/pb_leaves.py, PROPOSAL_IMPL_PLAN Phase B(ii)): committed
action pairs enumerate into weighted branch outcomes that STOP at the next pending decision
(PROJECT_DESIGN_PROPOSAL.md §10.5) — pending forced replacements are returned unresolved, never
handed to a scripted chooser; finished branches carry the ±1/0 terminal convention; the whole
walk from a fresh battle to a terminal result works by committing decision after decision."""
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

# Two level-5 Pokemon: any Specs hit knocks out, so turn 1 produces a pending-replacement leaf
# and turn 2 (after the forced switch) ends the game.
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


def _battle(p1: str, p2: str, seed: int = 0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(p1, p2, seed=seed, battle_tag=f"pbleaf-test-{seed}")


def _move_slot(battle, role: str, move_id: str) -> int:
    active = battle.active(role)
    for i, m in enumerate(active.moves):
        if str(m.id).lower().replace(" ", "") == move_id:
            return i
    raise AssertionError(f"{move_id} not on {active.id}")


def _first_legal(battle, role: str) -> int:
    return int(np.flatnonzero(battle.action_mask(role))[0])


def _signature(battle) -> tuple:
    sig = [battle.turn, battle.finished, battle.winner]
    for side in (battle.state.side_one, battle.state.side_two):
        for p in side.pokemon:
            sig.append((int(p.hp), str(p.status)))
    return tuple(sig)


def test_pending_replacement_leaf_is_returned_unresolved():
    """§10.5: the knockout branch stops at the opponent's pending forced replacement — the leaf
    still shows the fainted side owing a decision; the miss branch completes the turn."""
    from fog.search.pb_leaves import commit_pair_branches

    b = _battle(ATTACKER, WEAK2)
    outcomes, pruned = commit_pair_branches(
        b, "p1", _move_slot(b, "p1", "dracometeor"), _first_legal(b, "p2"))
    assert sum(o.probability for o in outcomes) == pytest.approx(1.0)
    assert pruned == pytest.approx(0.0, abs=0.05)
    pendings = {tuple(o.leaf.pending_sides()) for o in outcomes if o.leaf is not None}
    assert ("p2",) in pendings, "knockout branch must stop AT the pending replacement"
    assert ("p1", "p2") in pendings, "the miss branch completes the turn normally"
    for o in outcomes:
        assert o.terminal is None, "a 2-Pokemon side with one faint is not a finished game"
        if tuple(o.leaf.pending_sides()) == ("p2",):
            assert not o.leaf.finished
            fainted = [p for p in o.leaf.state.side_two.pokemon if int(p.hp) == 0]
            assert fainted, "this leaf exists because a Pokemon fainted"


def test_one_sided_pending_validation_and_replacement_commit():
    """At a forced replacement only the owing side commits: wrong-role and stale-opponent-index
    calls raise, and the correct one-sided commit advances to the next full turn."""
    from fog.search.pb_leaves import commit_pair_branches

    b = _battle(ATTACKER, WEAK2)
    outcomes, _ = commit_pair_branches(
        b, "p1", _move_slot(b, "p1", "dracometeor"), _first_legal(b, "p2"))
    leaf = next(o.leaf for o in outcomes if o.leaf is not None
                and tuple(o.leaf.pending_sides()) == ("p2",))

    with pytest.raises(ValueError, match="not pending"):
        commit_pair_branches(leaf, "p1", 0)                      # not p1's decision
    with pytest.raises(ValueError, match="must be None"):
        commit_pair_branches(leaf, "p2", _first_legal(leaf, "p2"), opp_action_idx=0)

    replacement, _ = commit_pair_branches(leaf, "p2", _first_legal(leaf, "p2"))
    assert all(o.terminal is None for o in replacement)
    assert all(tuple(o.leaf.pending_sides()) == ("p1", "p2") for o in replacement)


def test_walk_to_terminal_and_payoff_perspective():
    """Committing decision after decision (always the most likely branch) reaches a finished
    game, and the terminal payoff is from the COMMITTING role's perspective."""
    from fog.search.pb_leaves import commit_pair_branches

    b = _battle(ATTACKER, WEAK2, seed=3)
    for _ in range(10):
        pending = tuple(b.pending_sides())
        role = pending[0] if len(pending) == 1 else "p1"
        opp = None if len(pending) == 1 else _first_legal(b, "p2")
        outcomes, _ = commit_pair_branches(b, role, _first_legal(b, role), opp)
        best = max(outcomes, key=lambda o: o.probability)
        if best.terminal is not None:
            assert best.terminal == (1.0 if role == "p1" else -1.0), \
                "Hydreigon wins; payoff must flip with the committing role"
            return
        b = best.leaf
    raise AssertionError("no terminal reached in 10 commits")


def test_completed_turn_leaves_equal_raw_step_branches():
    """No hidden extra stepping: on a non-lethal turn the outcomes are exactly the
    step_branches forks (same states, same probabilities), and the root is not advanced."""
    from fog.search.pb_leaves import commit_pair_branches

    b1, b2 = _battle(ATTACKER, DEFENDER, seed=5), _battle(ATTACKER, DEFENDER, seed=5)
    root_sig = _signature(b1)
    outcomes, _ = commit_pair_branches(
        b1, "p1", _move_slot(b1, "p1", "darkpulse"), _move_slot(b1, "p2", "thunderwave"))
    assert _signature(b1) == root_sig, "root must not be advanced"
    raw, _ = b2.step_branches({"p1": b2.engine_action("p1", _move_slot(b2, "p1", "darkpulse")),
                               "p2": b2.engine_action("p2", _move_slot(b2, "p2", "thunderwave"))})
    got = {(round(o.probability, 12), _signature(o.leaf)) for o in outcomes}
    want = {(round(p, 12), _signature(f)) for p, f in raw}
    assert got == want


def test_pruning_reports_dropped_mass():
    from fog.search.pb_leaves import commit_pair_branches

    b = _battle(ATTACKER, DEFENDER)
    outcomes, pruned = commit_pair_branches(
        b, "p1", _move_slot(b, "p1", "darkpulse"), _move_slot(b, "p2", "thunderwave"),
        min_prob=0.10)
    # Gen 9: the kept branches are ordinary-hit/no-flinch/Thunder-Wave-hit (69%) and
    # ordinary-hit/flinch (23/24 * 20% = 19.166...%).  The other four exact branches are below
    # the deliberately requested 10% threshold.
    assert len(outcomes) == 2
    assert pruned == pytest.approx(0.11833333333333333, abs=1e-6)
    assert sum(o.probability for o in outcomes) == pytest.approx(1.0)
