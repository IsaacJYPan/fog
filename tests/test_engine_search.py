"""Engine-backed determinized search tests: fork censoring/isolation, stepper leaf
semantics (fix 1: leaves under fog), end-to-end action selection. Skipped without poke-engine."""
from __future__ import annotations

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

P_SEARCHER = """Great Tusk @ Heavy-Duty Boots
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

P_OPPONENT = """Tyranitar @ Leftovers
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
def agent():
    from fog.models.agent import FogAgent

    a = FogAgent.from_tables({})
    a.eval()
    return a


def _battle(seed=0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(P_SEARCHER, P_OPPONENT, seed=seed, battle_tag=f"search-test-{seed}")


def _perturbed_world(battle):
    """Opponent truth with slot 0's item and one hidden move swapped — a 'sampled' world that
    provably differs from engine truth."""
    from fog.constants.tables import TABLES

    world = battle.truth_tokens("p2")
    ids = TABLES.items["ids"]
    new_item = next(
        i for i, s in enumerate(ids) if not s.startswith("<") and i != int(world[0, 1])
    )
    sp_idx = int(world[0, 0])
    current_moves = {int(m) for m in world[0, 3:7]}
    learnable = np.flatnonzero(TABLES.learnset_mask(sp_idx))
    new_move = next(int(m) for m in learnable if int(m) not in current_moves)
    world[0, 1] = new_item
    world[0, 6] = new_move
    return world, ids[new_item], TABLES.moves["ids"][new_move]


def test_fork_uses_sampled_world_not_truth():
    battle = _battle()
    world, item_id, move_id = _perturbed_world(battle)
    fork = battle.fork_for_search("p1", world, seed=7)

    forked_opp = fork.side("p2").pokemon[0]
    true_opp = battle.side("p2").pokemon[0]
    assert str(forked_opp.id) == str(true_opp.id)  # species is public (preview)
    assert str(forked_opp.item) == item_id != str(true_opp.item)
    assert move_id in {str(m.id) for m in forked_opp.moves}
    # public board state carried over as a FRACTION (spread is hidden, so maxhp may differ —
    # only the HP bar percentage is public, exactly like Showdown)
    true_frac = true_opp.hp / true_opp.maxhp
    forked_frac = forked_opp.hp / forked_opp.maxhp
    assert abs(true_frac - forked_frac) < 0.01
    assert str(forked_opp.status).lower() == str(true_opp.status).lower()
    # searcher's own side is engine truth
    mine, forked_mine = battle.side("p1").pokemon[0], fork.side("p1").pokemon[0]
    assert str(forked_mine.item) == str(mine.item) and forked_mine.hp == mine.hp


def test_fork_step_never_mutates_original():
    battle = _battle(seed=1)
    pre_state = battle.state.to_string()
    pre_turn = battle.turn
    world, _, _ = _perturbed_world(battle)
    fork = battle.fork_for_search("p1", world, seed=3)

    legal = np.flatnonzero(fork.action_mask("p1"))
    moves = {"p1": fork.engine_action("p1", int(legal[0]))}
    if "p2" in fork.pending_sides():
        opp_legal = np.flatnonzero(fork.action_mask("p2"))
        moves["p2"] = fork.engine_action("p2", int(opp_legal[0]))
    fork.step(moves)

    assert battle.state.to_string() == pre_state
    assert battle.turn == pre_turn
    assert not battle.observations or max(battle.observations) < max(fork.observations, default=0) + 1


def test_stepper_leaf_is_censored():
    """Fix 1: the leaf observation's belief input must stay fogged — root reveals plus the
    simulated ply's reveals only, never the sampled team clamped in wholesale."""
    from fog.encoding.observation import Observation
    from fog.search.engine_step import EngineWorldStepper

    battle = _battle(seed=2)
    world, _, _ = _perturbed_world(battle)
    stepper = EngineWorldStepper(battle, "p1", seed=5)
    legal = np.flatnonzero(battle.action_mask("p1"))

    r = stepper(int(legal[0]), world)
    assert stepper.failures == 0
    assert r is not None
    if isinstance(r, float):
        assert -1.0 <= r <= 1.0
    else:
        assert isinstance(r, Observation)
        r.validate()
        tokens = r.opp_revealed["tokens"]
        assert (tokens == -1).any(), "leaf belief input must keep hidden attributes hidden"
        assert (tokens[0, 7:14] == -1).all(), \
            "submitted nature/EVs are never revealed by one simulated ply"


def test_engine_determinized_action_end_to_end(agent):
    from fog.search.engine_step import engine_determinized_action

    battle = _battle(seed=4)
    pre_state = battle.state.to_string()
    a = engine_determinized_action(agent, battle, "p1", k=2, seed=6)
    assert battle.action_mask("p1")[a], f"search chose illegal action {a}"
    assert battle.state.to_string() == pre_state, "search must not mutate the live battle"


def _play(battle, turns, seed=0):
    """Advance a battle a few turns of random/max-damage play for a mid-game position."""
    from fog.env.engine_scripted import EngineMaxDamagePolicy, EngineRandomPolicy

    rand, maxd = EngineRandomPolicy(seed=seed), EngineMaxDamagePolicy(seed=seed + 1)
    for _ in range(turns):
        if battle.finished:
            break
        moves = {}
        for role in battle.pending_sides():
            pol = rand if role == "p1" else maxd
            moves[role] = battle.engine_action(role, pol.choose(battle, role))
        battle.step(moves)
    return battle


def _play_until_searcher_pending(battle, role, max_turns=30, seed=0):
    """Advance to a decision where `role` must actually act — the only moment search is invoked in
    real play (poke-env calls choose_move only for the pending player)."""
    from fog.env.engine_scripted import EngineMaxDamagePolicy, EngineRandomPolicy

    rand, maxd = EngineRandomPolicy(seed=seed), EngineMaxDamagePolicy(seed=seed + 1)
    for _ in range(max_turns):
        if battle.finished or role in battle.pending_sides():
            break
        moves = {}
        for r in battle.pending_sides():
            pol = rand if r == "p1" else maxd
            moves[r] = battle.engine_action(r, pol.choose(battle, r))
        battle.step(moves)
    return battle


# ---------------------------------------------------------------- Showdown-path state reconstruction
def test_reconstruction_round_trip_exact():
    """engine state -> EngineView (poke-env-shaped) -> reconstruct -> the position matches. No ply
    is simulated, so there is no randomness: it must match exactly. Handing the TRUE world means any
    mismatch is reconstruction error, not sampling."""
    from fog.constants.tables import TABLES
    from fog.env.engine_battle import to_id
    from fog.env.state_recon import reconstruct_battle

    battle = _play(_battle(seed=8), 4, seed=8)
    if battle.finished:
        pytest.skip("battle ended early")
    role = "p1"
    view = battle.view(role)
    recon = reconstruct_battle(view, battle.truth_tokens("p2"), TABLES, battle.trackers[role],
                               role=role)
    orig, new = battle.state, recon.state
    assert to_id(orig.weather) == to_id(new.weather)
    assert to_id(orig.terrain) == to_id(new.terrain)
    assert bool(orig.trick_room) == bool(new.trick_room)
    for so, sn in ((orig.side_one, new.side_one), (orig.side_two, new.side_two)):
        assert str(so.active_index) == str(sn.active_index)
        assert so.side_conditions.stealth_rock == sn.side_conditions.stealth_rock
        assert so.side_conditions.spikes == sn.side_conditions.spikes
    # my side (side_one = p1 here): exact computed stats + HP
    o_by = {to_id(p.id): p for p in orig.side_one.pokemon if to_id(p.id) not in ("", "none")}
    n_by = {to_id(p.id): p for p in new.side_one.pokemon if to_id(p.id) not in ("", "none")}
    assert sorted(o_by) == sorted(n_by)
    for sid, po in o_by.items():
        pn = n_by[sid]
        assert (po.attack, po.speed, po.maxhp) == (pn.attack, pn.speed, pn.maxhp)
        assert po.hp == pn.hp and to_id(po.status) == to_id(pn.status)


def test_reconstruction_is_steppable_and_isolated():
    from fog.constants.tables import TABLES
    from fog.env.state_recon import reconstruct_battle

    battle = _play(_battle(seed=12), 3, seed=2)
    if battle.finished:
        pytest.skip("battle ended early")
    role = "p1"
    pre = battle.state.to_string()
    recon = reconstruct_battle(battle.view(role), battle.truth_tokens("p2"), TABLES,
                               battle.trackers[role], role=role)
    legal = np.flatnonzero(recon.action_mask(role))
    assert len(legal) >= 1
    clone = recon.clone_for_step(seed=1)
    moves = {role: clone.engine_action(role, int(legal[0]))}
    if "p2" in clone.pending_sides():
        opp = np.flatnonzero(clone.action_mask("p2"))
        moves["p2"] = clone.engine_action("p2", int(opp[0]))
    clone.step(moves)
    assert battle.state.to_string() == pre, "reconstruction/clone must not mutate the live battle"


def test_showdown_stepper_leaf_censored_and_end_to_end(agent):
    """The Showdown stepper (poke-env Battle stand-in = EngineView) yields fogged leaves and drives
    determinized_action to a legal action without touching the live battle."""
    import torch

    from fog.encoding.observation import Observation
    from fog.env.engine_runner import _stack_obs
    from fog.search.determinized import determinized_action
    from fog.search.engine_step import ShowdownWorldStepper

    role = "p1"
    battle = _play_until_searcher_pending(_battle(seed=15), role, seed=4)
    if battle.finished or role not in battle.pending_sides():
        pytest.skip("no searcher-pending position reached")
    view = battle.view(role)
    tracker = battle.trackers[role]
    obs = Observation.from_battle(view, tracker=tracker)
    batch = _stack_obs([obs], torch.device("cpu"))
    pre = battle.state.to_string()

    stepper = ShowdownWorldStepper(view, tracker, role=role, seed=3)
    a = determinized_action(agent, batch, step_fn=stepper, k=4, prior_weight=0.5)
    assert obs.legality_mask[a], "search chose an illegal action"
    assert stepper.failures == 0, "reconstruction should not fail on a clean pool position"
    assert battle.state.to_string() == pre, "search must not mutate the live battle"

    # a leaf directly: still fogged
    leaf = stepper._fork(battle.truth_tokens("p2"))  # reconstructed base for one world
    leaf_clone = leaf.clone_for_step(seed=7)
    legal = np.flatnonzero(leaf_clone.action_mask(role))
    from fog.env.engine_scripted import EngineMaxDamagePolicy
    from fog.search.engine_step import _play_one_ply

    r = _play_one_ply(leaf_clone, role, int(legal[0]), EngineMaxDamagePolicy(seed=0),
                      max_forced_steps=8)
    if not isinstance(r, float):
        assert (r.opp_revealed["tokens"] == -1).any(), "leaf must keep hidden attributes hidden"


def test_world_mon_preserves_disabled_move():
    """Fix #3: a disabled/choice-locked move on the source (revealed) mon survives the rebuild."""
    from types import SimpleNamespace

    from fog.constants.tables import TABLES
    from fog.env.engine_battle import _world_engine_mon, to_id

    battle = _battle(seed=20)
    world = battle.truth_tokens("p2")
    src = battle.side("p2").pokemon[0]
    locked = to_id(src.moves[0].id)
    # a duck-typed source whose first revealed move is disabled (choice lock / Encore / Disable)
    stub = SimpleNamespace(
        level=int(src.level), hp=int(src.hp), maxhp=int(src.maxhp),
        status="none", rest_turns=0, sleep_turns=0, terastallized=False,
        moves=[SimpleNamespace(id=to_id(m.id), pp=16, disabled=(i == 0))
               for i, m in enumerate(src.moves)],
    )
    rebuilt = _world_engine_mon(world[0], stub, TABLES)
    disabled_ids = {to_id(m.id) for m in rebuilt.moves if bool(m.disabled)}
    assert locked in disabled_ids
    assert len([m for m in rebuilt.moves if m.disabled]) == 1
