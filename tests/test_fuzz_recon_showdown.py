"""The Showdown-truth reconstruction fuzzer (ROADMAP P1.3). Needs poke-engine, Node and the
Showdown build; skipped otherwise."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fog.env.engine_battle import HAVE_ENGINE
from fog.env.showdown_direct import DEFAULT_SHOWDOWN_ROOT

SHOWDOWN_BUILT = (DEFAULT_SHOWDOWN_ROOT / "dist" / "sim" / "index.js").is_file()
pytestmark = pytest.mark.skipif(
    not HAVE_ENGINE or shutil.which("node") is None or not SHOWDOWN_BUILT,
    reason="requires poke-engine, Node and the pinned local Pokemon Showdown build",
)


def test_rebuilt_states_match_showdown_on_public_and_own_fields():
    from fog.scripts.fuzz_recon_showdown import run

    report = run(Path("data/teams/pool_b"), games=6, seed=3, max_turns=40)
    assert report["positions"] > 100 and report["reconstruction_errors"] == 0
    fields = report["mismatch_positions_by_field"]
    # Faults fixed so far must stay fixed: hazards, weather and its timer, terrain, Tera flags,
    # status, own HP / stats / items / PP, and which Pokemon is active.
    fixed = ("stealth_rock", "sticky_web", "spikes", "weather", "terrain", "trick_room",
             "terastallized", "tera_type", "status", "me.hp", "me.stats", "me.item", "me.moves",
             "me.pp_zero", "active", "missing_mon", "rest_sleep", "boosts")
    assert [name for name in fields if any(key in name for key in fixed)] == []
    # Whatever remains is rare; hidden Regenerator healing is unobservable and not counted.
    other = sum(count for name, count in fields.items() if "hidden_regenerator" not in name)
    assert other <= 0.1 * report["positions"]


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_tera_ogerpon_keeps_opponent_active_hp_and_used_tera(role):
    from fog.constants.tables import TABLES
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.state_recon import reconstruct_battle
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import to_packed
    from fog.search.engine_mcts import ShowdownEngineMCTSController

    ogerpon = """Ogerpon @ Heavy-Duty Boots
Ability: Defiant
Tera Type: Grass
- Swords Dance
- Ivy Cudgel

Snorlax @ Leftovers
Ability: Immunity
- Splash
"""
    other = """Mew @ Leftovers
Ability: Synchronize
- Seismic Toss
- Splash
"""
    opponent = "p2" if role == "p1" else "p1"
    teams = {role: other, opponent: ogerpon}
    options = {"formatid": "gen9customgame@@@!teampreview", "strictChoices": True,
               "seed": "sodium," + "1" * 64,
               **{r: {"name": r, "team": to_packed(t)} for r, t in teams.items()}}
    tracker = ShowdownEngineMCTSController(k=1, iterations=1)
    with ShowdownDirectClient() as client, LocalShowdownBattle.create(client, options) as local:
        choices = {role: "move 1", opponent: "move 1 terastallize"}
        assert not local.step(choices["p1"], choices["p2"]).choice_errors
        view = local.views[role]
        state = reconstruct_battle(view, truth_tokens_from_paste(ogerpon, TABLES), TABLES,
                                   tracker._tracker(view), role=role).state
        side = state.side_two if role == "p1" else state.side_one
        active = side.pokemon[int(side.active_index)]
        assert str(active.id).lower() == "ogerpon"
        assert abs(active.hp - 201) <= 2 and active.maxhp == 301
        assert active.terastallized and str(active.tera_type).lower() == "grass"
        assert side.attack_boost == 2 and side.speed_boost == 1

        choices[role] = "move 2"
        choices[opponent] = "switch 2"
        assert not local.step(choices["p1"], choices["p2"]).choice_errors
        state = reconstruct_battle(view, truth_tokens_from_paste(ogerpon, TABLES), TABLES,
                                   tracker._tracker(view), role=role).state
        side = state.side_two if role == "p1" else state.side_one
        assert str(side.pokemon[int(side.active_index)].id).lower() == "snorlax"
        assert next(p for p in side.pokemon if str(p.id).lower() == "ogerpon").terastallized

        # After re-entry poke-env uses the Tera forme as species. It must still match the
        # original preview/world slot, rather than resurrecting it and defaulting active to 0.
        choices[opponent] = "switch 2"
        assert not local.step(choices["p1"], choices["p2"]).choice_errors
        assert view.opponent_active_pokemon.species == "ogerpontealtera"
        state = reconstruct_battle(view, truth_tokens_from_paste(ogerpon, TABLES), TABLES,
                                   tracker._tracker(view), role=role).state
        side = state.side_two if role == "p1" else state.side_one
        active = side.pokemon[int(side.active_index)]
        assert str(active.id).lower() == "ogerpon"
        assert abs(active.hp - 201) <= 2 and active.terastallized
