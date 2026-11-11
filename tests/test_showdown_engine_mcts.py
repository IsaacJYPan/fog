"""EngineMCTS on the direct, in-memory Showdown backend, and the selection-script hooks."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from fog.eval.showdown_direct_selection import (
    ChooseAdapter,
    MatchResult,
    ledger_row,
    load_controller_factory,
    matched_schedule,
    sequential_stop,
)

REPO = Path(__file__).resolve().parents[1]


def _showdown_ready() -> bool:
    try:
        from fog.env.showdown_direct import DEFAULT_SHOWDOWN_ROOT
    except Exception:  # noqa: BLE001
        return False
    return shutil.which("node") is not None and (
        DEFAULT_SHOWDOWN_ROOT / "dist" / "sim" / "index.js"
    ).is_file()


needs_showdown = pytest.mark.skipif(
    not _showdown_ready(), reason="requires Node and the pinned local Pokemon Showdown build"
)
try:
    import poke_engine  # noqa: F401

    HAVE_ENGINE = True
except ImportError:
    HAVE_ENGINE = False
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="requires poke_engine")


# --------------------------------------------------------------------- selection-script hooks
class _ChooseOnly:
    def __init__(self, value):
        self.value = value
        self.finished = 0

    def choose(self, battle):
        return self.value

    def _battle_finished(self, battle):
        self.finished += 1


def test_controller_factory_wraps_choose_only_controllers():
    make = load_controller_factory("tests.test_showdown_engine_mcts:_ChooseOnly", {"value": "x"})
    controller = make()
    assert isinstance(controller, ChooseAdapter)
    assert controller.act(None) == "x"
    controller._battle_finished(None)
    assert controller.inner.finished == 1
    assert not hasattr(controller, "take_game_counters")
    with pytest.raises(ValueError):
        load_controller_factory("no_colon")


def test_sequential_stop_uses_block_means_and_caps_min_blocks():
    result = MatchResult(label="x", a="a", b="b", requested_games=40)
    stop = sequential_stop(100, 10)
    result.game_scores = [1.0, 1.0, 1.0, 0.0] * 2
    assert not stop(result)
    assert result.sequential["decision"] == "continue"
    assert result.sequential["min_blocks"] == 10
    result.game_scores = [1.0, 1.0, None, 0.0] * 10
    assert result.block_means()[0] == pytest.approx(2 / 3)
    assert stop(result)
    assert result.sequential["decision"] in ("accept_h1", "max_reached")


def test_ledger_row_reports_counters():
    result = MatchResult(label="m vs s", a="m", b="s", requested_games=4, completed_games=4,
                         a_wins=3, b_wins=1, game_scores=[1.0, 1.0, 0.0, 1.0])
    result.game_counters = [{"a": {"decisions": 10, "decision_seconds": 2.0, "fallbacks": 1}}]
    row = ledger_row(result.as_dict(), backend="showdown_direct", commit="abc", command="cmd",
                     date="2026-01-01")
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[:5] == ["2026-01-01", "m", "s", "showdown_direct", "4"]
    assert cells[5].startswith("0.750 [")
    assert cells[6:9] == ["0.200", "1/10", "`abc`"]


# ---------------------------------------------------------------------- request overlay (engine)
@needs_engine
def test_request_overlay_forces_switch_and_disables_unoffered_moves():
    import poke_engine as pe

    from fog.search.engine_mcts import apply_request_overlay

    def mon(name):
        return pe.Pokemon(id=name, level=100, types=("normal", "typeless"), hp=300, maxhp=300,
                          ability="none", item="none",
                          moves=[pe.Move(id="tackle", pp=10), pe.Move(id="protect", pp=10)])

    state = pe.State(side_one=pe.Side(pokemon=[mon("snorlax"), mon("blissey")]),
                     side_two=pe.Side(pokemon=[mon("chansey"), mon("ditto")]))

    class View:
        force_switch = True
        available_moves = []

    forced = apply_request_overlay(state, View(), "p1")
    result = pe.monte_carlo_tree_search(forced, iterations=50)
    assert {entry.move_choice for entry in result.side_one} == {"switch blissey"}

    class Move:
        id = "protect"

    class View2:
        force_switch = False
        available_moves = [Move()]

    limited = apply_request_overlay(state, View2(), "p1")
    moves = {m.id: m.disabled for m in limited.side_one.pokemon[0].moves}
    assert moves == {"tackle": True, "protect": False}
    assert state.side_one.pokemon[0].moves[0].disabled is False  # input untouched

    class View3(View2):
        trapped = True
        available_switches = []

    trapped = apply_request_overlay(state, View3(), "p1")
    result = pe.monte_carlo_tree_search(trapped, iterations=50)
    assert not any(e.move_choice.startswith("switch") for e in result.side_one)


# --------------------------------------------------------------------- real Showdown games
@needs_showdown
@needs_engine
def test_engine_mcts_plays_only_legal_actions_on_showdown():
    from fog.env.actions import action_legality_mask
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import play_local_battle
    from fog.env.teams import load_pool, to_packed
    from fog.eval.showdown_direct_selection import _options
    from fog.search.engine_mcts import ShowdownEngineMCTS, showdown_controller

    teams = load_pool(REPO / "data" / "teams" / "pool_b")
    packed = [to_packed(team) for team in teams]
    schedule = matched_schedule(len(teams), 4, seed=3)[:2]
    controller = showdown_controller(k=2, iterations=20, seed=0)

    chosen: list[tuple[np.ndarray, int]] = []
    original = ShowdownEngineMCTS.choose

    def recording_choose(self, battle, tracker, role="p1", fallback=None):
        action = original(self, battle, tracker, role, fallback)
        chosen.append((np.asarray(action_legality_mask(battle), dtype=bool), action))
        return action

    controller.search.choose = recording_choose.__get__(controller.search)
    with ShowdownDirectClient() as client:
        for index, game in enumerate(schedule):
            p1, p2 = ((controller, "simpleheuristics") if game.p1_is_a
                      else ("simpleheuristics", controller))
            result = play_local_battle(
                client, _options("gen9ou", packed, game), p1, p2, seed=game.controller_seed,
                battle_tag=f"battle-mcts-test-{index}", max_turns=60,
            )
            assert result.turns > 0
            counters = controller.take_game_counters()
            assert counters["decisions"] > 0
            assert counters["rejected"] == 0
            assert counters["order_errors"] == 0
            assert counters["decisions"] == (
                counters["searched"] + counters["single_option"] + counters["fallbacks"]
            )
    assert chosen and all(mask[action] for mask, action in chosen)
    assert controller.search.unmapped_choices == {}
