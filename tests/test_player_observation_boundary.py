from types import SimpleNamespace

import numpy as np
import pytest

from fog.encoding.player_observation import DecisionPhase, decision_phase_from_battle


def test_decision_phase_uses_only_visible_request() -> None:
    assert decision_phase_from_battle(
        SimpleNamespace(last_request={"teamPreview": True})
    ) == DecisionPhase.TEAM_PREVIEW
    assert decision_phase_from_battle(
        SimpleNamespace(last_request={"forceSwitch": [True]})
    ) == DecisionPhase.FORCED_SWITCH
    assert decision_phase_from_battle(
        SimpleNamespace(last_request={"active": [{"moves": []}]})
    ) == DecisionPhase.NORMAL_MOVE


def test_hidden_engine_attributes_cannot_change_decision_phase() -> None:
    visible = {"active": [{"moves": [{"id": "tackle"}]}]}
    first = SimpleNamespace(last_request=visible, rng_state=1, hidden_duration=2)
    second = SimpleNamespace(last_request=visible, rng_state=999, hidden_duration=7)
    assert decision_phase_from_battle(first) == decision_phase_from_battle(second)


def test_unrevealed_opponent_item_does_not_change_player_observation() -> None:
    from fog.env.engine_battle import HAVE_ENGINE

    if not HAVE_ENGINE:
        pytest.skip("poke-engine not installed")
    from fog.encoding.observation import Observation
    from fog.env.engine_battle import EngineBattle

    mine = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave
"""
    opponent_template = """Garchomp @ {item}
Ability: Rough Skin
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Swords Dance
- Stealth Rock
"""
    first = EngineBattle(
        mine,
        opponent_template.format(item="Leftovers"),
        seed=441,
        battle_tag="boundary-item-a",
    )
    second = EngineBattle(
        mine,
        opponent_template.format(item="Choice Band"),
        seed=441,
        battle_tag="boundary-item-b",
    )
    first_obs = Observation.from_battle(
        first.view("p1"), tracker=first.trackers["p1"]
    )
    second_obs = Observation.from_battle(
        second.view("p1"), tracker=second.trackers["p1"]
    )

    np.testing.assert_array_equal(first_obs.entities, second_obs.entities)
    np.testing.assert_array_equal(first_obs.entity_ids, second_obs.entity_ids)
    np.testing.assert_array_equal(first_obs.field, second_obs.field)
    np.testing.assert_array_equal(first_obs.history, second_obs.history)
    np.testing.assert_array_equal(first_obs.legality_mask, second_obs.legality_mask)


def test_hidden_effect_duration_is_not_copied_from_simulator_object() -> None:
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, L, _fill_mon_record

    base = {
        "base_stats": {},
        "types": [],
        "weight": None,
        "current_hp_fraction": 1.0,
        "fainted": False,
        "status": None,
        "boosts": {},
        "terastallized": False,
        "tera_type": None,
    }
    first = SimpleNamespace(**base, effects={"partiallytrapped": 1})
    second = SimpleNamespace(**base, effects={"partiallytrapped": 5})
    first_row = np.zeros(ENTITY_RAW_DIM, dtype=np.float32)
    second_row = np.zeros(ENTITY_RAW_DIM, dtype=np.float32)
    _fill_mon_record(first_row, first, None)
    _fill_mon_record(second_row, second, None)

    np.testing.assert_array_equal(first_row, second_row)
    assert not first_row[L.vol_counters : L.vol_counters + DIMS.volatile_counters].any()
