from types import SimpleNamespace as NS

import pytest

from fog.pretrained.actions import native_choices, request_choice


def state():
    return NS(
        player_active_pokemon=NS(
            moves=[NS(name="U-turn"), NS(name="Earthquake"), NS(name="Roost")]
        ),
        available_switches=[
            NS(name="Dragonite", base_species="Dragonite"),
            NS(name="Alomomola", base_species="Alomomola"),
        ],
    )


def request():
    return {
        "side": {
            "pokemon": [
                {"details": "Dragonite, L100", "condition": "100/100", "active": False},
                {"details": "Gliscor, L100", "condition": "100/100", "active": True},
                {"details": "Alomomola, L100", "condition": "100/100", "active": False},
            ]
        },
        "active": [
            {
                "moves": [
                    {"id": "uturn", "pp": 5},
                    {"id": "earthquake", "pp": 5, "disabled": True},
                    {"id": "roost", "pp": 0},
                ],
                "canTerastallize": "Water",
            }
        ],
    }


def test_disabled_moves_retain_native_indices_and_switches_use_request_slots():
    assert native_choices(state(), request()) == {
        2: "move 1",
        11: "move 1 terastallize",
        4: "switch 3",
        5: "switch 1",
    }


def test_forced_switch_trapping_and_revival():
    data = request()
    data["forceSwitch"] = [True]
    assert native_choices(state(), data) == {4: "switch 3", 5: "switch 1"}
    data = request()
    data["active"][0]["trapped"] = True
    assert native_choices(state(), data) == {2: "move 1", 11: "move 1 terastallize"}
    data = request()
    data["side"]["pokemon"][0]["condition"] = "0 fnt"
    data["side"]["pokemon"][0]["reviving"] = True
    data["side"]["pokemon"][2]["condition"] = "0 fnt"
    assert native_choices(state(), data)[4] == "switch 3"


def test_struggle_recharge_and_no_invalid_tera_help():
    data = request()
    data["active"][0] = {"moves": [{"id": "struggle", "pp": 1}]}
    assert native_choices(state(), data) == {0: "move 1", 4: "switch 3", 5: "switch 1"}
    data["active"][0]["moves"][0]["id"] = "recharge"
    assert 9 not in native_choices(state(), data)
    data["side"]["pokemon"].append(dict(data["side"]["pokemon"][0]))
    with pytest.raises(ValueError, match="maps to 2"):
        native_choices(state(), data)


def test_external_move_name_canonicalization_keeps_request_order_and_tera():
    assert request_choice("/choose move uturn terastallize", request()) == "move 1 terastallize"
    assert request_choice("/choose switch 3", request()) == "switch 3"
    assert request_choice("move 2", request()) == "move 2"
    with pytest.raises(ValueError, match="exactly one"):
        request_choice("/choose move missing", request())
