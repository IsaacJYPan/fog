from copy import deepcopy
from types import SimpleNamespace as NS

from fog.pretrained.requests import reconcile_own_roster


def test_illusion_ghost_is_removed_and_actual_active_identity_is_restored():
    def mon(name, hp=1):
        return NS(
            name=name,
            hp_pct=hp,
            status="nostatus",
            item="noitem",
            ability="illusion",
            atk_boost=0,
            effect="noeffect",
        )

    ghost = mon("zoroarkhisui")
    disguise = mon("luxray")
    disguise.atk_boost = 2
    state = NS(player_active_pokemon=disguise, available_switches=[ghost, mon("pelipper")])
    request = {
        "side": {
            "pokemon": [
                {"details": "Zoroark-Hisui", "active": True, "condition": "60/100"},
                {"details": "Luxray", "active": False, "condition": "0 fnt"},
                {"details": "Pelipper", "active": False, "condition": "80/100"},
            ]
        }
    }

    def convert(member):
        result = mon(member["details"].lower().replace("-", ""))
        result.hp_pct = 0.6 if member["active"] else 0.8
        return result

    result = reconcile_own_roster(state, request, convert)
    assert result.player_active_pokemon.name == "zoroarkhisui"
    assert result.player_active_pokemon.atk_boost == 2
    assert [p.name for p in result.available_switches] == ["pelipper"]
    assert result.available_switches[0].hp_pct == 0.8
    assert ghost.hp_pct == 1


def test_authoritative_revived_reserve_overrides_stale_faint_condition():
    stale = NS(name="dragonite", hp_pct=0, status="fnt", item="noitem", ability="multiscale")
    state = NS(player_active_pokemon=NS(name="pawmot"), available_switches=[stale])
    request = {
        "side": {
            "pokemon": [
                {"details": "Pawmot", "active": True, "condition": "100/100"},
                {"details": "Dragonite", "active": False, "condition": "50/100"},
            ]
        }
    }

    def convert(member):
        result = deepcopy(stale)
        result.hp_pct, result.status = 0.5, "nostatus"
        return result

    result = reconcile_own_roster(state, request, convert)
    assert result.available_switches[0].hp_pct == 0.5
    assert result.available_switches[0].status == "nostatus"
