from types import SimpleNamespace

import numpy as np
import pytest

from fog.training.teacher_replay import mapping_state, preview_choice, public_trace, shard_arrays, teacher_roles


@pytest.mark.parametrize("native_p1", [False, True])
def test_only_strong_roles_supply_policy_targets(native_p1):
    record = {"schedule": {"p1_is_a": native_p1}, "opponent": "learner"}
    native = "p1" if native_p1 else "p2"
    assert teacher_roles(record) == {native: "kakuna34"}
    record["opponent"] = "foulplay"
    assert set(teacher_roles(record).values()) == {"kakuna34", "foulplay"}
    record["opponent"] = "kakuna"
    assert teacher_roles(record) == {"p1": "kakuna34", "p2": "kakuna34"}


def test_unknown_teacher_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        teacher_roles({"schedule": {"p1_is_a": True}, "opponent": "other"})


def test_mapping_reads_only_the_acting_players_identities():
    source = {"player_active_pokemon": {"moves": [{"name": "protect"}]},
              "available_switches": [{"name": "hoopaunbound", "base_species": "hoopa"}]}
    value = mapping_state(source)
    assert isinstance(value, SimpleNamespace)
    assert value.player_active_pokemon.moves[0].name == "protect"
    assert value.available_switches[0].name == "hoopaunbound"


def test_preview_preserves_form_and_original_reserve_order():
    record = {"perspectives": {"p1": {"states": [{"player_active_pokemon": {"name": "hoopaunbound", "base_species": "hoopa"}}]}}}
    request = {"side": {"pokemon": [{"details": name} for name in
               ("Dragonite, F", "Hoopa-Unbound", "Kingambit, M", "Slowking-Galar, F", "Gliscor, M", "Corviknight, M")]}}
    assert preview_choice(record, "p1", request) == "team 213456"
    request["side"]["pokemon"][0]["details"] = "Hoopa-Unbound"
    with pytest.raises(ValueError, match="ambiguous"):
        preview_choice(record, "p1", request)


def test_trace_normalization_preserves_every_nonclock_event():
    lines = ["|t:|123", "|-damage|p1a: x|32/100", "|move|p2a: y|Protect", "|win|One"]
    assert public_trace(lines) == lines[1:]


def test_shard_padding_keeps_recent_history_and_does_not_change_labels():
    from fog.encoding.dims import DIMS

    rows = [{"history": np.arange(i + 1, dtype=np.float32)[:, None], "action": np.int64(i)} for i in range(2)]
    arrays = shard_arrays(rows)
    assert arrays["history"].shape == (2, DIMS.history_cap, 1)
    assert arrays["history"][1, -2:, 0].tolist() == [0, 1]
    assert arrays["action"].tolist() == [0, 1]
