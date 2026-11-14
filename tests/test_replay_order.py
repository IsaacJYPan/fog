"""Replay slot randomisation keeps identity, provenance, labels and history together."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np

from fog.constants.tables import TABLES
from fog.encoding.entities import L
from fog.encoding.events import H_MY_MOVE, _OFF_MYACT
from fog.scripts.parse_replays import _PerspectiveState, parse_replay_policy_versions
from fog.training.replay_policy_data import (
    EXACT,
    INFERRED,
    complete_private_team,
    permute_replay_preview,
    replay_lead_positions,
)


def _revealed():
    result = np.full((6, 15), -1, dtype=np.int64)
    result[0, 0] = TABLES.species_index("garchomp")
    result[0, 3:7] = [TABLES.move_index(m) for m in (
        "earthquake", "stealthrock", "protect", "stoneedge",
    )]
    return result


def test_move_order_is_seeded_and_uniform():
    revealed = _revealed()
    a, pa = complete_private_team(revealed, seed=7)
    b, pb = complete_private_team(revealed, seed=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(pa, pb)
    counts = np.zeros((4, 4), dtype=int)
    for seed in range(1024):
        sheet, provenance = complete_private_team(revealed, seed=seed)
        for old, move in enumerate(revealed[0, 3:7]):
            column = list(sheet[0, 3:7]).index(move)
            counts[old, column] += 1
            assert provenance[0, 3 + column] == EXACT
        assert (sheet[1:] == -1).all()
        assert (provenance[1:] == 0).all()
    assert np.all(np.abs(counts / 1024 - 0.25) < 0.05)


def test_sparse_exact_and_sampled_move_provenance_follow_identity():
    revealed = _revealed()
    revealed[0, [3, 4, 6]] = -1
    exact_move = revealed[0, 5]
    for seed in range(32):
        sheet, provenance = complete_private_team(revealed, seed=seed)
        assert len(set(sheet[0, 3:7])) == 4
        for move, source in zip(sheet[0, 3:7], provenance[0, 3:7]):
            assert source == (EXACT if move == exact_move else INFERRED)


def _replay():
    preview = []
    for role, species in (
        ("p1", ["Garchomp", "Corviknight", "Kingambit", "Iron Valiant", "Great Tusk", "Dragapult"]),
        ("p2", ["Dragapult", "Great Tusk", "Iron Valiant", "Kingambit", "Corviknight", "Garchomp"]),
    ):
        preview.extend(f"|poke|{role}|{name}|" for name in species)
    log = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] OU
|clearpoke
""" + "\n".join(preview) + """
|teampreview
|start
|switch|p1a: Chomp|Garchomp|100/100
|switch|p2a: Pult|Dragapult|100/100
|turn|1
|move|p1a: Chomp|Earthquake|p2a: Pult
|move|p2a: Pult|Shadow Ball|p1a: Chomp
|turn|2
|switch|p1a: Bird|Corviknight|100/100
|move|p2a: Pult|Shadow Ball|p1a: Bird
|turn|3
|move|p1a: Bird|Roost|p1a: Bird
|move|p2a: Pult|Shadow Ball|p1a: Bird
|turn|4
|move|p1a: Bird|Roost|p1a: Bird
|move|p2a: Pult|Shadow Ball|p1a: Bird
|win|Alice
"""
    return {"log": log, "players": ["Alice", "Bob"], "rating": 1600}


def test_preview_shuffle_is_reproducible_and_each_species_reaches_every_position():
    replay = _replay()
    first = permute_replay_preview(replay, replay_id="fixture", seed=0)
    assert first == permute_replay_preview(replay, replay_id="fixture", seed=0)
    assert Counter(first["log"].splitlines()) == Counter(replay["log"].splitlines())
    assert replay_lead_positions(replay) == {"p1": 0, "p2": 0}
    counts = {role: np.zeros(6) for role in ("p1", "p2")}
    for seed in range(1024):
        shuffled = permute_replay_preview(replay, replay_id="fixture", seed=seed)
        for role, position in replay_lead_positions(shuffled).items():
            counts[role][position] += 1
    for count in counts.values():
        assert np.all(np.abs(count / 1024 - 1 / 6) < 0.05)


def test_converted_actions_provenance_and_history_read_the_installed_order():
    replay = _replay()
    expected = {
        "p1": {1: "earthquake", 2: "corviknight", 3: "roost", 4: "roost"},
        "p2": {turn: "shadowball" for turn in range(1, 5)},
    }
    for seed in range(8):
        result = parse_replay_policy_versions(replay, replay_id="fixture", seed=seed, imputations=1)
        assert len(result["rows"]) == 8
        assert result["lead_positions"] == replay_lead_positions(
            permute_replay_preview(replay, replay_id="fixture", seed=seed)
        )
        previous = {}
        for row in result["rows"]:
            obs, action = row["obs"], row["action"]
            assert row["mask"][action] and obs.legality_mask[action]
            assert row["action_feature_provenance"][action] == EXACT
            role, turn = row["role"], row["turn"]
            active = int(obs.active_idx[0])
            assert obs.entities[active, L.is_active] == 1
            if action < 8:
                mid = obs.entity_ids[active, 3 + action % 4]
                assert TABLES.moves["ids"][mid] == expected[role][turn]
                assert row["own_set_provenance"][active, 3 + action % 4] == EXACT
            else:
                bench = [slot for slot in range(6) if slot != active]
                species = obs.entity_ids[bench[action - 8], 0]
                assert TABLES.species["ids"][species] == expected[role][turn]
            if turn > 1:
                prior = previous[role]
                history_action = obs.history[-1, _OFF_MYACT : _OFF_MYACT + 14]
                assert history_action.argmax() == prior["action"]
                assert history_action.sum() == 1
                if prior["action"] < 8:
                    mid = obs.history_ids[-1, H_MY_MOVE]
                    assert TABLES.moves["ids"][mid] == expected[role][turn - 1]
            previous[role] = row


def test_policy_rejects_a_move_dictionary_that_changed_since_the_snapshot():
    state = _PerspectiveState("p1", "Alice", private_set=_revealed())
    state.battle = SimpleNamespace(
        active_pokemon=SimpleNamespace(moves={"earthquake": object()}), turn=1,
    )
    ids = np.zeros((12, 17), dtype=np.int64)
    ids[0, 3] = TABLES.move_index("protect")
    state._pending_obs = SimpleNamespace(entity_ids=ids, active_idx=np.array([0, 6]))
    state._pending_mask = np.ones(14, dtype=bool)
    state._pending_filter_reason = "awaiting_action"
    state.record_action(["", "move", "p1a: Chomp", "Earthquake", "p2a: Pult"])
    assert not state._pending_policy_valid
    assert state._pending_filter_reason == "called_or_unaligned_move"
    assert 1 not in state.tracker.my_actions


def test_policy_rejects_a_switch_identity_that_changed_since_the_snapshot():
    state = _PerspectiveState("p1", "Alice", private_set=_revealed())
    active = SimpleNamespace(species="garchomp")
    target = SimpleNamespace(species="corviknight")
    state.battle = SimpleNamespace(
        active_pokemon=active, team={"p1: Chomp": active, "p1: Bird": target}, turn=1,
    )
    ids = np.zeros((12, 17), dtype=np.int64)
    ids[1, 0] = TABLES.species_index("zoroarkhisui")
    state._pending_obs = SimpleNamespace(entity_ids=ids, active_idx=np.array([0, 6]))
    state._pending_mask = np.ones(14, dtype=bool)
    state._pending_filter_reason = "awaiting_action"
    state.record_action(["", "switch", "p1a: Bird", "Corviknight", "100/100"])
    assert not state._pending_policy_valid
    assert state._pending_filter_reason == "forced_or_unaligned_switch"
    assert 1 not in state.tracker.my_actions


def test_shuffled_policy_conversion_keeps_future_opponent_facts_out_of_inputs():
    replay = _replay()
    revealed = {**replay, "log": replay["log"].replace(
        "|turn|3\n", "|-item|p2a: Pult|Choice Specs\n|turn|3\n",
    )}
    observations = []
    targets = []
    for source in (replay, revealed):
        result = parse_replay_policy_versions(source, replay_id="isolation", seed=7, imputations=1)
        row = next(r for r in result["rows"] if r["role"] == "p1" and r["turn"] == 1)
        observations.append(row["obs"])
        targets.append(row["belief"])
    before, after = observations
    for name in ("entities", "entity_ids", "field", "history", "history_ids", "history_state",
                 "active_idx", "legality_mask", "action_features"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    np.testing.assert_array_equal(before.opp_revealed["tokens"], after.opp_revealed["tokens"])
    # The later public fact is allowed in the training target, but never in turn-one inputs.
    species = TABLES.species_index("dragapult")
    slot = list(targets[1][:, 0]).index(species)
    assert targets[1][slot, 1] == TABLES.item_index("choicespecs")
    assert targets[0][slot, 1] == -1
