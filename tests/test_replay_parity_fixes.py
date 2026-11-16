"""Public prefixes must reconstruct own PP and replacement choices causally."""
import numpy as np

from fog.constants.tables import TABLES
from fog.encoding.entities import L
from fog.scripts.parse_replays import parse_replay


def fixture(*, pivot=False, pressure=False):
    from fog.env.team_truth import truth_tokens_from_paste

    own = '''Garchomp @ Leftovers
Ability: Rough Skin
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
Tera Type: Ground
- Earthquake
- U-turn
- Protect
- Swords Dance

Dragonite @ Heavy-Duty Boots
Ability: Multiscale
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
Tera Type: Normal
- Extreme Speed
- Dragon Dance
- Earthquake
- Roost
'''
    foe = '''Corviknight @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
Tera Type: Flying
- Roost
- Brave Bird
- U-turn
- Defog
'''
    log = '''|gametype|singles
|player|p1|SideOne|
|player|p2|SideTwo|
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Garchomp, M|
|poke|p1|Dragonite, M|
|poke|p2|Corviknight, M|
|teampreview
|start
|switch|p1a: Garchomp|Garchomp, M|100/100
|switch|p2a: Corviknight|Corviknight, M|100/100
'''
    if pressure:
        log += '|-ability|p2a: Corviknight|Pressure\n'
    log += '|turn|1\n|move|p1a: Garchomp|' + ('U-turn' if pivot else 'Earthquake') + '|p2a: Corviknight\n'
    if pivot:
        log += '|switch|p1a: Dragonite|Dragonite, M|100/100\n'
    else:
        log += '|move|p2a: Corviknight|Brave Bird|p1a: Garchomp\n|-damage|p1a: Garchomp|0 fnt\n|faint|p1a: Garchomp\n|upkeep\n|switch|p1a: Dragonite|Dragonite, M|100/100\n'
    log += '|turn|2\n|move|p1a: Dragonite|Earthquake|p2a: Corviknight\n|win|SideOne\n'
    return {"log": log, "players": ["SideOne", "SideTwo"]}, {
        "p1": truth_tokens_from_paste(own, TABLES),
        "p2": truth_tokens_from_paste(foe, TABLES),
    }


def own_rows(pivot=False, pressure=False):
    replay, sheets = fixture(pivot=pivot, pressure=pressure)
    return [r for r in parse_replay(replay, private_sets=sheets)["rows"] if r["role"] == "p1"]


def test_own_pp_is_full_initially_and_counts_public_pressure():
    from poke_env.battle import Move

    for pressure in (False, True):
        rows = own_rows(pressure=pressure)
        assert rows[0]["obs"].entities[0, L.move_pp] == 1
        max_pp = Move("earthquake", gen=9).max_pp
        expected = (max_pp - (2 if pressure else 1)) / max_pp
        assert np.isclose(rows[-1]["obs"].entities[0, L.move_pp], expected)
        assert rows[-1]["obs"].entities[1, L.move_pp] == 1


def test_faint_replacement_is_a_valid_switch_only_row():
    rows = own_rows()
    replacements = [r for r in rows if r["obs"].decision_phase == 1]
    assert len(replacements) == 1
    row = replacements[0]
    assert row["turn"] == 1 and row["action"] == 8
    assert row["policy_valid"]
    assert not row["mask"][:8].any()
    assert row["obs"].entities[0, L.fainted] == 1
    # Live history deliberately excludes the current turn, even at replacement requests.
    assert row["obs"].history_len == 0


def test_pivot_is_a_forced_request_even_while_active_survives():
    replacements = [r for r in own_rows(pivot=True) if r["obs"].decision_phase == 1]
    assert len(replacements) == 1
    assert replacements[0]["policy_valid"]
    assert not replacements[0]["mask"][:8].any()
    assert replacements[0]["obs"].entities[0, L.fainted] == 0


def test_replacement_does_not_overwrite_first_history_action():
    from fog.encoding.dims import DIMS

    rows = own_rows()
    last = rows[-1]["obs"]
    action = last.history[-1, DIMS.hist_snapshot:DIMS.hist_snapshot + DIMS.hist_my_action]
    assert action[0] == 1
    assert action.sum() == 1


def test_live_replacement_keeps_move_and_normal_retry_can_update_it():
    from types import SimpleNamespace

    from fog.encoding.events import record_history_decision
    from fog.encoding.tracker import BattleTracker

    tracker = BattleTracker(my_role="p1")
    battle = SimpleNamespace(force_switch=False)
    record_history_decision(tracker, 3, battle, TABLES, 1)
    record_history_decision(tracker, 3, battle, TABLES, 2)  # rejected normal choice retried
    assert tracker.my_actions[3] == 2
    battle.force_switch = True
    record_history_decision(tracker, 3, battle, TABLES, 8)
    assert tracker.my_actions[3] == 2
    # A replacement may be the first recorded choice of a turn (for example turn zero).
    record_history_decision(tracker, 0, battle, TABLES, 9)
    assert tracker.my_actions[0] == 9


def test_own_preview_and_original_item_match_request_representation():
    replay, sheets = fixture()
    replay["log"] = replay["log"].replace('|turn|1', '|-enditem|p1a: Garchomp|Leftovers\n|turn|1')
    rows = parse_replay(replay, private_sets=sheets)["rows"]
    first = rows[0]["obs"]
    assert first.entities[1, L.hp] == 1
    assert first.entities[:2, L.tera_type:L.tera_type + 19].sum() == 0
    assert first.entity_ids[0, 1] == TABLES.item_index("<NO_ITEM>")
    assert first.entity_ids[0, 15] == TABLES.item_index("leftovers")


def test_serialized_history_keeps_live_window_and_decision_metadata():
    from fog.encoding.dims import DIMS
    from fog.scripts.parse_replays import rows_to_arrays

    rows = own_rows()
    arrays = rows_to_arrays(rows)
    assert arrays["history"].shape[1] == DIMS.history_cap
    np.testing.assert_array_equal(arrays["row_turn"], [r["turn"] for r in rows])
    assert (arrays["row_role"] == 1).all()


def test_trick_item_reveal_updates_current_item_without_changing_original():
    replay, sheets = fixture()
    replay["log"] = replay["log"].replace('|turn|1', '|-item|p1a: Garchomp|Assault Vest|[from] move: Trick\n|turn|1')
    first = parse_replay(replay, private_sets=sheets)["rows"][0]["obs"]
    assert first.entity_ids[0, 1] == TABLES.item_index("assaultvest")
    assert first.entity_ids[0, 15] == TABLES.item_index("leftovers")


def test_choice_lock_is_reconstructed_from_own_item_and_public_move():
    replay, sheets = fixture()
    sheets["p1"][0, 1] = TABLES.item_index("choiceband")
    replay["log"] = replay["log"].replace('|-damage|p1a: Garchomp|0 fnt\n|faint|p1a: Garchomp', '|-damage|p1a: Garchomp|80/100')
    replay["log"] = replay["log"].replace('|switch|p1a: Dragonite|Dragonite, M|100/100\n', '')
    replay["log"] = replay["log"].replace('|move|p1a: Dragonite|Earthquake', '|move|p1a: Garchomp|Earthquake')
    rows = [r for r in parse_replay(replay, private_sets=sheets)["rows"] if r["role"] == "p1"]
    mask = rows[-1]["mask"]
    assert mask[0] and mask[4] and mask[8]
    assert not mask[[1, 2, 3, 5, 6, 7]].any()


def test_nickname_alias_keeps_original_private_item_after_consumption():
    replay, sheets = fixture()
    replay["log"] = replay["log"].replace('p1a: Garchomp', 'p1a: Chomp')
    replay["log"] = replay["log"].replace('|turn|1', '|-enditem|p1a: Chomp|Leftovers\n|turn|1')
    first = parse_replay(replay, private_sets=sheets)["rows"][0]["obs"]
    assert first.entity_ids[0, 15] == TABLES.item_index("leftovers")


def test_assault_vest_blocks_status_moves_after_trick():
    replay, sheets = fixture()
    replay["log"] = replay["log"].replace('|turn|1', '|-item|p1a: Garchomp|Assault Vest|[from] move: Trick\n|turn|1')
    first = parse_replay(replay, private_sets=sheets)["rows"][0]
    assert not first["mask"][[2, 3, 6, 7]].any()


def test_tera_ogerpon_replacement_label_uses_submitted_species():
    replay, sheets = fixture()
    sheets["p1"][1, 0] = TABLES.species_index("ogerponwellspring")
    replay["log"] = replay["log"].replace('|poke|p1|Dragonite, M|', '|poke|p1|Ogerpon-Wellspring, F|')
    replay["log"] = replay["log"].replace('p1a: Dragonite', 'p1a: Ogerpon').replace('|Dragonite, M|', '|Ogerpon-Wellspring-Tera, F, tera:Water|')
    rows = parse_replay(replay, private_sets=sheets)["rows"]
    replacements = [r for r in rows if r["role"] == "p1" and r["obs"].decision_phase == 1]
    assert len(replacements) == 1
    assert replacements[0]["policy_valid"] and replacements[0]["action"] == 8


def test_normal_struggle_phase_survives_both_dataset_loaders(tmp_path):
    import polars as pl
    import torch

    from fog.scripts.parse_replays import rows_to_arrays
    from fog.training.dataset import ReplayDataset, StreamingReplayDataset
    from fog.models.shared_latent import SharedBattleLatent

    rows = own_rows()[:1]
    rows[0]["mask"][:] = False
    rows[0]["mask"][[8, 13]] = True
    rows[0]["action"] = 8
    arrays = rows_to_arrays(rows)
    shard = tmp_path / 'game.npz'
    manifest = tmp_path / 'manifest.parquet'
    np.savez_compressed(shard, **arrays)
    pl.DataFrame([{'id': 'fixture', 'shard': str(shard), 'n_rows': 1}]).write_parquet(manifest)
    for loader in (ReplayDataset, StreamingReplayDataset):
        batch = next(loader(manifest).batches(1, shuffle=False))
        phase = SharedBattleLatent._decision_phase(batch, batch['legality_mask']).long().argmax(-1)
        assert phase.item() == 0  # Struggle and a voluntary switch, not a replacement.
    # Old shards retain the model's old fallback exactly.
    del arrays['decision_phase']
    np.savez_compressed(shard, **arrays)
    legacy = next(ReplayDataset(manifest).batches(1, shuffle=False))
    assert torch.equal(legacy['decision_phase'], torch.tensor([1]))
