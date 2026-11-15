import numpy as np
import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS, L
from fog.encoding.events import H_MY_MOVE, H_START_MY_SPECIES
from fog.eval.replay_slot_order_check import (
    _policy_statistic,
    _value_statistic,
    game_resampled_intervals,
    lead_order_summary,
    not_yet_moved,
    shuffle_move_slots,
)


def _batch(rows: int = 64) -> dict:
    generator = torch.Generator().manual_seed(3)
    entity_ids = torch.randint(0, 50, (rows, 12, N_ID_FIELDS), generator=generator)
    # Distinct move ids per row make "the label follows its move" checkable.
    entity_ids[:, :, 3:7] = torch.arange(4).view(1, 1, 4) + 100
    entities = torch.rand(rows, 12, ENTITY_RAW_DIM, generator=generator)
    legality = torch.rand(rows, 14, generator=generator) > 0.3
    action = torch.randint(0, 13, (rows,), generator=generator)
    return {
        "entity_ids": entity_ids,
        "entities": entities,
        "active_idx": torch.stack(
            [torch.randint(0, 6, (rows,), generator=generator), torch.full((rows,), 6)], dim=1
        ),
        "legality_mask": legality,
        "legal_provenance": legality.to(torch.uint8),
        "action_feature_provenance": torch.randint(0, 3, (rows, 14), generator=generator),
        "own_set_provenance": torch.randint(0, 3, (rows, 6, 15), generator=generator),
        "action_features": torch.rand(rows, 14, 3, generator=generator),
        "action": action,
    }


def test_shuffle_moves_label_legality_and_pp_together() -> None:
    batch = _batch()
    shuffled = shuffle_move_slots(batch, torch.Generator().manual_seed(0))
    rows = torch.arange(batch["action"].shape[0])
    mine = batch["active_idx"][:, 0]
    old_ids = batch["entity_ids"][rows, mine, 3:7]
    new_ids = shuffled["entity_ids"][rows, mine, 3:7]
    assert not torch.equal(old_ids, new_ids)
    assert torch.equal(old_ids.sort(dim=1).values, new_ids.sort(dim=1).values)
    move = batch["action"] < 8
    old_slot, new_slot = batch["action"] % 4, shuffled["action"] % 4
    # The labelled move, its PP and its legality bits are the same before and after.
    assert torch.equal(old_ids[rows, old_slot][move], new_ids[rows, new_slot][move])
    old_pp = batch["entities"][rows, mine][:, L.move_pp : L.move_pp + 4]
    new_pp = shuffled["entities"][rows, mine][:, L.move_pp : L.move_pp + 4]
    assert torch.equal(old_pp[rows, old_slot][move], new_pp[rows, new_slot][move])
    old_flags = batch["entities"][rows, mine][:, L.epistemic + 2 : L.epistemic + 6]
    new_flags = shuffled["entities"][rows, mine][:, L.epistemic + 2 : L.epistemic + 6]
    assert torch.equal(old_flags[rows, old_slot][move], new_flags[rows, new_slot][move])
    for key in ("action_feature_provenance", "action_features", "legal_provenance"):
        assert torch.equal(
            batch[key][rows, batch["action"]][move],
            shuffled[key][rows, shuffled["action"]][move],
        )
    assert torch.equal(
        batch["own_set_provenance"][rows, mine, 3 + old_slot][move],
        shuffled["own_set_provenance"][rows, mine, 3 + new_slot][move],
    )
    assert torch.equal(
        batch["legality_mask"][rows, batch["action"]][move],
        shuffled["legality_mask"][rows, shuffled["action"]][move],
    )
    assert torch.equal(batch["action"] >= 4, shuffled["action"] >= 4)
    # Switch rows, the other Pokemon and the non-move entries are untouched.
    assert torch.equal(batch["action"][~move], shuffled["action"][~move])
    assert torch.equal(batch["legality_mask"][:, 8:], shuffled["legality_mask"][:, 8:])
    assert torch.equal(batch["entity_ids"][:, 6:], shuffled["entity_ids"][:, 6:])
    assert torch.equal(
        batch["legality_mask"][:, :4].sum(1), shuffled["legality_mask"][:, :4].sum(1)
    )


def test_not_yet_moved_reads_only_real_history_rows() -> None:
    n_moves = 900
    ids = torch.full((3, 4, DIMS.hist_id_fields), 0, dtype=torch.long)
    ids[..., H_MY_MOVE] = n_moves
    entity_ids = torch.zeros(3, 12, N_ID_FIELDS, dtype=torch.long)
    entity_ids[:, 0, 0] = 7
    # Row 0: the acting species moved on a real turn. Row 1: only another species moved.
    # Row 2: a move by this species sits in padding and must be ignored.
    ids[0, 3, H_START_MY_SPECIES], ids[0, 3, H_MY_MOVE] = 7, 12
    ids[1, 3, H_START_MY_SPECIES], ids[1, 3, H_MY_MOVE] = 8, 12
    ids[2, 0, H_START_MY_SPECIES], ids[2, 0, H_MY_MOVE] = 7, 12
    batch = {
        "history_ids": ids,
        "history_len": torch.tensor([2, 2, 1]),
        "active_idx": torch.tensor([[0, 6]] * 3),
        "entity_ids": entity_ids,
    }
    assert not_yet_moved(batch, n_moves).tolist() == [False, True, True]


def test_intervals_resample_whole_games_and_keep_row_weighting():
    # Each game's rows are perfectly correlated. Resampling rows would falsely narrow this CI.
    totals = np.array([[100, 0, 0, 0, 0], [100, 100, 100, 100, 100]], dtype=float)
    intervals = game_resampled_intervals(totals, _policy_statistic, samples=1000, seed=4)
    assert intervals["top1_stored"] == [0.0, 1.0]
    assert intervals == game_resampled_intervals(
        totals, _policy_statistic, samples=1000, seed=4,
    )
    unequal = totals.copy()
    unequal[1] *= 2
    assert _policy_statistic(unequal.sum(0))["top1_stored"] == 2 / 3


def test_value_totals_match_calibration_and_do_not_average_bin_ece():
    from fog.eval.replay_holdout_report import value_calibration

    # Errors in the same reliability bin cancel after combining games, before taking abs.
    values = np.array([-0.1, -0.1, 0.6, 0.9])
    outcome = np.array([-1, 1, 1, -1])
    p, target = (values + 1) / 2, (outcome + 1) / 2
    total = np.zeros(32)
    total[0], total[1] = len(p), ((p - target) ** 2).sum()
    bins = np.minimum((p * 10).astype(int), 9)
    np.add.at(total, 2 + bins, 1)
    np.add.at(total, 12 + bins, p)
    np.add.at(total, 22 + bins, target)
    expected = value_calibration(values, outcome)
    actual = _value_statistic(total)
    np.testing.assert_allclose([actual["brier"], actual["ece"]],
                               [expected["brier"], expected["ece"]])


def test_lead_summary_counts_games_and_explicitly_reports_missing_leads():
    import polars as pl

    manifest = pl.DataFrame({"lead_p1": [0, 2, None], "lead_p2": [0, 0, 1]})
    summary = lead_order_summary(manifest)
    assert summary["p1"]["share_position0"] == 0.5
    assert summary["p1"]["unknown_leads"] == 1
    assert summary["p2"]["share_position0"] == 2 / 3
    assert summary["games_with_both_leads_known"] == 2
    assert summary["share_games_both_leads_position0"] == 0.5
    assert summary["share_games_either_lead_position0"] == 1
