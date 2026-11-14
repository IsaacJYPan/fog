import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS, L
from fog.encoding.events import H_MY_MOVE, H_START_MY_SPECIES
from fog.eval.replay_slot_order_check import not_yet_moved, shuffle_move_slots


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
