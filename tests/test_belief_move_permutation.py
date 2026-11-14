"""The vectorised hidden-move shuffle must do what the per-row loop did."""
from __future__ import annotations

import itertools
from collections import Counter

import torch

from fog.models import belief as belief_module
from fog.models.belief import BeliefModel, N_ATTRS, N_SLOTS


class _Stub:
    sizes = {"move": 50}


def _reference(truth, revealed, generator):
    """The loop this function replaced, kept here as the specification."""
    out = truth.clone()
    for row in range(len(out)):
        for slot in range(N_SLOTS):
            flats = list(belief_module._SLOT_MOVE_FLATS[slot])
            hidden = [p for p in flats if not bool(revealed[row, p])]
            valid = [p for p in hidden if 0 <= int(out[row, p]) < _Stub.sizes["move"]]
            if len(valid) <= 1:
                continue
            perm = torch.randperm(len(valid), generator=generator)
            out[row, valid] = out[row, valid].clone()[perm]
    return out


def _case(rows: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    truth = torch.randint(-1, 60, (rows, N_ATTRS), generator=generator)
    revealed = torch.rand(rows, N_ATTRS, generator=generator) < 0.4
    return truth, revealed


def _shuffle(truth, revealed, seed):
    generator = torch.Generator().manual_seed(seed)
    return BeliefModel._permute_hidden_moves(_Stub(), truth, revealed, "random", generator)


def test_only_hidden_valid_moves_move_and_each_slot_keeps_its_multiset():
    truth, revealed = _case(256, 0)
    out = _shuffle(truth, revealed, 1)
    move_positions = torch.tensor(belief_module._SLOT_MOVE_FLATS)
    movable = torch.zeros_like(revealed)
    movable[:, move_positions.flatten()] = True
    movable &= ~revealed & (truth >= 0) & (truth < _Stub.sizes["move"])
    assert torch.equal(out[~movable], truth[~movable])
    assert torch.equal(
        out[:, move_positions].sort(dim=-1).values, truth[:, move_positions].sort(dim=-1).values
    )
    assert not torch.equal(out, truth)
    assert torch.equal(
        BeliefModel._permute_hidden_moves(_Stub(), truth, revealed, "canonical", None), truth
    )


def test_every_arrangement_is_equally_likely_and_matches_the_loop():
    # One row, first slot: columns 0 and 2 movable with distinct values, column 1 revealed,
    # column 3 an out-of-range value. Two arrangements for the pair; a second slot has three
    # movable columns, so six arrangements.
    first = list(belief_module._SLOT_MOVE_FLATS[0])
    second = list(belief_module._SLOT_MOVE_FLATS[1])
    truth = torch.full((1, N_ATTRS), -1)
    revealed = torch.zeros(1, N_ATTRS, dtype=torch.bool)
    truth[0, first] = torch.tensor([3, 7, 5, 55])
    revealed[0, first[1]] = True
    truth[0, second] = torch.tensor([11, 12, 13, -1])
    draws = 6000
    new = Counter()
    old = Counter()
    generator = torch.Generator().manual_seed(4)
    for seed in range(draws):
        out = _shuffle(truth, revealed, seed)
        new[tuple(out[0, first + second].tolist())] += 1
        ref = _reference(truth, revealed, generator)
        old[tuple(ref[0, first + second].tolist())] += 1
    expected = {
        (a, 7, b, 55, *rest, -1)
        for a, b in ((3, 5), (5, 3))
        for rest in itertools.permutations((11, 12, 13))
    }
    assert set(new) == expected == set(old)
    share = draws / len(expected)
    for counts in (new, old):
        assert all(abs(count - share) < 5 * share**0.5 for count in counts.values())
