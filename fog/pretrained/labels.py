"""Offline submitted-set labels, kept outside every player decision payload."""

from __future__ import annotations

import numpy as np

from fog.constants.schema import NO_ITEM, RARE_SPECIES, SET_MOVE_COLS
from fog.encoding.reveals import align_truth_to_views, opponent_views_from_battle, tokens_from_views


def exact_team_tokens(paste, tables):
    from fog.env.team_truth import truth_tokens_from_paste

    truth = truth_tokens_from_paste(paste, tables)
    occupied = truth[:, 0] >= 0
    # Omission in an actual submitted paste means no held item, rather than
    # an unobserved replay field. Other unrepresented/default fields stay masked.
    truth[occupied & (truth[:, 1] < 0), 1] = tables.item_index(NO_ITEM)
    return truth


def submitted_labels(battle, submitted: np.ndarray, tables) -> dict:
    public = tokens_from_views(opponent_views_from_battle(battle, tables))
    truth = np.full_like(public, -1)
    rare = tables.species_index(RARE_SPECIES)
    for slot, species in enumerate(public[:, 0]):
        matches = np.flatnonzero(submitted[:, 0] == species)
        if species != rare and species >= 0 and len(matches) == 1:
            truth[slot] = submitted[matches[0]]
    aligned, consistent = align_truth_to_views(truth, public)
    verified = aligned >= 0
    # Illusion/Transform can publicly attribute moves to a different submitted set.
    # Keep that evidence in the input; do not fabricate matching submitted labels.
    verified[~consistent, :] = False
    for column in (0, 1, 2, 14):
        known = public[:, column] >= 0
        if column == 1:
            known &= public[:, column] != tables.item_index(NO_ITEM)
        verified[:, column] &= ~known | (public[:, column] == truth[:, column])
    for slot in range(6):
        # Alignment may insert an observed off-table/generated move. Only original
        # submitted moves may become verified autoregressive teacher facts.
        original = {move for move in truth[slot, list(SET_MOVE_COLS)] if move >= 0}
        for column in SET_MOVE_COLS:
            verified[slot, column] &= aligned[slot, column] in original
    return {"reveals": public.tolist(), "truth": aligned.tolist(), "verified": verified.tolist()}
