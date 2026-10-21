"""Opponent's-view-of-me observation for the self-belief / leakage path.

The self-belief predictor Φ measures how readable I am. Its input is "the opponent's view of me": the
same public board, but with MY hidden attributes masked to what the opponent has actually seen, and
the two sides' entity slots swapped (from the opponent's seat, their team is slots 0-5 and I am the
hidden opponent in slots 6-11).

Frame convention: the mirror is encoded in the SAME frame the backbone sees everywhere else —
"viewer's team = slots 0-5 with is_mine=1, viewer's side = first per-side field block". That means
the swap flips the is_mine bits and the per-side field blocks (hazards/screens/forced-switch/delayed),
so a backbone shared between the policy path and the self-belief path (aux_into_trunk) reuses the
same features for both.

Two entry points:
  * `build_opponent_view_np`  — unbatched numpy dict pair (backbone frag, my_view dict). This is what
    the live player stores into RolloutBuffer frags (contract C3 "opp_view") so the self-belief
    predictor trains inside ppo_update.
  * `build_opponent_view`     — batched-torch convenience wrapper (single observation) used for the
    per-decision frozen-probe leakage measurement.
"""
from __future__ import annotations

import numpy as np
import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import (
    EV_ID_FIELDS,
    N_ID_FIELDS,
    NATURE_ID_FIELD,
    ORIGINAL_ABILITY_ID_FIELD,
    ORIGINAL_ITEM_ID_FIELD,
    L,
)
from fog.encoding.events import H_FIRST_ACTOR


def _downgrade_my_entities(entities: np.ndarray, entity_ids: np.ndarray, my_view_tokens: np.ndarray,
                           unk: dict) -> tuple[np.ndarray, np.ndarray]:
    """Copy my 6 entity records but blank the ids + epistemic flags the opponent hasn't seen.

    my_view_tokens [6,15]: what the opponent has seen of me (-1 = hidden). entities/entity_ids are my
    slots 0-5 only ([6,...]).
    """
    ent = entities.copy()
    ids = entity_ids.copy()
    for j in range(6):
        tok = my_view_tokens[j]
        # ids: species always public (preview); others -> UNK unless revealed
        ids[j, 1] = tok[1] if tok[1] >= 0 else unk["item"]
        ids[j, 2] = tok[2] if tok[2] >= 0 else unk["ability"]
        for k in range(4):
            ids[j, 3 + k] = tok[3 + k] if tok[3 + k] >= 0 else unk["move"]
        ids[j, 7] = unk["move"]
        from fog.constants.schema import SET_EV_COLS, SET_NATURE_COL, SET_TERA_COL

        ids[j, NATURE_ID_FIELD] = (
            tok[SET_NATURE_COL] if tok[SET_NATURE_COL] >= 0 else unk["nature"]
        )
        for target, col in zip(EV_ID_FIELDS, SET_EV_COLS):
            ids[j, target] = tok[col] if tok[col] >= 0 else unk["ev"]
        if ids.shape[-1] >= N_ID_FIELDS:
            ids[j, ORIGINAL_ITEM_ID_FIELD] = unk["item"]
            ids[j, ORIGINAL_ABILITY_ID_FIELD] = unk["ability"]
        # epistemic flags cover item, ability, moves, nature, six EV stats, and tera.
        for bit, col in enumerate(
            (1, 2, 3, 4, 5, 6, SET_NATURE_COL, *SET_EV_COLS, SET_TERA_COL)
        ):
            ent[j, L.epistemic + bit] = 1.0 if tok[col] >= 0 else 0.0
        # my sheet-known tera type is hidden from the opponent until terastallized
        if tok[SET_TERA_COL] < 0:
            ent[j, L.tera_type : L.tera_type + DIMS.tera_type] = 0.0
        # PP fractions are private (opponent sees moves used, not my PP bookkeeping)
        ent[j, L.move_pp : L.move_pp + DIMS.move_pp] = 0.0
    return ent, ids


def mirror_field(field: np.ndarray) -> np.ndarray:
    """Swap the per-side blocks of a field vector so the viewer's side comes first.

    Symmetric blocks (weather/terrain/trickroom/turn/gravity) pass through unchanged.
    """
    out = field.copy()
    o = DIMS.field_weather + DIMS.field_terrain
    half = DIMS.field_hazards // 2                     # 6 per side
    out[o : o + half], out[o + half : o + 2 * half] = (
        field[o + half : o + 2 * half].copy(), field[o : o + half].copy())
    o += DIMS.field_hazards
    half = DIMS.field_screens // 2                     # 5 per side
    out[o : o + half], out[o + half : o + 2 * half] = (
        field[o + half : o + 2 * half].copy(), field[o : o + half].copy())
    o += DIMS.field_screens + DIMS.field_trickroom + DIMS.field_turn
    out[o], out[o + 1] = field[o + 1], field[o]        # forced-switch flags
    o += DIMS.field_forced_switch
    half = DIMS.field_delayed // 2                     # 5 per side (Future Sight / Wish)
    out[o : o + half], out[o + half : o + 2 * half] = (
        field[o + half : o + 2 * half].copy(), field[o : o + half].copy())
    return out


def mirror_history(history: np.ndarray) -> np.ndarray:
    """Swap viewer-relative legacy history fields; zero blocks that cannot be inverted exactly."""
    out = np.asarray(history, dtype=np.float32).copy()
    if not len(out):
        return out

    def swap(a, b) -> None:
        left = out[..., a].copy()
        out[..., a] = out[..., b]
        out[..., b] = left

    swap(0, 1)       # active species scalars
    swap(2, 3)       # hp
    left = out[..., 4:11].copy()
    out[..., 4:11] = out[..., 11:18]
    out[..., 11:18] = left
    swap(18, 19)     # legacy boost sums

    my_action = DIMS.hist_snapshot
    opp_action = my_action + DIMS.hist_my_action
    # The old record has my action *slot* but opponent action *identity*, so there is no lossless
    # side swap. The categorical sidecar below carries both identities; do not feed a false frame.
    out[..., my_action : opp_action + DIMS.hist_opp_action] = 0.0

    res = opp_action + DIMS.hist_opp_action
    swap(res + 0, res + 1)  # damage
    swap(res + 2, res + 3)  # crit
    left = out[..., res + 4 : res + 9].copy()
    out[..., res + 4 : res + 9] = out[..., res + 9 : res + 14]
    out[..., res + 9 : res + 14] = left
    swap(res + 14, res + 15)  # misses
    out[..., res + 16 : res + 18] = 0.0  # only opponent reveals existed in the legacy block
    swap(res + 18, res + 19)  # faints
    out[..., res + 20 : res + 22] = 0.0  # first actor/exact outgoing damage are asymmetric
    return out


def mirror_history_ids(history_ids: np.ndarray) -> np.ndarray:
    """Exact perspective swap for the categorical history sidecar."""
    out = np.asarray(history_ids, dtype=np.int64).copy()
    for left, right in ((0, 1), (2, 4), (3, 5), (6, 7), (8, 9), (10, 11),
                        (12, 13), (15, 16)):
        tmp = out[..., left].copy()
        out[..., left] = out[..., right]
        out[..., right] = tmp
    actor = out[..., H_FIRST_ACTOR].copy()
    out[..., H_FIRST_ACTOR] = np.where(actor == 1, 2, np.where(actor == 2, 1, 0))
    return out


def mirror_history_state(history_state: np.ndarray) -> np.ndarray:
    """Swap the two seven-stat active boost blocks."""
    out = np.asarray(history_state, dtype=np.float32).copy()
    left = out[..., :7].copy()
    out[..., :7] = out[..., 7:14]
    out[..., 7:14] = left
    return out


def build_opponent_view_np(obs, my_view_tokens: np.ndarray, my_truth_tokens: np.ndarray,
                           tables) -> tuple[dict, dict]:
    """Unbatched numpy (backbone_frag, my_view_dict) for the self-belief predictor.

    obs: my own Observation. my_view_tokens [6,15]: opponent's reveal state of me.
    my_truth_tokens [6,15]: my full team. Slots are swapped so I sit in 6-11.

    Public events still carry viewer-relative side and active-slot fields. The returned frag
    therefore contains a perspective-swapped history rather than aliasing the primary history.
    """
    from fog.constants.schema import N_EV_UNITS, N_NATURES
    from fog.encoding.reveals import (
        align_training_truth_to_views,
        labels_from_truth,
        masks_for_tokens,
    )

    unk = {
        "item": tables.n_items, "ability": tables.n_abilities, "move": tables.n_moves,
        "nature": N_NATURES, "ev": N_EV_UNITS,
    }

    my_ent, my_ids = _downgrade_my_entities(
        obs.entities[:6], obs.entity_ids[:6], my_view_tokens, unk
    )
    # swap: opponent (my slots 6-11, as I see them) -> new 0-5; me (downgraded) -> new 6-11
    opp_ent = obs.entities[6:12].copy()
    opp_ent[:, L.is_mine] = 1.0    # frame consistency: the viewer's own team reads is_mine=1
    my_ent[:, L.is_mine] = 0.0
    ent = np.concatenate([opp_ent, my_ent], axis=0)
    ids = np.concatenate([obs.entity_ids[6:12], my_ids], axis=0)
    # active_idx swaps sides
    mine, opp = (obs.active_idx if obs.active_idx is not None else np.array([0, 6]))
    active = np.array([int(opp) - 6, int(mine) + 6], dtype=np.int64)

    frag = {
        "entities": ent.astype(np.float32),
        "entity_ids": ids.astype(np.int64),
        "field": mirror_field(np.asarray(obs.field, dtype=np.float32)),
        "history": mirror_history(obs.history),
        "history_ids": mirror_history_ids(
            obs.history_ids if obs.history_ids is not None
            else np.zeros((len(obs.history), DIMS.hist_id_fields), dtype=np.int64)
        ),
        "history_state": mirror_history_state(
            obs.history_state if obs.history_state is not None
            else np.zeros((len(obs.history), DIMS.hist_state), dtype=np.float32)
        ),
        "history_len": np.int64(obs.history_len),
        "active_idx": active,
        # Action legality is private request state.  The opponent mirror cannot reconstruct it
        # from our observation, so use the same constant public placeholder used by public-only
        # probes instead of copying our (wrong-perspective and potentially leaking) mask.  The
        # typed shared-latent route consumes this field even when the mirror is used only for the
        # self-belief auxiliary, so omitting it makes otherwise valid PPO rollout batches fail at
        # update time.
        "legality_mask": np.ones(14, dtype=bool),
    }
    # my_view belief input: opponent's reveal tokens of me + masks + my truth labels. The view
    # packs MY revealed moves into the earliest subslots (RevealLedger order), so the truth grid
    # is aligned to that clamp before it becomes labels — an inconsistency here (revealed move
    # not on my own sheet) can only be a census bug; targets still follow the clamp.
    masks = masks_for_tokens(my_view_tokens, tables)
    aligned_truth, supervision_tokens, supervision_consistent = align_training_truth_to_views(
        my_truth_tokens, my_view_tokens
    )
    my_view = {
        "tokens": my_view_tokens.astype(np.int64),
        "supervision_tokens": supervision_tokens.astype(np.int64),
        "supervision_consistent": supervision_consistent.astype(bool),
        "masks": {k: v for k, v in masks.items() if k != "item"},
        "labels": labels_from_truth(aligned_truth),
    }
    return frag, my_view


def build_opponent_view(obs, my_view_tokens: np.ndarray, my_truth_tokens: np.ndarray, tables,
                        device=None) -> tuple[dict, dict]:
    """Batched-torch ([1, ...]) wrapper over `build_opponent_view_np`.

    Used for the per-decision frozen-probe measurement; training goes through the numpy path.
    """
    frag, my_view = build_opponent_view_np(obs, my_view_tokens, my_truth_tokens, tables)

    def _t(a, dt):
        t = torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
        return (t.to(device) if device is not None else t).unsqueeze(0)

    batch = {
        "entities": _t(frag["entities"], torch.float32),
        "entity_ids": _t(frag["entity_ids"], torch.long),
        "field": _t(frag["field"], torch.float32),
        "history": _t(frag["history"], torch.float32),
        "history_ids": _t(frag["history_ids"], torch.long),
        "history_state": _t(frag["history_state"], torch.float32),
        "history_len": _t(np.asarray(frag["history_len"], dtype=np.int64), torch.long),
        "active_idx": _t(frag["active_idx"], torch.long),
        "legality_mask": _t(frag["legality_mask"], torch.bool),
    }
    mv = {
        "tokens": _t(my_view["tokens"], torch.long),
        "supervision_tokens": _t(my_view["supervision_tokens"], torch.long),
        "supervision_consistent": _t(my_view["supervision_consistent"], torch.bool),
        "masks": {k: _t(v, torch.bool) for k, v in my_view["masks"].items()},
        "labels": {k: _t(v, torch.long) for k, v in my_view["labels"].items()},
    }
    return batch, mv
