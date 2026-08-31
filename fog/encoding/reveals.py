"""Information-state tracking: who provably knows what (spec §2.1 epistemic flags, §3.2/§3.3 inputs).

This module is the single source of truth for building the C1 belief-net input dicts
(ARCHITECTURE.md §4):

  * `opp_revealed` — MY view of the OPPONENT's team: species from preview, everything else only if
    publicly revealed. Read off poke-env's opponent battle state (which is itself an information
    state — poke-env only ever knows what the protocol revealed).
  * `my_view` — the OPPONENT's view of MY team, for the self-belief predictor (§3.3). This CANNOT be
    read off my own battle state (I see everything); it is derived by running the same revelation
    rules over my actions via a `RevealLedger` that the player/parser feeds from events.

Everything here is pure numpy + duck typing (no poke-env import), so the revelation logic is unit-
testable with mock mons; the one battle-facing function (`opponent_views_from_battle`) uses getattr
probing in the style of env/actions.py.

Token conventions (schema.SET_SLOT_ATTRS order — species, item, ability, move0..3, nature,
EV-unit hp/atk/def/spa/spd/spe, tera): value = categorical index, -1 = hidden/unknown. Revealed
moves are packed into the EARLIEST move
slots in reveal order (a §3.2 requirement — the belief net clamps them there), and every
truth/candidate grid consumed against such a view must be passed through `align_truth_to_views`
first so targets follow the same convention (the 2026-07-12 move-order fix). `<NO_ITEM>` after
Knock Off is a *revealed* item state (its own index), NOT -1: the opponent knowing you hold nothing
is information.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from fog.constants.schema import (
    NO_ITEM,
    RARE_SPECIES,
    SET_EV_COLS,
    SET_MOVE_COLS,
    SET_N_ATTRS,
    SET_NATURE_COL,
    SET_TERA_COL,
)

N_SLOTS = 6
N_ATTRS = SET_N_ATTRS
_MOVE_COLS = SET_MOVE_COLS
_NON_ID = re.compile(r"[^a-z0-9]")

# family -> token columns, mirrored from fog.models.self_belief._FAMILY_COLS (numpy side)
FAMILY_COLS: dict[str, tuple[int, ...]] = {
    "species": (0,),
    "item": (1,),
    "ability": (2,),
    "move": _MOVE_COLS,
    "nature": (SET_NATURE_COL,),
    "ev": SET_EV_COLS,
    "tera": (SET_TERA_COL,),
}


def to_id(name: object) -> str:
    return _NON_ID.sub("", str(name).lower())


@dataclass
class SlotView:
    """One slot's information state from some viewer's perspective. None = unknown to the viewer."""

    species: int | None = None
    item: int | None = None          # table index; includes <NO_ITEM> — that IS knowledge
    ability: int | None = None
    moves: list[int] = field(default_factory=list)  # revealed, in reveal order, <=4
    nature: int | None = None
    evs: list[int] = field(default_factory=list)
    tera: int | None = None          # tera TYPE index, known once terastallized


def tokens_from_views(views: list[SlotView]) -> np.ndarray:
    """[6,15] int64 token tensor (C1 contract); -1 where unknown; moves packed earliest-first."""
    tokens = np.full((N_SLOTS, N_ATTRS), -1, dtype=np.int64)
    for j, v in enumerate(views[:N_SLOTS]):
        for col, val in (
            (0, v.species), (1, v.item), (2, v.ability),
            (SET_NATURE_COL, v.nature), (SET_TERA_COL, v.tera),
        ):
            if val is not None:
                tokens[j, col] = val
        for col, value in zip(SET_EV_COLS, v.evs):
            tokens[j, col] = value
        for k, mv in enumerate(v.moves[:4]):
            tokens[j, _MOVE_COLS[k]] = mv
    return tokens


def masks_for_tokens(tokens: np.ndarray, tables) -> dict[str, np.ndarray]:
    """Per-family legality masks [6, n_x] bool for the belief net, keyed off each slot's species.

    Known species -> learnset / legal-abilities rows from `tables`, species mask clamped one-hot.
    Nature and EV values use global fixed vocabularies and need no static mask; the belief sampler
    applies the dynamic total-EV budget.
    Unknown species (token -1) -> all-True everywhere (uninformative prior).
    """
    masks = {
        "species": np.ones((N_SLOTS, tables.n_species), dtype=bool),
        "item": np.ones((N_SLOTS, tables.n_items), dtype=bool),
        "ability": np.ones((N_SLOTS, tables.n_abilities), dtype=bool),
        "move": np.ones((N_SLOTS, tables.n_moves), dtype=bool),
    }
    rare_idx = tables.species_index(RARE_SPECIES)

    for j in range(N_SLOTS):
        sp = int(tokens[j, 0])
        if sp < 0:
            continue
        masks["species"][j] = False
        masks["species"][j, sp] = True
        if sp == rare_idx:
            continue  # rare tail: keep uninformative attribute masks
        # HARD legality (a species genuinely cannot have these): learnset + legal abilities.
        masks["move"][j] = tables.learnset_mask(sp)
        masks["ability"][j] = tables.legal_abilities_mask(sp)
        # Illusion can make the public protocol attribute the active Zoroark's move to its
        # disguise slot. Once an observed move contradicts that slot's apparent species, the
        # apparent species is no longer a sound source of hard move/ability legality. Relax both
        # masks using only public evidence (not the private truth label), so training and
        # inference remain consistent and the relaxed mask cannot leak the target class.
        revealed_moves = [
            int(move) for move in tokens[j, list(_MOVE_COLS)] if int(move) >= 0
        ]
        if any(
            move >= masks["move"].shape[1] or not masks["move"][j, move]
            for move in revealed_moves
        ):
            masks["move"][j] = True
            masks["ability"][j] = True
        # Items are NOT species-locked — any Pokemon can hold any item. The per-species usage list is
        # a prior, not a legality constraint, so it must not become a hard -inf (that makes the CE on
        # a legitimately-rare item's ground-truth label diverge). The belief net learns item priors
        # from data instead; item mask stays all-True.
    return masks


def align_truth_to_views(
    truth_tokens: np.ndarray, view_tokens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align truth/candidate MOVE subslots to the input-packing convention (the one ordering rule
    everywhere, decided 2026-07-11): per slot, revealed moves first — at the subslots where
    `tokens_from_views` clamps them, in the view's subslot order — then the remaining sheet moves
    in their original paste order. Non-move columns pass through unchanged.

    Without this, chain inputs (reveal-order packing) and truth grids (paste order) disagree on
    the same 4 subslots: training targets sit under contradictory clamps or on the move-
    distinctness -inf (silently dropped by the loss guard), and pool_id eliminates the TRUE
    candidate whenever a revealed move's paste position != its clamp position.

    truth_tokens [..., 6, 15]; view_tokens [..., 6, 15] or a single [6, 15] observation shared by
    every leading row (the pool_id candidates case). Returns (aligned, consistent):
      aligned    -- new array, same shape as truth_tokens;
      consistent -- bool [..., 6]; False where a slot's FULLY-labeled sheet lacks a revealed
                    move — a genuine contradiction (pool_id maps the candidate to -inf; on a
                    training truth grid it means a reveal-census bug). Slots whose sheet has any
                    -1 move entry (censored labels, <4-move sets, off-table indices) are never
                    marked inconsistent: membership can't be proven either way.

    Revealed moves are placed even when inconsistent, so training targets always match the input
    clamp; trailing sheet moves that no longer fit in 4 subslots are dropped (only possible in
    the inconsistent case).
    """
    truth = np.asarray(truth_tokens, dtype=np.int64)
    lead = truth.shape[:-2]
    view = np.asarray(view_tokens, dtype=np.int64)
    view = np.broadcast_to(view, lead + (N_SLOTS, N_ATTRS))

    aligned = truth.copy().reshape(-1, N_SLOTS, N_ATTRS)
    flat_view = view.reshape(-1, N_SLOTS, N_ATTRS)
    consistent = np.ones((aligned.shape[0], N_SLOTS), dtype=bool)

    mv = list(_MOVE_COLS)
    for i in range(aligned.shape[0]):
        for j in range(N_SLOTS):
            revealed = [int(m) for m in flat_view[i, j, mv] if m >= 0]
            if not revealed:
                continue  # nothing clamped -> paste order is already the convention
            sheet = [int(m) for m in aligned[i, j, mv]]
            remaining = [m for m in sheet if m >= 0]
            has_gap = len(remaining) < 4
            for m in revealed:
                if m in remaining:
                    remaining.remove(m)
                elif not has_gap:
                    consistent[i, j] = False
            new = (revealed + remaining)[:4]
            aligned[i, j, mv] = new + [-1] * (4 - len(new))

    return aligned.reshape(truth.shape), consistent.reshape(lead + (N_SLOTS,))


def align_training_truth_to_views(
    truth_tokens: np.ndarray, view_tokens: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align normal reveals without corrupting truth under Illusion-like contradictions.

    A move publicly attributed to an Illusion disguise can be absent from that preview slot's
    submitted set. Keep that contradiction in the model input as useful evidence, but retain the
    original truth row and treat its real move fields as hidden for supervision.
    """
    truth = np.asarray(truth_tokens, dtype=np.int64)
    aligned, consistent = align_truth_to_views(truth, view_tokens)
    supervision_view = np.broadcast_to(
        np.asarray(view_tokens, dtype=np.int64), truth.shape
    ).copy()
    flat_truth = truth.reshape(-1, N_SLOTS, N_ATTRS)
    flat_aligned = aligned.reshape(-1, N_SLOTS, N_ATTRS)
    flat_supervision = supervision_view.reshape(-1, N_SLOTS, N_ATTRS)
    flat_consistent = consistent.reshape(-1, N_SLOTS)
    for row, slot in np.argwhere(~flat_consistent):
        flat_aligned[row, slot] = flat_truth[row, slot]
        flat_supervision[row, slot, list(_MOVE_COLS)] = -1
    return aligned, supervision_view, consistent


def labels_from_truth(truth_tokens: np.ndarray) -> dict[str, np.ndarray]:
    """[6,15] full-truth tokens -> per-family C2 label arrays (numpy; torchify at batch time)."""
    out: dict[str, np.ndarray] = {}
    for fam, cols in FAMILY_COLS.items():
        sel = truth_tokens[..., list(cols)]
        out[fam] = sel if fam in ("move", "ev") else sel.squeeze(-1)
    return out


def original_field_hidden(tokens: np.ndarray, tables) -> np.ndarray:
    """Whether each submitted-set field is still a prediction target.

    Most public tokens use ``-1`` for hidden.  Item is the important exception: ``<NO_ITEM>`` says
    that the Pokemon currently holds nothing, not what item appeared on its submitted set.  The
    current-state fact stays in the board/history encodings while the original item remains hidden
    and trainable here.
    """
    values = np.asarray(tokens, dtype=np.int64)
    hidden = values < 0
    hidden = hidden.copy()
    hidden[..., 1] |= values[..., 1] == tables.item_index("<NO_ITEM>")
    return hidden


def build_belief_input(views: list[SlotView], tables, truth_tokens: np.ndarray | None = None) -> dict:
    """Assemble a C1 dict (single observation, unbatched). With `truth_tokens` (my own team, for the
    self-belief predictor) the dict also carries `labels` — full ground truth, C2 shapes."""
    tokens = tokens_from_views(views)
    out = {"tokens": tokens, "masks": masks_for_tokens(tokens, tables)}
    if truth_tokens is not None:
        out["labels"] = labels_from_truth(truth_tokens)
    return out


# ------------------------------------------------------------------------------------ my-side ledger
class RevealLedger:
    """Tracks what MY play has publicly revealed — the opponent's knowledge of me.

    The player (live) or parser (replays) calls the record_* hooks off protocol events:
      |move|p1a...        -> record_move            (also first reveal of the mon itself)
      |-item| / Knock Off -> record_item            (pass the <NO_ITEM> index after knock-off!)
      |-enditem|          -> record_item(NO_ITEM)   (consumed berry/gem/booster — distinct info)
      |-ability|          -> record_ability
      |-terastallize|     -> record_tera
    Keys are mon identifiers (any stable string — use the poke-env identifier or species id).
    Reset per battle (the M1 history-leak guard applies here too).
    """

    def __init__(self) -> None:
        self._moves: dict[str, list[int]] = {}
        self._items: dict[str, int] = {}
        self._abilities: dict[str, int] = {}
        self._teras: dict[str, int] = {}

    def reset(self) -> None:
        self._moves.clear()
        self._items.clear()
        self._abilities.clear()
        self._teras.clear()

    def record_move(self, mon_key: str, move_idx: int) -> None:
        seen = self._moves.setdefault(mon_key, [])
        if move_idx not in seen and len(seen) < 4:
            seen.append(move_idx)

    def record_item(self, mon_key: str, item_idx: int) -> None:
        self._items[mon_key] = item_idx

    def record_ability(self, mon_key: str, ability_idx: int) -> None:
        self._abilities[mon_key] = ability_idx

    def record_tera(self, mon_key: str, tera_idx: int) -> None:
        self._teras[mon_key] = tera_idx

    def views(self, team_keys: list[str], species_tokens: list[int]) -> list[SlotView]:
        """Opponent's view of my team. Species are public from preview (standard OU) — pass my
        preview-order species indices; everything else fills only if this ledger recorded it."""
        out = []
        for key, sp in zip(team_keys, species_tokens):
            out.append(
                SlotView(
                    species=sp,
                    item=self._items.get(key),
                    ability=self._abilities.get(key),
                    moves=list(self._moves.get(key, [])),
                    nature=None,  # nature/EVs are never directly revealed (damage inference only)
                    tera=self._teras.get(key),
                )
            )
        return out


# --------------------------------------------------------------------------- poke-env battle facing
def opponent_slots(battle) -> list[tuple[str | None, object | None]]:
    """Stable opponent slot order: (species_id, mon_or_None) x6.

    Preview order when available (|poke| message order — the spec §2.1 convention), with revealed
    mons matched onto their preview slot by species (unique under Species Clause; randombattle has
    no dupes either). Mons revealed but absent from preview (Illusion weirdness) append after.
    Without preview info, falls back to opponent_team insertion (reveal) order.
    """
    team = getattr(battle, "opponent_team", None) or {}
    by_species = {to_id(getattr(m, "species", "")): m for m in team.values()}
    preview = list(getattr(battle, "teampreview_opponent_team", None) or [])
    slots: list[tuple[str | None, object | None]] = []
    seen: set[str] = set()
    for p in preview[:N_SLOTS]:
        sid = to_id(getattr(p, "species", p))
        slots.append((sid, by_species.get(sid)))
        seen.add(sid)
    for sid, mon in by_species.items():
        if sid not in seen and len(slots) < N_SLOTS:
            slots.append((sid, mon))
    while len(slots) < N_SLOTS:
        slots.append((None, None))
    return slots[:N_SLOTS]


def opponent_views_from_battle(battle, tables) -> list[SlotView]:
    """MY view of the opponent, off poke-env's opponent state (already an information state).

    Duck-typed like env/actions.py. Item semantics (verify in the golden-battle test — this is the
    documented poke-env 0.10 convention): attribute missing or == "unknown_item" -> unknown;
    None/"" -> <NO_ITEM> (knocked off / consumed, publicly known); otherwise the item id.
    Slot order: `opponent_slots` (preview order; stable across the battle).
    """
    views: list[SlotView] = []
    for sid, mon in opponent_slots(battle):
        v = SlotView()
        if sid:
            v.species = tables.species_index(sid)
        if mon is None:
            views.append(v)
            continue

        item = getattr(mon, "item", "unknown_item")
        if item is None or item == "":
            v.item = tables.item_index(NO_ITEM)
        elif to_id(item) not in ("unknownitem",):
            v.item = tables.item_index(to_id(item))
        # else: leave unknown (-1); UNKNOWN_ITEM sentinel index is for the *entity record*, not here

        ability = getattr(mon, "ability", None)
        if ability:
            # off-table ability -> None: treat as unrevealed rather than crash (rare-tail path)
            v.ability = tables.ability_index(to_id(ability), default=None)

        for move_id in list(getattr(mon, "moves", None) or {}):
            idx = tables.move_index(to_id(move_id), default=None)
            if idx is not None and idx not in v.moves and len(v.moves) < 4:
                v.moves.append(idx)

        if getattr(mon, "terastallized", False):
            tera = getattr(mon, "tera_type", None)
            if tera is not None:
                from fog.constants.schema import TYPE_INDEX

                # PokemonType enum or plain string; unknown/Stellar -> index 18
                v.tera = TYPE_INDEX.get(to_id(getattr(tera, "name", tera)), 18)
        views.append(v)

    return views  # opponent_slots already pads to 6


__all__ = [
    "FAMILY_COLS",
    "RevealLedger",
    "SlotView",
    "align_training_truth_to_views",
    "align_truth_to_views",
    "build_belief_input",
    "labels_from_truth",
    "masks_for_tokens",
    "opponent_views_from_battle",
    "original_field_hidden",
    "tokens_from_views",
]
