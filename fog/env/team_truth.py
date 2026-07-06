"""Ground-truth team labels for self-play belief supervision (spec §3.2, §5, §6).

In self-play the simulator provides both full teams, so belief targets are the opponent's TRUE set
(not just what leaked). We obtain that truth by matching a finished battle's revealed opponent species
against the known team pool: within a small pool the 6-species multiset is effectively unique, so the
match recovers the full team including its still-hidden items/moves/spreads/tera.

`truth_tokens_from_paste` parses a Showdown paste into the C1 token layout [6, 9] (SLOT_ATTRS order),
with spread mapped to its nearest generated cluster. `PoolTruth` indexes a pool for battle-end lookup.
Unmatched battles return None -> the caller falls back to censored (end-of-battle-revealed) labels,
which are always available.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from fog.constants.schema import MAX_SPREADS_PER_SPECIES, TYPE_INDEX

_NON_ID = re.compile(r"[^a-z0-9]")


def _to_id(s: object) -> str:
    return _NON_ID.sub("", str(s).lower())


def _nearest_spread(species_id: str, evs: list[int], nature: str, tables) -> int:
    """Index of the closest generated spread cluster for this species (<OTHER_SPREAD> = last)."""
    clusters = (tables.spreads.get("species", {}) or {}).get(species_id, [])
    if not clusters:
        return MAX_SPREADS_PER_SPECIES  # <OTHER_SPREAD>
    from fog.constants.build_spreads import nature_modified_stats

    feats = tables.species["features"].get(species_id)
    if feats is None:
        return MAX_SPREADS_PER_SPECIES
    base = np.array(feats["base_stats"], dtype=np.int64)
    try:
        target = nature_modified_stats(base, np.asarray(evs), nature)
    except Exception:
        return MAX_SPREADS_PER_SPECIES
    best, best_d = MAX_SPREADS_PER_SPECIES, float("inf")
    for i, c in enumerate(clusters[:MAX_SPREADS_PER_SPECIES]):
        d = float(np.abs(np.array(c["stats"]) - target).sum()) if c.get("stats") else float("inf")
        if d < best_d:
            best, best_d = i, d
    return best


def truth_tokens_from_paste(paste: str, tables) -> np.ndarray:
    """Showdown paste -> [6, 9] int64 truth tokens (SLOT_ATTRS: species,item,ability,move0-3,spread,
    tera). Uses poke-env's paste parser for robustness."""
    from poke_env.teambuilder import Teambuilder

    class _T(Teambuilder):
        def yield_team(self):  # pragma: no cover
            return ""

    mons = _T().parse_showdown_team(paste)
    tokens = np.full((6, 9), -1, dtype=np.int64)
    for j, mon in enumerate(mons[:6]):
        sid = _to_id(mon.species or (mon.nickname or ""))
        tokens[j, 0] = tables.species_index(sid)
        if mon.item:
            tokens[j, 1] = tables.item_index(_to_id(mon.item))
        if mon.ability:
            tokens[j, 2] = tables.ability_index(_to_id(mon.ability), default=-1)
        for k, mv in enumerate([m for m in (mon.moves or []) if m][:4]):
            tokens[j, 3 + k] = tables.move_index(_to_id(mv), default=-1)
        evs = [int(x or 0) for x in (mon.evs or [0] * 6)]
        tokens[j, 7] = _nearest_spread(sid, evs, mon.nature or "hardy", tables)
        tera = getattr(mon, "tera_type", None) or getattr(mon, "teraType", None)
        tokens[j, 8] = TYPE_INDEX.get(_to_id(tera), 18) if tera else -1
    return tokens


class PoolTruth:
    """Match a finished battle's opponent to its pool team by revealed-species multiset."""

    def __init__(self, pool_pastes: list[str], tables) -> None:
        self.tables = tables
        self._truth = [truth_tokens_from_paste(p, tables) for p in pool_pastes]
        self._species: list[Counter] = [Counter(t[:, 0].tolist()) for t in self._truth]

    def match(self, revealed_species_idx: list[int]) -> np.ndarray | None:
        """Return [6,9] truth for the unique pool team containing all revealed species, else None."""
        want = Counter(s for s in revealed_species_idx if s >= 0)
        if not want:
            return None
        candidates = [
            i for i, sp in enumerate(self._species) if all(sp.get(k, 0) >= v for k, v in want.items())
        ]
        return self._truth[candidates[0]].copy() if len(candidates) == 1 else None


def revealed_opponent_species(battle, tables) -> list[int]:
    """Species indices revealed for the opponent so far (from poke-env opponent_team)."""
    from fog.encoding.reveals import opponent_slots

    return [tables.species_index(sid) for sid, _ in opponent_slots(battle) if sid]
