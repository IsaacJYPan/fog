"""Ground-truth team labels for self-play belief supervision.

In self-play the simulator provides both full teams, so belief targets are the opponent's TRUE set
(not just what leaked). The PREFERRED truth source is the recording `PoolTeambuilder` (env/teams.py):
the trainer sampled the opponent's team, so with sequential battles it simply knows the paste — no
inference. The matchers here are the fallback for concurrent battles and the progressive source for
the eval-time set oracle:

  * `match(revealed_species)` — unique species-multiset containment. Sufficient on species-distinct
    pools (POOL-B) but PERMANENTLY AMBIGUOUS on shared-species pools (POOL-A, where all teams share
    the same 6 species by design) — it returns None there for every battle. Do not rely on it for
    training supervision; that mistake once disabled belief training on exactly the pool where sets
    are hidden.
  * `match_views(views)` — consistency match over ALL revealed attributes (species + item/ability/
    moves/tera per slot, injective slot assignment). Disambiguates shared-species pools once any
    set-distinguishing attribute is revealed.

`truth_tokens_from_paste` parses a Showdown paste into the C1 token layout [6, 15]
(SET_SLOT_ATTRS order), preserving nature and all six EV contributions exactly. Unmatched battles
return None -> the caller falls back to censored labels (no supervision for that game).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from fog.constants.schema import (
    NATURE_INDEX,
    SET_EV_COLS,
    SET_N_ATTRS,
    SET_NATURE_COL,
    SET_TERA_COL,
    TYPE_INDEX,
)

_NON_ID = re.compile(r"[^a-z0-9]")


def _to_id(s: object) -> str:
    return _NON_ID.sub("", str(s).lower())


def truth_tokens_from_paste(paste: str, tables) -> np.ndarray:
    """Showdown paste -> [6,15] exact submitted-set truth tokens.

    EV labels are native stat-contribution units ``floor(EV/4)`` in [0,63]. Showdown accepts EV
    remainders, but they do not affect stats; decoding therefore canonicalizes them to multiples of
    four. Uses poke-env's paste parser for robustness.
    """
    from poke_env.teambuilder import Teambuilder

    class _T(Teambuilder):
        def yield_team(self):  # pragma: no cover
            return ""

    mons = _T().parse_showdown_team(paste)
    tokens = np.full((6, SET_N_ATTRS), -1, dtype=np.int64)
    for j, mon in enumerate(mons[:6]):
        sid = _to_id(mon.species or (mon.nickname or ""))
        tokens[j, 0] = tables.species_index(sid)
        if mon.item:
            tokens[j, 1] = tables.item_index(_to_id(mon.item))
        if mon.ability:
            tokens[j, 2] = tables.ability_index(_to_id(mon.ability), default=-1)
        for k, mv in enumerate([m for m in (mon.moves or []) if m][:4]):
            tokens[j, 3 + k] = tables.move_index(_to_id(mv), default=-1)
        nature = _to_id(mon.nature or "serious")
        tokens[j, SET_NATURE_COL] = NATURE_INDEX.get(nature, NATURE_INDEX["serious"])
        evs = [int(x or 0) for x in (mon.evs or [0] * 6)]
        tokens[j, list(SET_EV_COLS)] = [min(63, max(0, ev // 4)) for ev in evs]
        tera = getattr(mon, "tera_type", None) or getattr(mon, "teraType", None)
        tokens[j, SET_TERA_COL] = TYPE_INDEX.get(_to_id(tera), 18) if tera else -1
    return tokens


class PoolTruth:
    """Match a battle's opponent to its pool team (see module docstring for which matcher to use)."""

    def __init__(self, pool_pastes: list[str], tables) -> None:
        from fog.constants.schema import NO_ITEM

        self.tables = tables
        self._truth = [truth_tokens_from_paste(p, tables) for p in pool_pastes]
        self._species: list[Counter] = [Counter(t[:, 0].tolist()) for t in self._truth]
        self._no_item = tables.item_index(NO_ITEM)

    def match(self, revealed_species_idx: list[int]) -> np.ndarray | None:
        """Return [6,15] truth for the unique pool team containing all revealed species, else None.

        WARNING: always None on shared-species pools (POOL-A) — prefer `match_views`/`match_battle`.
        """
        want = Counter(s for s in revealed_species_idx if s >= 0)
        if not want:
            return None
        candidates = [
            i for i, sp in enumerate(self._species) if all(sp.get(k, 0) >= v for k, v in want.items())
        ]
        return self._truth[candidates[0]].copy() if len(candidates) == 1 else None

    def match_views(self, views) -> np.ndarray | None:
        """[6,15] truth for the UNIQUE pool team consistent with the revealed per-slot views, else None.

        A team is consistent when the revealed slots admit an injective assignment onto its slots with
        species equal and every revealed attribute matching: item (skipping `<NO_ITEM>` — knock-off /
        consumption leaves the view at <NO_ITEM> while the sheet keeps the original item), ability,
        tera type, and revealed moves ⊆ the slot's moveset.
        """
        revealed = [v for v in views if v.species is not None and int(v.species) >= 0]
        if not revealed:
            return None
        candidates = [i for i, t in enumerate(self._truth) if self._consistent(revealed, t)]
        return self._truth[candidates[0]].copy() if len(candidates) == 1 else None

    def consistent_indices(self, views) -> list[int]:
        """Pool indices of EVERY team consistent with the revealed per-slot views (the exact
        enumeration source for legacy joint search, ARCHITECTURE.md S6). Unlike `match_views` this does not
        require uniqueness: before anything distinguishing is revealed it returns the whole pool.
        No revealed views => everything is consistent."""
        revealed = [v for v in views if v.species is not None and int(v.species) >= 0]
        if not revealed:
            return list(range(len(self._truth)))
        return [i for i, t in enumerate(self._truth) if self._consistent(revealed, t)]

    def truth_tokens(self, index: int) -> np.ndarray:
        """[6,15] truth tokens for pool team `index` (a copy)."""
        return self._truth[index].copy()

    def match_battle(self, battle) -> np.ndarray | None:
        """`match_views` over the battle's current opponent information state (progressive: call it
        any turn; returns truth once the reveals disambiguate). The eval set-oracle path."""
        from fog.encoding.reveals import opponent_views_from_battle

        return self.match_views(opponent_views_from_battle(battle, self.tables))

    def hidden_info_warning(self) -> str | None:
        """None when the pool conceals something; else a message explaining why a leakage run on this
        pool is structurally vacuous: if every team's species multiset is unique, team preview alone
        identifies the exact team, the probe memorizes the species->set mapping, still-hidden CE
        collapses to ~0 and the β penalty multiplies nothing."""
        multisets = [tuple(sorted(c.items())) for c in self._species]
        if len(set(multisets)) == len(multisets):
            return (
                "every team in this pool has a UNIQUE species multiset: preview identifies the team "
                "at turn 0 and there is (almost) no hidden information to conceal. Use a shared-"
                "species set-variant pool (POOL-A style) for belief/leakage experiments."
            )
        return None

    def _slot_consistent(self, view, row: np.ndarray) -> bool:
        if int(row[0]) != int(view.species):
            return False
        if (
            view.item is not None and int(view.item) != self._no_item
            and row[1] >= 0 and int(view.item) != int(row[1])
        ):
            return False
        if view.ability is not None and row[2] >= 0 and int(view.ability) != int(row[2]):
            return False
        if (
            view.tera is not None
            and row[SET_TERA_COL] >= 0
            and int(view.tera) != int(row[SET_TERA_COL])
        ):
            return False
        sheet_moves = {int(m) for m in row[3:7] if m >= 0}
        return not (sheet_moves and any(int(m) not in sheet_moves for m in view.moves))

    def _consistent(self, views, truth: np.ndarray) -> bool:
        """Injective assignment of revealed views onto team slots (backtracking; <=6x6)."""
        used = [False] * truth.shape[0]

        def rec(i: int) -> bool:
            if i == len(views):
                return True
            for j in range(truth.shape[0]):
                if not used[j] and self._slot_consistent(views[i], truth[j]):
                    used[j] = True
                    if rec(i + 1):
                        return True
                    used[j] = False
            return False

        return rec(0)


def assert_uniform_slot_order(cand_tokens: np.ndarray) -> None:
    """Guard for slot-by-slot candidate scoring (pool_id): within any group of candidates that
    share a species MULTISET, every candidate must list those species in the SAME slot order.
    Otherwise a set-variant of the true team sits permuted against the observation's
    preview/paste slot order and scores as a species contradiction — silently, since the species
    conditionals just bury it. TRUE for all committed pools (POOL-A verified 2026-07-11) but
    unenforced until now; a regenerated pool_gen or replay-built pool could violate it.

    cand_tokens [C, 6, 15]. Raises ValueError naming the first offending pair. Species-distinct
    candidates are exempt (no slot-by-slot ambiguity to guard)."""
    cand = np.asarray(cand_tokens, dtype=np.int64)
    groups: dict[tuple, tuple[int, tuple]] = {}
    for i in range(cand.shape[0]):
        seq = tuple(int(s) for s in cand[i, :, 0])
        key = tuple(sorted(seq))
        first = groups.setdefault(key, (i, seq))
        if first[1] != seq:
            raise ValueError(
                f"pool candidates {first[0]} and {i} share a species multiset but list it in "
                f"different slot orders ({first[1]} vs {seq}); slot-by-slot scoring against the "
                f"observation's preview order is invalid for such a pool. Reorder the pastes."
            )


def revealed_opponent_species(battle, tables) -> list[int]:
    """Species indices revealed for the opponent so far (from poke-env opponent_team)."""
    from fog.encoding.reveals import opponent_slots

    return [tables.species_index(sid) for sid, _ in opponent_slots(battle) if sid]
