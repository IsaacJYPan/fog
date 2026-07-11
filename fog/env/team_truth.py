"""Ground-truth team labels for self-play belief supervision (spec §3.2, §5, §6).

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

`truth_tokens_from_paste` parses a Showdown paste into the C1 token layout [6, 9] (SLOT_ATTRS order),
with spread mapped to its nearest generated cluster. Unmatched battles return None -> the caller
falls back to censored labels (no supervision for that game).
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
    """Match a battle's opponent to its pool team (see module docstring for which matcher to use)."""

    def __init__(self, pool_pastes: list[str], tables) -> None:
        from fog.constants.schema import NO_ITEM

        self.tables = tables
        self._truth = [truth_tokens_from_paste(p, tables) for p in pool_pastes]
        self._species: list[Counter] = [Counter(t[:, 0].tolist()) for t in self._truth]
        self._no_item = tables.item_index(NO_ITEM)

    def match(self, revealed_species_idx: list[int]) -> np.ndarray | None:
        """Return [6,9] truth for the unique pool team containing all revealed species, else None.

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
        """[6,9] truth for the UNIQUE pool team consistent with the revealed per-slot views, else None.

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
        enumeration source for joint search, SEARCH_PLAN §4.1). Unlike `match_views` this does not
        require uniqueness: before anything distinguishing is revealed it returns the whole pool.
        No revealed views => everything is consistent."""
        revealed = [v for v in views if v.species is not None and int(v.species) >= 0]
        if not revealed:
            return list(range(len(self._truth)))
        return [i for i, t in enumerate(self._truth) if self._consistent(revealed, t)]

    def truth_tokens(self, index: int) -> np.ndarray:
        """[6,9] truth tokens for pool team `index` (a copy)."""
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
        if view.tera is not None and row[8] >= 0 and int(view.tera) != int(row[8]):
            return False
        sheet_moves = {int(m) for m in row[3:7] if m >= 0}
        if sheet_moves and any(int(m) not in sheet_moves for m in view.moves):
            return False
        return True

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


def revealed_opponent_species(battle, tables) -> list[int]:
    """Species indices revealed for the opponent so far (from poke-env opponent_team)."""
    from fog.encoding.reveals import opponent_slots

    return [tables.species_index(sid) for sid, _ in opponent_slots(battle) if sid]
