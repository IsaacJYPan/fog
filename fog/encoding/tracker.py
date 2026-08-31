"""Incremental per-battle event tracking (2026-07 encoding audit fields; spec §2.1/§2.2/§2.3).

poke-env exposes most battle state, but several decision-relevant signals only exist in the event
stream. `BattleTracker` consumes each completed turn's protocol events exactly once and maintains:

  * last_used_move   per mon (reset on switch-out) — Encore/choice-lock/Torment inference
  * booster_consumed per mon (|-enditem|...|Booster Energy) — item-inference signal
  * fallen_at_entry  per mon — own-side faint count at last switch-in (Supreme Overlord locks at entry)
  * volatile-counter start turns (taunt/encore/disable/partiallytrapped/perishsong) — turns remaining
  * Future Sight / Wish pending state per side + Gravity (field_delayed block)
  * per-turn start-of-turn snapshots + my chosen action indices + hp seeds (history retro-fidelity)

The live player owns one tracker per battle (keyed by battle_tag) and feeds it alongside the
RevealLedger; the replay parser can drive the same class from raw logs. All methods are duck-typed
numpy/python — no poke-env import.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fog.encoding.events import event_mon_key, event_role, hp_fraction, to_id

# volatile-counter effect -> nominal duration in turns (approximate where item/move-dependent)
_COUNTER_DURATION = {
    "taunt": 3,
    "encore": 3,
    "disable": 4,
    "partiallytrapped": 5,
    "perishsong": 3,
}
_PARTIAL_TRAP_MOVES = {
    "bind", "clamp", "firespin", "infestation", "magmastorm", "sandtomb",
    "snaptrap", "thundercage", "whirlpool", "wrap",
}


@dataclass
class _SideDelayed:
    future_sight_turn: int | None = None   # turn Future Sight was set on this side (hits end of t+2)
    wish_turn: int | None = None           # turn Wish was used by this side (heals end of t+1)
    wish_frac: float = 0.5                 # heal as hp-frac of the receiver; 0.5 = half-max default


class BattleTracker:
    """One battle's incremental event-derived state. Feed turns in order via `feed_turn`."""

    def __init__(self, my_role: str = "p1") -> None:
        self.my_role = my_role
        self.last_used_move: dict[str, str] = {}      # mon_key -> move id string
        self.booster_consumed: set[str] = set()
        self.original_booster: set[str] = set()       # private sheet snapshot at team preview
        self.original_item: dict[str, str] = {}
        self.original_ability: dict[str, str] = {}
        # Exact submitted own-sheet rows keyed by species-table id. Request-side calculated stats
        # can admit multiple nature/EV reconstructions (for example Sassy vs Calm with 0 IVs);
        # the player knows the paste it submitted, so retain that authoritative private context.
        self.exact_private_truth: dict[int, np.ndarray] = {}
        self.fallen_at_entry: dict[str, int] = {}
        self.faints: dict[str, int] = {"p1": 0, "p2": 0}
        self.effect_start: dict[tuple[str, str], int] = {}   # (mon_key, effect) -> start turn
        self.delayed: dict[str, _SideDelayed] = {"p1": _SideDelayed(), "p2": _SideDelayed()}
        self.hp: dict[str, float] = {}                # mon_key -> current hp fraction
        self._active: dict[str, str | None] = {"p1": None, "p2": None}
        # poke-env's current observation grows as public events arrive within a turn. Keep a cursor
        # so current-turn facts can be consumed immediately without double-counting older events.
        self._event_counts: dict[int, int] = {}
        # decision-time recordings (filled by the player, not by events)
        self.snapshots: dict[int, np.ndarray] = {}    # turn -> [hist_snapshot] start-of-turn snapshot
        self.snapshot_ids: dict[int, np.ndarray] = {}
        self.snapshot_state: dict[int, np.ndarray] = {}
        self.my_actions: dict[int, int] = {}          # turn -> chosen action index 0-13
        self.hp_at_turn_start: dict[int, dict[str, float]] = {}

    def clone(self) -> BattleTracker:
        """A copy safe to feed one more turn without touching the original (search-fork leaves).

        The dicts `feed_turn` mutates are copied; the large decision-time recordings
        (`snapshots`, `my_actions`) are SHARED by reference — `feed_turn` never writes them and a
        leaf encoding only reads them, so sharing is safe and keeps forks O(1) in game length.
        """
        import copy

        new = BattleTracker.__new__(BattleTracker)
        new.my_role = self.my_role
        new.last_used_move = dict(self.last_used_move)
        new.booster_consumed = set(self.booster_consumed)
        new.original_booster = set(self.original_booster)
        new.original_item = dict(self.original_item)
        new.original_ability = dict(self.original_ability)
        new.exact_private_truth = {
            species: row.copy() for species, row in self.exact_private_truth.items()
        }
        new.fallen_at_entry = dict(self.fallen_at_entry)
        new.faints = dict(self.faints)
        new.effect_start = dict(self.effect_start)
        new.delayed = {r: copy.copy(d) for r, d in self.delayed.items()}
        new.hp = dict(self.hp)
        new._active = dict(self._active)
        new._event_counts = dict(self._event_counts)
        new.snapshots = self.snapshots        # shared: feed_turn never writes it
        new.snapshot_ids = self.snapshot_ids
        new.snapshot_state = self.snapshot_state
        new.my_actions = self.my_actions      # shared: feed_turn never writes it
        new.hp_at_turn_start = dict(self.hp_at_turn_start)
        return new

    # ---------------------------------------------------------------------------- player recordings
    def record_exact_private_truth(self, truth_tokens: np.ndarray) -> None:
        """Remember the exact submitted own team sheet, keyed by unique species id."""
        rows = np.asarray(truth_tokens, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] < 14:
            raise ValueError("exact private truth must have shape [slots, >=14]")
        for row in rows:
            species = int(row[0])
            if species >= 0:
                self.exact_private_truth[species] = row.copy()

    def record_private_team(self, battle: object) -> None:
        """Remember irreversible private starting-sheet facts before requests mutate them."""
        for key, mon in (getattr(battle, "team", None) or {}).items():
            mon_key = str(key)
            item = to_id(getattr(mon, "item", None) or "")
            ability = to_id(getattr(mon, "ability", None) or "")
            if item and mon_key not in self.original_item:
                self.original_item[mon_key] = item
            if ability and mon_key not in self.original_ability:
                self.original_ability[mon_key] = ability
            if item == "boosterenergy":
                self.original_booster.add(mon_key)

    def booster_was_consumed(self, mon_key: str, mon: object) -> bool:
        """True after a known starting Booster Energy is no longer currently held."""
        if mon_key in self.booster_consumed:
            return True
        current = to_id(getattr(mon, "item", None) or "")
        return mon_key in self.original_booster and current in ("", "noitem")

    def record_decision(
        self,
        turn: int,
        snapshot: np.ndarray | None,
        action_idx: int | None,
        *,
        snapshot_ids: np.ndarray | None = None,
        snapshot_state: np.ndarray | None = None,
    ) -> None:
        """Record the start-of-turn snapshot + my chosen action for turn `turn` (history §2.3)."""
        if snapshot is not None and turn not in self.snapshots:
            self.snapshots[turn] = np.asarray(snapshot, dtype=np.float32)
        if snapshot_ids is not None and turn not in self.snapshot_ids:
            self.snapshot_ids[turn] = np.asarray(snapshot_ids, dtype=np.int64)
        if snapshot_state is not None and turn not in self.snapshot_state:
            self.snapshot_state[turn] = np.asarray(snapshot_state, dtype=np.float32)
        if action_idx is not None:
            self.my_actions[turn] = int(action_idx)
        if turn not in self.hp_at_turn_start:
            self.hp_at_turn_start[turn] = dict(self.hp)

    # ------------------------------------------------------------------------------- event feeding
    def feed_turn(self, turn: int, events: list[list[str]]) -> None:
        """Consume the newly appended suffix of one turn's cumulative public event list."""
        start = self._event_counts.get(turn, 0)
        if len(events) <= start:
            return
        self._event_counts[turn] = len(events)
        self.hp_at_turn_start.setdefault(turn, dict(self.hp))

        for ev in events[start:]:
            if len(ev) < 2:
                continue
            kind = ev[1]
            args = ev[2:]
            if not args:
                continue
            role = event_role(args[0])
            key = event_mon_key(args[0])

            if kind == "move" and len(args) >= 2:
                move_id = to_id(args[1])
                self.last_used_move[key] = move_id
                if role and move_id == "wish":
                    self.delayed[role].wish_turn = turn
                    self.delayed[role].wish_frac = 0.5  # half the user's max HP (receiver unknown)

            elif kind in ("switch", "drag") and role:
                prev = self._active.get(role)
                if prev is not None:
                    self.last_used_move.pop(prev, None)   # reset on switch-out
                self._active[role] = key
                self.fallen_at_entry[key] = self.faints.get(role, 0)
                if len(args) >= 3:
                    frac = hp_fraction(args[2])
                    if frac is not None:
                        self.hp[key] = frac
                # volatiles (and their counters) end on switch-out
                for k_eff in [ke for ke in self.effect_start if ke[0] == prev]:
                    self.effect_start.pop(k_eff, None)

            elif kind == "faint" and role:
                self.faints[role] = self.faints.get(role, 0) + 1

            elif kind in ("-damage", "-heal") and len(args) >= 2:
                frac = hp_fraction(args[1])
                if frac is not None:
                    self.hp[key] = frac

            elif kind == "-enditem" and len(args) >= 2:
                if to_id(args[1]) == "boosterenergy":
                    self.booster_consumed.add(key)

            elif kind == "-start" and len(args) >= 2:
                eff = to_id(args[1])
                eff = eff.removeprefix("move")
                if eff in ("futuresight", "doomdesire"):
                    # pending doom lands on the side OPPOSITE the user
                    target = "p2" if role == "p1" else "p1"
                    self.delayed[target].future_sight_turn = turn
                elif eff.startswith("perish"):
                    self.effect_start[(key, "perishsong")] = turn
                elif eff in _COUNTER_DURATION:
                    self.effect_start[(key, eff)] = turn

            elif kind == "-activate" and len(args) >= 2:
                eff = to_id(args[1])
                eff = eff.removeprefix("move")
                if eff in _PARTIAL_TRAP_MOVES:
                    self.effect_start[(key, "partiallytrapped")] = turn

            elif kind == "-end" and len(args) >= 2:
                eff = to_id(args[1])
                eff = eff.removeprefix("move")
                if eff in ("futuresight", "doomdesire") and role:
                    self.delayed[role].future_sight_turn = None
                elif eff in _COUNTER_DURATION:
                    self.effect_start.pop((key, eff), None)

        # Wish resolves at the end of the turn after it was used
        for side in self.delayed.values():
            if side.wish_turn is not None and turn > side.wish_turn:
                side.wish_turn = None

    # --------------------------------------------------------------------------------- encoder taps
    def counter_remaining(self, mon_key: str, effect: str, now: int) -> float:
        """Approximate turns remaining /8 for a tracked volatile counter (0 when untracked)."""
        start = self.effect_start.get((mon_key, effect))
        if start is None:
            return 0.0
        dur = _COUNTER_DURATION.get(effect, 3)
        return max(0.0, dur - (now - start)) / 8.0

    def delayed_block(self, now: int) -> np.ndarray:
        """The field_delayed block [10]: per side (mine first) — FS pending, FS turns-left/3,
        Wish pending, Wish turns-left/2, Wish heal frac."""
        out = np.zeros(10, dtype=np.float32)
        order = (self.my_role, "p2" if self.my_role == "p1" else "p1")
        for i, role in enumerate(order):
            side = self.delayed[role]
            o = 5 * i
            if side.future_sight_turn is not None:
                out[o] = 1.0
                out[o + 1] = max(0.0, (side.future_sight_turn + 2) - now) / 3.0
            if side.wish_turn is not None:
                out[o + 2] = 1.0
                out[o + 3] = max(0.0, (side.wish_turn + 1) - now) / 2.0
                out[o + 4] = side.wish_frac
        return out

    def fallen_for(self, mon_key: str, role: str) -> float:
        """Supreme Overlord fallen count /5 at this mon's last switch-in (leads count from 0)."""
        return min(5, self.fallen_at_entry.get(mon_key, 0)) / 5.0
