"""Replay JSON -> per-turn decision rows (spec §4.3). HIGHEST-DEFECT-RISK COMPONENT.

Each row (from the ACTING player's perspective): the §2 observation tensors, the action label, censored
belief labels (opponent attributes revealed by the END of the replay), the acting player's own
eventually-revealed attributes (self-belief labels, §5), and the game outcome.

Built on poke-env's battle-state tracker (spec-mandated): two `Battle` objects (one per perspective)
are driven through the public protocol log. Public replays lack the private `|request|` JSON, so
legality masks are reconstructed from state (revealed moves + non-fainted bench) with the chosen action
forced legal — sufficient for behavior cloning (the label is always in-support).

Storage: one compressed `.npz` shard per replay under data/parsed/shards/ + a parquet manifest
(data/parsed/manifest.parquet). BC reads the manifest and streams shards. Failures are quarantined
(expect 2-5%; investigate if higher, §4.3).

    python -m fog.scripts.parse_replays --raw data/raw_replays --out data/parsed
"""
from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

import numpy as np

from fog.constants.tables import TABLES
from fog.encoding.dims import DIMS
from fog.encoding.history import pad_or_truncate_front
from fog.encoding.observation import Observation
from fog.env.actions import team_order, team_order_minus_active

_LOG = logging.getLogger("fog.parse")
_LOG.disabled = True

HIST_STORE_CAP = 24   # front-padded history window stored per row (GRU consumes any T at train time)


def _to_id(s: object) -> str:
    import re

    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _mon_species_key(token: str) -> str:
    # 'p1a: Great Tusk' -> role, species-ish nickname
    import re

    m = re.match(r"(p[12])[ab]?:\s*(.*)", token)
    return (m.group(1), m.group(2)) if m else ("", token)


class _PerspectiveState:
    """Accumulates one player's decisions while its Battle is driven through the log."""

    def __init__(self, role: str):
        from poke_env.battle import Battle

        from fog.encoding.tracker import BattleTracker

        self.role = role
        self.battle = Battle(f"battle-gen9ou-{role}", f"{role}user", _LOG, gen=9)
        self.battle._player_role = role
        self.tracker = BattleTracker(my_role=role)   # event-derived encoder-fidelity fields
        self.move_order: dict[str, list[str]] = {}   # mon-key -> discovered move ids (slot order)
        self.rows: list[dict] = []
        self._pending_obs = None
        self._pending_mask = None
        self._pending_action: int | None = None

    def feed_tracker(self, turn: int, events: list[list[str]]) -> None:
        try:
            self.tracker.feed_turn(turn, events)
        except Exception:
            pass  # fidelity fields degrade to zeros; never quarantine a replay over them

    def snapshot(self) -> None:
        """Capture start-of-turn observation + a state-reconstructed legality mask; flush prior."""
        self._flush()
        try:
            self._pending_obs = Observation.from_battle(self.battle, tracker=self.tracker)
            self._pending_mask = _state_legality_mask(self.battle)
            from fog.encoding.events import build_snapshot

            self.tracker.record_decision(
                int(getattr(self.battle, "turn", 0) or 0), build_snapshot(self.battle, TABLES), None
            )
        except Exception:
            self._pending_obs = None
            self._pending_mask = None
        self._pending_action = None

    def _flush(self) -> None:
        if self._pending_obs is not None and self._pending_action is not None:
            self.rows.append(
                {"obs": self._pending_obs, "mask": self._pending_mask, "action": self._pending_action}
            )
        self._pending_obs = None
        self._pending_mask = None
        self._pending_action = None

    def record_action(self, parts: list[str]) -> None:
        """parts = a |move| or |switch| line initiated by THIS player. Only the first per turn."""
        if self._pending_obs is None or self._pending_action is not None:
            return  # no snapshot yet, or already recorded this player's decision this turn
        kind = parts[1]
        actor = parts[2]
        role, _ = _mon_species_key(actor)
        if role != self.role:
            return
        battle = self.battle
        if kind == "move":
            mon_key = actor
            move_id = _to_id(parts[3]) if len(parts) > 3 else ""
            order = self.move_order.setdefault(mon_key, [])
            if move_id and move_id not in order and len(order) < 4:
                order.append(move_id)
            slot = order.index(move_id) if move_id in order else 0
            tera = any("terastallize" in p.lower() or "[tera]" in p.lower() for p in parts[4:])
            self._pending_action = (4 + slot) if tera else slot
        elif kind == "switch":
            target_species = _to_id(parts[3].split(",")[0]) if len(parts) > 3 else ""
            bench = team_order_minus_active(battle)
            slot = 0
            for i, mon in enumerate(bench[:5]):
                if _to_id(getattr(mon, "species", "")) == target_species:
                    slot = i
                    break
            self._pending_action = 8 + slot
        if self._pending_action is not None:  # history retro-fidelity: my action per turn (§2.3)
            self.tracker.my_actions[int(getattr(battle, "turn", 0) or 0)] = self._pending_action

    def finish(self) -> None:
        self._flush()


def _state_legality_mask(battle) -> np.ndarray:
    """Reconstruct a plausible legal-action set from public state (replays lack the request JSON):
    the active mon's revealed moves as move slots + every non-fainted bench mon as a switch. The
    acting player's true chosen action is force-added later; this set gives BC real alternatives to
    discriminate among rather than a trivial 1-of-2 choice."""
    mask = np.zeros(14, dtype=bool)
    active = getattr(battle, "active_pokemon", None)
    if active is not None:
        for i in range(len(list((getattr(active, "moves", None) or {}))[:4])):
            mask[i] = True
    for i, mon in enumerate(team_order_minus_active(battle)[:5]):
        if not getattr(mon, "fainted", False):
            mask[8 + i] = True
    return mask


def _censored_truth(battle) -> np.ndarray:
    """[6,9] tokens of the OPPONENT's revealed attributes (censored labels, -1 = never revealed)."""
    from fog.encoding.reveals import build_belief_input, opponent_views_from_battle

    views = opponent_views_from_battle(battle, TABLES)
    return build_belief_input(views, TABLES)["tokens"]


def _own_censored_truth(battle) -> np.ndarray:
    """[6,9] tokens of the acting player's OWN revealed-by-end attributes (self-belief labels, §5)."""
    from fog.encoding.reveals import SlotView, tokens_from_views

    views = []
    for mon in team_order(battle)[:6]:
        v = SlotView()
        sp = getattr(mon, "species", None)
        if sp:
            v.species = TABLES.species_index(_to_id(sp))
        item = getattr(mon, "item", None)
        if item and _to_id(item) != "unknownitem":
            v.item = TABLES.item_index(_to_id(item))
        ability = getattr(mon, "ability", None)
        if ability:
            v.ability = TABLES.ability_index(_to_id(ability), default=None)
        for mid in list((getattr(mon, "moves", None) or {}))[:4]:
            idx = TABLES.move_index(_to_id(mid), default=None)
            if idx is not None and idx not in v.moves:
                v.moves.append(idx)
        if getattr(mon, "terastallized", False):
            tt = getattr(mon, "tera_type", None)
            if tt is not None:
                from fog.constants.schema import TYPE_INDEX

                v.tera = TYPE_INDEX.get(_to_id(getattr(tt, "name", tt)), 18)
        views.append(v)
    while len(views) < 6:
        views.append(SlotView())
    return tokens_from_views(views)


def parse_replay(replay_json: dict) -> dict:
    """Drive both perspectives; return {rows: [...], p1_win: bool|None}. Rows carry obs + labels."""
    log = replay_json["log"]
    players = replay_json.get("players", [])
    p1, p2 = _PerspectiveState("p1"), _PerspectiveState("p2")
    p1_win: bool | None = None

    turn_events: list[list[str]] = []   # protocol lines since the last |turn| (shared, tracker feed)
    for line in log.split("\n"):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""

        if tag == "turn":
            # advance battle state FIRST so battle.turn / observations include the completed turn
            # (snapshotting before parsing |turn| silently dropped the just-finished turn from the
            # history window — a real off-by-one, fixed 2026-07)
            for st in (p1, p2):
                try:
                    st.battle.parse_message(parts)
                except Exception:
                    pass
            completed = int(getattr(p1.battle, "turn", 1) or 1) - 1
            for st in (p1, p2):
                st.feed_tracker(completed, turn_events)
            turn_events = []
            p1.snapshot()
            p2.snapshot()
            continue

        turn_events.append(parts)
        if tag in ("move", "switch"):
            p1.record_action(parts)
            p2.record_action(parts)
        elif tag == "win" and len(parts) > 2:
            winner = parts[2].strip()
            if players:
                p1_win = winner == players[0]

        for st in (p1, p2):
            try:
                st.battle.parse_message(parts)
            except Exception:
                pass  # tolerate unknown/edge messages; state stays best-effort

    final_turn = int(getattr(p1.battle, "turn", 0) or 0)
    for st in (p1, p2):
        st.feed_tracker(final_turn, turn_events)
    p1.finish()
    p2.finish()

    # censored labels + outcome, per perspective (broadcast to that perspective's rows)
    out_rows = []
    for st, won in ((p1, p1_win), (p2, None if p1_win is None else not p1_win)):
        belief = _censored_truth(st.battle)
        self_belief = _own_censored_truth(st.battle)
        outcome = 0.0 if won is None else (1.0 if won else -1.0)
        for r in st.rows:
            out_rows.append(
                {"obs": r["obs"], "mask": r["mask"], "action": r["action"], "belief": belief,
                 "self_belief": self_belief, "outcome": outcome}
            )
    return {"rows": out_rows, "p1_win": p1_win}


def rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Stack decision rows into fixed-shape arrays for one npz shard."""
    n = len(rows)
    ent = np.zeros((n, 12, rows[0]["obs"].entities.shape[1]), dtype=np.float32)
    eid = np.zeros((n, 12, rows[0]["obs"].entity_ids.shape[1]), dtype=np.int64)
    fld = np.zeros((n, DIMS.field_dim), dtype=np.float32)
    hist = np.zeros((n, HIST_STORE_CAP, DIMS.turn_dim), dtype=np.float32)
    act_idx = np.zeros((n, 2), dtype=np.int64)
    mask = np.zeros((n, 14), dtype=bool)
    opp_tokens = np.full((n, 6, 9), -1, dtype=np.int64)  # belief-net input; masks rebuilt at load time
    action = np.zeros(n, dtype=np.int64)
    belief = np.zeros((n, 6, 9), dtype=np.int64)
    self_belief = np.zeros((n, 6, 9), dtype=np.int64)
    outcome = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(rows):
        o = r["obs"]
        ent[i] = o.entities
        eid[i] = o.entity_ids
        fld[i] = o.field
        hist[i] = pad_or_truncate_front(o.history, cap=HIST_STORE_CAP)
        act_idx[i] = o.active_idx if o.active_idx is not None else (0, 6)
        m = np.asarray(r["mask"], dtype=bool).copy() if r.get("mask") is not None else np.zeros(14, bool)
        m[r["action"]] = True  # the human's chosen action is always in-support
        if not m.any():
            m[13] = True
        mask[i] = m
        if o.opp_revealed is not None:
            opp_tokens[i] = o.opp_revealed["tokens"]
        action[i] = r["action"]
        belief[i] = r["belief"]
        self_belief[i] = r["self_belief"]
        outcome[i] = r["outcome"]
    return dict(
        entities=ent, entity_ids=eid, field=fld, history=hist, active_idx=act_idx,
        legality_mask=mask, opp_tokens=opp_tokens, action=action, belief_targets=belief,
        self_belief_targets=self_belief, outcome=outcome,
    )


def main() -> None:
    import polars as pl

    ap = argparse.ArgumentParser(description="Parse raw replays -> parsed shards + manifest (§4.3).")
    ap.add_argument("--raw", type=Path, default=Path("data/raw_replays"))
    ap.add_argument("--out", type=Path, default=Path("data/parsed"))
    ap.add_argument("--limit", type=int, default=0, help="max replays (0 = all)")
    args = ap.parse_args()

    shard_dir = args.out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = args.out / "_quarantine.json"

    raw_files = sorted(args.raw.glob("*.json"))
    if args.limit:
        raw_files = raw_files[: args.limit]

    manifest, quarantine = [], []
    parsed = 0
    for f in raw_files:
        try:
            replay = json.loads(f.read_text(encoding="utf-8"))
            result = parse_replay(replay)
            rows = result["rows"]
            if not rows:
                quarantine.append({"id": f.stem, "reason": "no decision rows"})
                continue
            arrays = rows_to_arrays(rows)
            shard = shard_dir / f"{f.stem}.npz"
            np.savez_compressed(shard, **arrays)
            manifest.append(
                {"id": f.stem, "n_rows": len(rows), "shard": str(shard),
                 "rating": int(replay.get("rating") or 0), "p1_win": result["p1_win"]}
            )
            parsed += 1
        except Exception as e:  # noqa: BLE001 - quarantine is the whole point
            quarantine.append({"id": f.stem, "reason": f"{type(e).__name__}: {e}",
                               "trace": traceback.format_exc()[-500:]})

    if manifest:
        pl.DataFrame(manifest).write_parquet(args.out / "manifest.parquet")
    quarantine_path.write_text(json.dumps(quarantine, indent=2), encoding="utf-8")
    total = parsed + len(quarantine)
    rate = len(quarantine) / total if total else 0.0
    print(f"parsed {parsed} replays ({sum(m['n_rows'] for m in manifest)} rows); "
          f"quarantined {len(quarantine)} ({rate:.1%})")
    if rate > 0.05:
        print("WARNING: quarantine rate > 5% — investigate (§4.3).")


if __name__ == "__main__":
    main()
