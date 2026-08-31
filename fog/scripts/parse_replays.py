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

HIST_STORE_CAP = 24   # front-padded window; causal history encoder consumes any T up to its cap


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

    def __init__(self, role: str, username: str, private_set=None, private_provenance=None):
        from poke_env.battle import Battle

        from fog.encoding.tracker import BattleTracker

        self.role = role
        # AbstractBattle compares public |player| usernames with this value. A fake username makes
        # both perspective objects settle on p1 after the p2 player line.
        self.battle = Battle(f"battle-gen9ou-{role}", username, _LOG, gen=9)
        self.battle._player_role = role
        self.tracker = BattleTracker(my_role=role)   # event-derived encoder-fidelity fields
        self.private_set = None if private_set is None else np.asarray(private_set, dtype=np.int64)
        self.private_provenance = (
            None if private_provenance is None else np.asarray(private_provenance, dtype=np.uint8)
        )
        if self.private_set is not None:
            self.tracker.record_exact_private_truth(self.private_set)
        self._preview_slot = 0
        self.move_order: dict[str, list[str]] = {}   # mon-key -> discovered move ids (slot order)
        self.rows: list[dict] = []
        self._pending_obs = None
        self._pending_mask = None
        self._pending_action: int | None = None
        self._pending_policy_valid = False
        self._pending_filter_reason = "unresolved"
        self._pending_legal_provenance = np.zeros(14, dtype=np.uint8)
        self._pending_action_feature_provenance = np.zeros(14, dtype=np.uint8)
        self._pending_turn: int | None = None
        self._tera_this_turn: set[str] = set()

    def register_own_preview(self, player: str, details: str) -> None:
        """Populate the own-side public preview that replay request JSON would normally supply."""
        if player != self.role:
            return
        species = details.split(",")[0].strip()
        mon = self.battle.get_pokemon(
            f"{player}: {species}",
            force_self_team=True,
            details=details,
        )
        if mon not in self.battle._teampreview_team:
            self.battle._teampreview_team.append(mon)
        if self.private_set is not None and self._preview_slot < len(self.private_set):
            from fog.training.replay_policy_data import apply_private_set

            apply_private_set(mon, self.private_set[self._preview_slot])
        self._preview_slot += 1

    def feed_tracker(self, turn: int, events: list[list[str]]) -> None:
        try:
            self.tracker.feed_turn(turn, events)
        except Exception:  # noqa: BLE001, S110 - tracker fidelity is intentionally best-effort
            pass  # fidelity fields degrade to zeros; never quarantine a replay over them

    def _prepare_inferred_request(self) -> None:
        """Build a request-like action list from one sampled own team."""
        active = getattr(self.battle, "active_pokemon", None)
        forced = active is None or bool(getattr(active, "fainted", False))
        self.battle._force_switch = forced
        self.battle._trapped = False
        self.battle._available_moves = (
            [] if forced else list((getattr(active, "moves", None) or {}).values())
        )
        self.battle._available_switches = [
            mon
            for mon in team_order_minus_active(self.battle)[:5]
            if not bool(getattr(mon, "fainted", False))
        ]
        used_tera = any(
            bool(getattr(mon, "terastallized", False)) for mon in team_order(self.battle)
        )
        self.battle._can_tera = bool(
            not forced and not used_tera and getattr(active, "tera_type", None) is not None
        )

    def _action_provenance(self) -> np.ndarray:
        result = np.zeros(14, dtype=np.uint8)
        if self.private_provenance is None:
            return result
        from fog.constants.schema import SET_MOVE_COLS, SET_TERA_COL
        from fog.training.replay_policy_data import EXACT

        active = getattr(self.battle, "active_pokemon", None)
        active_slot = next(
            (i for i, mon in enumerate(team_order(self.battle)[:6]) if mon is active), None
        )
        if active_slot is not None:
            for offset, column in enumerate(SET_MOVE_COLS):
                source = self.private_provenance[active_slot, column]
                result[offset] = source
                result[4 + offset] = min(
                    source, self.private_provenance[active_slot, SET_TERA_COL]
                )
        for offset, _mon in enumerate(team_order_minus_active(self.battle)[:5]):
            result[8 + offset] = EXACT
        return result

    def snapshot(
        self,
        own_revealed: dict | None = None,
        opponent_submitted: dict | None = None,
    ) -> None:
        """Capture start-of-turn observation + a state-reconstructed legality mask; flush prior."""
        self._flush()
        try:
            if self.private_set is not None:
                self._prepare_inferred_request()
            self._pending_obs = Observation.from_battle(
                self.battle,
                tracker=self.tracker,
                own_revealed=(self.private_set if self.private_set is not None else own_revealed),
            )
            if opponent_submitted is not None:
                self._pending_obs.opp_revealed = opponent_submitted
            self._pending_mask = (
                np.asarray(self._pending_obs.legality_mask, dtype=bool)
                if self.private_set is not None
                else _state_legality_mask(self.battle)
            )
            self._pending_turn = int(getattr(self.battle, "turn", 0) or 0)
            self._pending_legal_provenance[:] = 0
            self._pending_legal_provenance[self._pending_mask] = 1
            self._pending_action_feature_provenance = self._action_provenance()
            active = getattr(self.battle, "active_pokemon", None)
            if self.private_set is None:
                self._pending_filter_reason = "awaiting_action"
            elif active is None or bool(getattr(active, "fainted", False)):
                self._pending_filter_reason = "forced_switch"
            elif int(self._pending_mask.sum()) < 2:
                self._pending_filter_reason = "forced_single_choice"
            else:
                self._pending_filter_reason = "awaiting_action"
            from fog.encoding.events import record_history_decision

            record_history_decision(
                self.tracker,
                int(getattr(self.battle, "turn", 0) or 0),
                self.battle,
                TABLES,
                None,
            )
        except Exception:  # noqa: BLE001 - quarantine is handled at the complete-row boundary
            self._pending_obs = None
            self._pending_mask = None
        self._pending_action = None

    def _flush(self) -> None:
        if self._pending_obs is not None and self._pending_action is not None:
            legal_provenance = self._pending_legal_provenance.copy()
            if 0 <= self._pending_action < len(legal_provenance):
                legal_provenance[self._pending_action] = 2
            self.rows.append(
                {
                    "obs": self._pending_obs,
                    "mask": self._pending_mask,
                    "action": self._pending_action,
                    "policy_valid": self._pending_policy_valid,
                    "filter_reason": self._pending_filter_reason,
                    "own_set_provenance": self.private_provenance,
                    "legal_provenance": legal_provenance,
                    "action_feature_provenance": self._pending_action_feature_provenance.copy(),
                    "role": self.role,
                    "turn": self._pending_turn,
                }
            )
        self._pending_obs = None
        self._pending_mask = None
        self._pending_action = None
        self._tera_this_turn.clear()
        self._pending_policy_valid = False
        self._pending_filter_reason = "unresolved"
        self._pending_legal_provenance[:] = 0
        self._pending_action_feature_provenance[:] = 0
        self._pending_turn = None

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
            move_id = _to_id(parts[3]) if len(parts) > 3 else ""
            called = any(str(part).lower().startswith("[from]") for part in parts[4:])
            if self.private_set is not None:
                order = list(
                    (getattr(getattr(battle, "active_pokemon", None), "moves", None) or {}).keys()
                )[:4]
                found = move_id in order
            else:
                order = self.move_order.setdefault(actor, [])
                found = bool(move_id and move_id in order)
                if move_id and move_id not in order and len(order) < 4:
                    order.append(move_id)
            slot = order.index(move_id) if move_id in order else 0
            tera = actor in self._tera_this_turn or any(
                "terastallize" in part.lower() or "[tera]" in part.lower()
                for part in parts[4:]
            )
            self._pending_action = (4 + slot) if tera else slot
            self._pending_policy_valid = bool(
                found
                and not called
                and self._pending_filter_reason == "awaiting_action"
                and self._pending_mask is not None
                and self._pending_mask[self._pending_action]
            )
            self._pending_filter_reason = (
                "accepted" if self._pending_policy_valid else "called_or_unaligned_move"
            )
        elif kind == "switch":
            target_species = _to_id(parts[3].split(",")[0]) if len(parts) > 3 else ""
            bench = team_order_minus_active(battle)
            slot = 0
            found = False
            for i, mon in enumerate(bench[:5]):
                if _to_id(getattr(mon, "species", "")) == target_species:
                    slot = i
                    found = True
                    break
            self._pending_action = 8 + slot
            self._pending_policy_valid = bool(
                found
                and self._pending_filter_reason == "awaiting_action"
                and self._pending_mask is not None
                and self._pending_mask[self._pending_action]
            )
            self._pending_filter_reason = (
                "accepted" if self._pending_policy_valid else "forced_or_unaligned_switch"
            )
        if self._pending_action is not None:  # history retro-fidelity: my action per turn (§2.3)
            self.tracker.my_actions[int(getattr(battle, "turn", 0) or 0)] = self._pending_action

    def record_tera(self, parts: list[str]) -> None:
        if len(parts) >= 3:
            actor = parts[2]
            role, _ = _mon_species_key(actor)
            if role == self.role:
                self._tera_this_turn.add(actor)

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
        for i in range(len(list(getattr(active, "moves", None) or {})[:4])):
            mask[i] = True
    for i, mon in enumerate(team_order_minus_active(battle)[:5]):
        if not getattr(mon, "fainted", False):
            mask[8 + i] = True
    return mask


def _censored_truth(battle) -> np.ndarray:
    """Opponent submitted-set facts recoverable from the complete public replay.

    This is intentionally not the terminal current-state token grid: a consumed item is public
    ``<NO_ITEM>`` at game end, but the belief target is the original item named by the protocol.
    """
    from fog.training.replay_supervision import public_replay_truth

    return public_replay_truth(battle, TABLES)


def _submitted_reveal_input(battle) -> dict:
    """Original-set facts recoverable at this prefix, separate from current battle state."""
    from fog.encoding.reveals import masks_for_tokens
    from fog.training.replay_supervision import public_replay_truth

    tokens = public_replay_truth(battle, TABLES)
    return {"tokens": tokens, "masks": masks_for_tokens(tokens, TABLES)}


def _own_censored_truth(battle) -> np.ndarray:
    """[6,15] tokens of the acting player's own publicly revealed attributes."""
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
        for mid in list(getattr(mon, "moves", None) or {})[:4]:
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


def replay_player_names(replay_json: dict) -> dict[str, str]:
    """Return exact protocol usernames for both roles, with metadata as a fallback."""
    names: dict[str, str] = {}
    for line in str(replay_json.get("log", "")).splitlines():
        parts = line.split("|")
        if (
            len(parts) >= 4
            and parts[1] == "player"
            and parts[2] in ("p1", "p2")
            and parts[3]
        ):
            names[parts[2]] = parts[3]
    metadata = list(replay_json.get("players", []))
    for index, role in enumerate(("p1", "p2")):
        if role not in names and index < len(metadata):
            names[role] = str(metadata[index])
    if set(names) != {"p1", "p2"}:
        raise ValueError(f"replay is missing player identity messages: {sorted(names)}")
    return names


def parse_replay(
    replay_json: dict,
    *,
    private_sets: dict[str, np.ndarray] | None = None,
    private_provenance: dict[str, np.ndarray] | None = None,
) -> dict:
    """Drive both perspectives; return {rows: [...], p1_win: bool|None}. Rows carry obs + labels."""
    log = replay_json["log"]
    players = replay_json.get("players", [])
    names = replay_player_names(replay_json)
    private_sets = private_sets or {}
    private_provenance = private_provenance or {}
    p1 = _PerspectiveState(
        "p1", names["p1"], private_sets.get("p1"), private_provenance.get("p1")
    )
    p2 = _PerspectiveState(
        "p2", names["p2"], private_sets.get("p2"), private_provenance.get("p2")
    )
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
                except Exception:  # noqa: BLE001, S110 - tolerate unsupported protocol messages
                    pass
            completed = int(getattr(p1.battle, "turn", 1) or 1) - 1
            for st in (p1, p2):
                st.feed_tracker(completed, turn_events)
            turn_events = []
            from fog.encoding.reveals import build_belief_input, opponent_views_from_battle

            # A public replay has no private |request| payload or submitted sheet.  Each acting
            # side may use only what the opposite public perspective has seen of it.
            p1_public = build_belief_input(
                opponent_views_from_battle(p2.battle, TABLES), TABLES
            )
            p2_public = build_belief_input(
                opponent_views_from_battle(p1.battle, TABLES), TABLES
            )
            p1.snapshot(p1_public, _submitted_reveal_input(p1.battle))
            p2.snapshot(p2_public, _submitted_reveal_input(p2.battle))
            continue

        turn_events.append(parts)
        if tag == "poke" and len(parts) >= 4:
            for st in (p1, p2):
                st.register_own_preview(parts[2], parts[3])
        if tag in ("move", "switch"):
            p1.record_action(parts)
            p2.record_action(parts)
        elif tag == "-terastallize":
            p1.record_tera(parts)
            p2.record_tera(parts)
        elif tag == "win" and len(parts) > 2:
            winner = parts[2].strip()
            if players:
                p1_win = winner == players[0]

        for st in (p1, p2):
            try:
                st.battle.parse_message(parts)
            except Exception:  # noqa: BLE001, S110 - retain a best-effort public replay
                pass  # tolerate unknown/edge messages; state stays best-effort

    final_turn = int(getattr(p1.battle, "turn", 0) or 0)
    for st in (p1, p2):
        st.feed_tracker(final_turn, turn_events)
    p1.finish()
    p2.finish()
    for st in (p1, p2):
        if st.battle.player_role != st.role:
            raise ValueError(
                f"{st.role} replay battle settled on player_role={st.battle.player_role!r}"
            )

    # censored labels + outcome, per perspective (broadcast to that perspective's rows)
    belief_by_role = {
        "p1": _censored_truth(p1.battle),
        "p2": _censored_truth(p2.battle),
    }
    out_rows = []
    for st, won in ((p1, p1_win), (p2, None if p1_win is None else not p1_win)):
        belief = belief_by_role[st.role]
        other = "p2" if st.role == "p1" else "p1"
        # The opposite perspective's opponent reconstruction is precisely what this player's
        # public self-belief target should contain, including original-vs-current semantics.
        self_belief = belief_by_role[other]
        outcome = 0.0 if won is None else (1.0 if won else -1.0)
        for r in st.rows:
            out_rows.append(
                {
                    "obs": r["obs"],
                    "mask": r["mask"],
                    "action": r["action"],
                    "belief": belief,
                    "self_belief": self_belief,
                    "outcome": outcome,
                    "policy_valid": r["policy_valid"],
                    "filter_reason": r.get("filter_reason", "unresolved"),
                    "own_set_provenance": r.get("own_set_provenance"),
                    "legal_provenance": r.get("legal_provenance"),
                    "action_feature_provenance": r.get("action_feature_provenance"),
                    "role": r["role"],
                    "turn": r["turn"],
                }
            )
    return {"rows": out_rows, "p1_win": p1_win}


def parse_replay_policy_versions(
    replay_json: dict,
    *,
    replay_id: str,
    imputations: int = 3,
    seed: int = 0,
) -> dict:
    """Return filtered expert decisions under several plausible own private sheets.

    The first chronological pass learns only facts that each player eventually exposed about its
    own submitted set.  Those facts were private knowledge from turn one, so using them in that
    same player's reconstructed observation is legitimate.  Opponent observations are rebuilt
    chronologically on every pass and never receive end-of-game opponent facts.
    """
    if imputations < 1:
        raise ValueError("imputations must be positive")
    baseline = parse_replay(replay_json)
    revealed: dict[str, np.ndarray] = {}
    for role in ("p1", "p2"):
        row = next((item for item in baseline["rows"] if item["role"] == role), None)
        if row is not None:
            revealed[role] = np.asarray(row["self_belief"], dtype=np.int64)
    if set(revealed) != {"p1", "p2"}:
        return {"rows": [], "p1_win": baseline["p1_win"], "filter_counts": {"missing_side": 1}}

    from collections import Counter

    from fog.training.replay_policy_data import complete_private_team, stable_replay_seed

    accepted: list[dict] = []
    filters: Counter[str] = Counter()
    for imputation in range(imputations):
        private_sets: dict[str, np.ndarray] = {}
        provenance: dict[str, np.ndarray] = {}
        for role in ("p1", "p2"):
            private_sets[role], provenance[role] = complete_private_team(
                revealed[role],
                seed=stable_replay_seed(replay_id, role, imputation, seed),
            )
        version = parse_replay(
            replay_json,
            private_sets=private_sets,
            private_provenance=provenance,
        )
        for row in version["rows"]:
            reason = str(row.get("filter_reason", "unresolved"))
            if not row.get("policy_valid", False):
                filters[reason] += 1
                continue
            # A legal list containing only the demonstrated action conveys no preference.
            if int(np.asarray(row["mask"], dtype=bool).sum()) < 2:
                filters["forced_single_choice"] += 1
                continue
            row["imputation_id"] = imputation
            row["replay_id"] = replay_id
            accepted.append(row)
            filters["accepted"] += 1
    return {"rows": accepted, "p1_win": baseline["p1_win"], "filter_counts": dict(filters)}


def rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Stack decision rows into fixed-shape arrays for one npz shard."""
    n = len(rows)
    ent = np.zeros((n, 12, rows[0]["obs"].entities.shape[1]), dtype=np.float32)
    eid = np.zeros((n, 12, rows[0]["obs"].entity_ids.shape[1]), dtype=np.int64)
    fld = np.zeros((n, DIMS.field_dim), dtype=np.float32)
    hist = np.zeros((n, HIST_STORE_CAP, DIMS.turn_dim), dtype=np.float32)
    hist_ids = np.zeros((n, HIST_STORE_CAP, DIMS.hist_id_fields), dtype=np.int64)
    hist_state = np.zeros((n, HIST_STORE_CAP, DIMS.hist_state), dtype=np.float32)
    hist_len = np.zeros(n, dtype=np.int64)
    act_idx = np.zeros((n, 2), dtype=np.int64)
    mask = np.zeros((n, 14), dtype=bool)
    from fog.constants.schema import SET_N_ATTRS

    opp_tokens = np.full((n, 6, SET_N_ATTRS), -1, dtype=np.int64)
    action = np.zeros(n, dtype=np.int64)
    belief = np.zeros((n, 6, SET_N_ATTRS), dtype=np.int64)
    self_belief = np.zeros((n, 6, SET_N_ATTRS), dtype=np.int64)
    outcome = np.zeros(n, dtype=np.float32)
    from fog.encoding.action_features import ACTION_FEATURE_DIM

    action_features = np.zeros((n, 14, ACTION_FEATURE_DIM), dtype=np.float32)
    policy_valid = np.zeros(n, dtype=bool)
    own_set_provenance = np.zeros((n, 6, SET_N_ATTRS), dtype=np.uint8)
    opponent_set_provenance = np.zeros((n, 6, SET_N_ATTRS), dtype=np.uint8)
    legal_provenance = np.zeros((n, 14), dtype=np.uint8)
    action_feature_provenance = np.zeros((n, 14), dtype=np.uint8)
    imputation_id = np.full(n, -1, dtype=np.int16)

    for i, r in enumerate(rows):
        o = r["obs"]
        ent[i] = o.entities
        eid[i] = o.entity_ids
        fld[i] = o.field
        hist[i] = pad_or_truncate_front(o.history, cap=HIST_STORE_CAP)
        hist_ids[i] = pad_or_truncate_front(o.history_ids, cap=HIST_STORE_CAP)
        hist_state[i] = pad_or_truncate_front(o.history_state, cap=HIST_STORE_CAP)
        hist_len[i] = min(int(o.history_len), HIST_STORE_CAP)
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
        if o.action_features is not None:
            from fog.encoding.public_action_features import public_action_features

            action_features[i] = public_action_features(o.action_features)
        policy_valid[i] = bool(r.get("policy_valid", False))
        if r.get("own_set_provenance") is not None:
            own_set_provenance[i] = r["own_set_provenance"]
        opponent_set_provenance[i] = np.where(opp_tokens[i] >= 0, 2, 0).astype(np.uint8)
        if r.get("legal_provenance") is not None:
            legal_provenance[i] = r["legal_provenance"]
        if r.get("action_feature_provenance") is not None:
            action_feature_provenance[i] = r["action_feature_provenance"]
        imputation_id[i] = int(r.get("imputation_id", -1))
    return {
        "entities": ent,
        "entity_ids": eid,
        "field": fld,
        "history": hist,
        "history_ids": hist_ids,
        "history_state": hist_state,
        "history_len": hist_len,
        "active_idx": act_idx,
        "legality_mask": mask,
        "opp_tokens": opp_tokens,
        "action": action,
        "belief_targets": belief,
        "self_belief_targets": self_belief,
        "outcome": outcome,
        "action_features": action_features,
        "policy_valid": policy_valid,
        "own_set_provenance": own_set_provenance,
        "opponent_set_provenance": opponent_set_provenance,
        "legal_provenance": legal_provenance,
        "action_feature_provenance": action_feature_provenance,
        "imputation_id": imputation_id,
    }


def replay_files(raw_dir: Path, limit: int = 0) -> list[Path]:
    """Data replay JSON only; scraper cursors and other underscore metadata are not battles."""
    files = sorted(path for path in raw_dir.glob("*.json") if not path.name.startswith("_"))
    return files[:limit] if limit else files


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

    raw_files = replay_files(args.raw, args.limit)

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
