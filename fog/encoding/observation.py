"""The assembled decision-point observation (spec §2) and the entry point that builds it.

`Observation` is the contract between the parser/env (which produce it) and the model (which consumes
it). Tensor construction from a poke-env `Battle` (or a parsed replay row) is delegated to `entities`,
`field`, and `history`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fog.encoding import entities as _entities
from fog.encoding import field as _field
from fog.encoding import history as _history
from fog.encoding.dims import DIMS

if TYPE_CHECKING:  # avoid importing poke-env at module load
    from poke_env.environment import AbstractBattle


@dataclass
class Observation:
    """One decision point, from the acting player's perspective.

    Shapes:
      entities        [12, D_ent_raw]   -- raw feature blocks; embeddings applied inside the model
      entity_ids      [12, N_id_fields] -- integer ids for embedding lookups (species/item/ability/...)
      field           [D_field]
      history         [T, D_turn]
      history_ids     [T, D_hist_ids] categorical identities aligned with history
      history_state   [T, 14] individual boost stages of both start-of-turn actives
      history_len     scalar (true T before front-padding)
      legality_mask   [14] bool         -- from the request JSON, never learned
      belief_labels   optional dict     -- censored ground-truth for aux losses (§3.2, §5)
    """

    entities: np.ndarray
    entity_ids: np.ndarray
    field: np.ndarray
    history: np.ndarray
    history_len: int
    legality_mask: np.ndarray
    action_features: np.ndarray | None = None  # [14,F] facts aligned with each action slot
    history_ids: np.ndarray | None = None
    history_state: np.ndarray | None = None
    active_idx: np.ndarray | None = None       # [2] int64: my/opp active entity-token indices
    decision_phase: int = 0                    # request-visible DecisionPhase
    opp_revealed: dict[str, Any] | None = None
    opp_view: dict[str, Any] | None = None
    belief_labels: dict[str, Any] | None = None

    @classmethod
    def from_battle(
        cls,
        battle: AbstractBattle,
        *,
        perspective: str = "p1",
        tracker=None,
        own_revealed: dict[str, Any] | np.ndarray | None = None,
    ) -> Observation:
        """Build an observation from a live poke-env battle state.

        Falls back to the zero-fill shape-test path when the generated constant tables are missing
        (TablesNotBuilt) — real play REQUIRES built tables (and an agent from FogAgent.from_tables).
        `tracker` (encoding.tracker.BattleTracker, optional) fills the event-derived audit fields
        (last_used_move / booster / fallen / volatile counters / Future Sight-Wish / history
        snapshots+actions). ``own_revealed`` is used only by public-replay parsing, where the
        acting player's unavailable private request/team sheet must be represented as unknown
        rather than fabricated from terminal current state.
        """
        from fog.constants.tables import TABLES, TablesNotBuilt

        try:
            _ = TABLES.n_species  # trigger lazy load
        except TablesNotBuilt:
            tables = None
        else:
            tables = TABLES

        opp_revealed = _empty_opp_revealed()
        if tables is not None and battle is not None:
            from fog.encoding.reveals import build_belief_input, opponent_views_from_battle

            opp_revealed = build_belief_input(opponent_views_from_battle(battle, tables), tables)

        if tracker is not None and battle is not None:
            tracker.record_private_team(battle)
        ent, ent_ids, active_idx = _entities.encode_entities(
            battle, tables, opp_tokens=opp_revealed["tokens"], perspective=perspective,
            tracker=tracker,
        )
        if own_revealed is not None and tables is not None:
            from fog.constants.schema import N_EV_UNITS, N_NATURES
            from fog.encoding.mirror import _downgrade_my_entities

            own_tokens = (
                np.asarray(own_revealed["tokens"], dtype=np.int64)
                if isinstance(own_revealed, dict)
                else np.asarray(own_revealed, dtype=np.int64)
            )
            unknown = {
                "item": tables.n_items,
                "ability": tables.n_abilities,
                "move": tables.n_moves,
                "nature": N_NATURES,
                "ev": N_EV_UNITS,
            }
            ent[:6], ent_ids[:6] = _downgrade_my_entities(
                ent[:6], ent_ids[:6], own_tokens, unknown
            )
        fld = _field.encode_field(battle, perspective=perspective, tracker=tracker)
        hist, hist_len = _history.encode_history(
            battle, tables, perspective=perspective, tracker=tracker
        )
        hist_ids, hist_state = _history.encode_history_sidecars(
            battle, tables, perspective=perspective, tracker=tracker
        )
        mask = _entities.legality_mask(battle, perspective=perspective)
        from fog.encoding.action_features import encode_action_features

        action_features = encode_action_features(battle)
        from fog.encoding.player_observation import decision_phase_from_battle

        return cls(
            entities=ent,
            entity_ids=ent_ids,
            field=fld,
            history=hist,
            history_len=hist_len,
            legality_mask=mask,
            action_features=action_features,
            history_ids=hist_ids,
            history_state=hist_state,
            active_idx=active_idx,
            decision_phase=int(decision_phase_from_battle(battle)),
            opp_revealed=opp_revealed,
        )

    @classmethod
    def public_from_engine_battle(cls, battle: Any, role: str) -> Observation:
        """Build the public-only value view of an ``EngineBattle``.

        A normal player observation contains that player's complete private sheet in entity slots
        0-5.  A public-belief value query must not contain either private sheet: candidate sheets
        are supplied separately to the value network.  This constructor starts from ``role``'s
        ordinary view, then replaces its own private fields with exactly what the other player has
        publicly observed.  The opponent side was already censored by ``EngineView``.

        The legality mask is also private because it can reveal unrevealed moves, remaining PP,
        choice locks, and trapping.  The shared value trunk does not consume it today, but storing
        a constant mask prevents a future trunk change from silently opening that leak.
        """
        from fog.constants.schema import N_EV_UNITS, N_NATURES
        from fog.encoding.mirror import _downgrade_my_entities

        if not hasattr(battle, "view") or not hasattr(battle, "trackers"):
            raise TypeError("public_from_engine_battle requires an EngineBattle-like object")
        other = "p2" if role == "p1" else "p1"
        obs = cls.from_battle(battle.view(role), tracker=battle.trackers[role])
        reverse = cls.from_battle(battle.view(other), tracker=battle.trackers[other])
        tables = battle.tables
        unknown = {
            "item": tables.n_items,
            "ability": tables.n_abilities,
            "move": tables.n_moves,
            "nature": N_NATURES,
            "ev": N_EV_UNITS,
        }
        own_public_tokens = np.asarray(reverse.opp_revealed["tokens"], dtype=np.int64)
        own_entities, own_ids = _downgrade_my_entities(
            obs.entities[:6], obs.entity_ids[:6], own_public_tokens, unknown
        )
        obs.entities[:6] = own_entities
        obs.entity_ids[:6] = own_ids
        obs.legality_mask = np.ones(14, dtype=bool)
        return obs

    def validate(self) -> None:
        """Cheap shape assertions used in tests and the golden-battle integration test."""
        assert self.entities.shape[0] == 12
        assert self.field.shape[0] == DIMS.field_dim, (self.field.shape, DIMS.field_dim)
        assert self.history.ndim == 2 and self.history.shape[1] == DIMS.turn_dim
        if self.history_ids is not None:
            assert self.history_ids.shape == (self.history.shape[0], DIMS.hist_id_fields)
        if self.history_state is not None:
            assert self.history_state.shape == (self.history.shape[0], DIMS.hist_state)
        assert self.legality_mask.shape == (14,)
        assert self.legality_mask.any(), "legality mask must be nonempty"
        if self.action_features is not None:
            from fog.encoding.action_features import ACTION_FEATURE_DIM

            assert self.action_features.shape == (14, ACTION_FEATURE_DIM)
            assert np.isfinite(self.action_features).all()
        assert 0 <= int(self.decision_phase) < 6


def _empty_opp_revealed() -> dict[str, Any]:
    """All-hidden C1 dict — the fallback when tables/battle are unavailable (shape-test path)."""
    from fog.constants.schema import SET_N_ATTRS

    return {"tokens": np.full((6, SET_N_ATTRS), -1, dtype=np.int64), "masks": {}}
