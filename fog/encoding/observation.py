"""The assembled decision-point observation (spec §2) and the entry point that builds it.

`Observation` is the contract between the parser/env (which produce it) and the model (which consumes
it). Tensor construction from a poke-env `Battle` (or a parsed replay row) is delegated to `entities`,
`field`, and `history` — all partly stubbed pending the parser.
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
    active_idx: np.ndarray | None = None       # [2] int64: my/opp active entity-token indices
    opp_revealed: dict[str, Any] | None = None
    opp_view: dict[str, Any] | None = None
    belief_labels: dict[str, Any] | None = None

    @classmethod
    def from_battle(
        cls, battle: "AbstractBattle", *, perspective: str = "p1", tracker=None
    ) -> "Observation":
        """Build an observation from a live poke-env battle state.

        Falls back to the zero-fill shape-test path when the generated constant tables are missing
        (TablesNotBuilt) — real play REQUIRES built tables (and an agent from FogAgent.from_tables).
        `tracker` (encoding.tracker.BattleTracker, optional) fills the event-derived audit fields
        (last_used_move / booster / fallen / volatile counters / Future Sight-Wish / history
        snapshots+actions).
        """
        tables = None
        try:
            from fog.constants.tables import TABLES, TablesNotBuilt

            TABLES.n_species  # trigger lazy load
            tables = TABLES
        except Exception:
            tables = None

        opp_revealed = _empty_opp_revealed()
        if tables is not None and battle is not None:
            from fog.encoding.reveals import build_belief_input, opponent_views_from_battle

            opp_revealed = build_belief_input(opponent_views_from_battle(battle, tables), tables)

        ent, ent_ids, active_idx = _entities.encode_entities(
            battle, tables, opp_tokens=opp_revealed["tokens"], perspective=perspective,
            tracker=tracker,
        )
        fld = _field.encode_field(battle, perspective=perspective, tracker=tracker)
        hist, hist_len = _history.encode_history(
            battle, tables, perspective=perspective, tracker=tracker
        )
        mask = _entities.legality_mask(battle, perspective=perspective)
        return cls(
            entities=ent,
            entity_ids=ent_ids,
            field=fld,
            history=hist,
            history_len=hist_len,
            legality_mask=mask,
            active_idx=active_idx,
            opp_revealed=opp_revealed,
        )

    def validate(self) -> None:
        """Cheap shape assertions used in tests and the golden-battle integration test."""
        assert self.entities.shape[0] == 12
        assert self.field.shape[0] == DIMS.field_dim, (self.field.shape, DIMS.field_dim)
        assert self.history.ndim == 2 and self.history.shape[1] == DIMS.turn_dim
        assert self.legality_mask.shape == (14,)
        assert self.legality_mask.any(), "legality mask must be nonempty"


def _empty_opp_revealed() -> dict[str, Any]:
    """Temporary single-observation C1 placeholder until reveal extraction lands."""
    return {"tokens": np.full((6, 9), -1, dtype=np.int64), "masks": {}}
