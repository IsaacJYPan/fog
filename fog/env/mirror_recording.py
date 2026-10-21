"""Opponent-view recording and frozen-probe leakage measurements."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class MirrorRecorder:
    """Keep mirror fragments batch-compatible as optional providers fail or recover."""

    def __init__(self, report_error: Callable[[str, BaseException], None]) -> None:
        self.enabled: bool | None = None
        self.last_error: str | None = None
        self.probe: Any = None
        self.family_weights: dict[str, float] | None = None
        self._report_error = report_error

    def build(
        self,
        battle: Any,
        obs: Any,
        ledger: Any,
        *,
        recording: bool,
        private_truth_provider: Callable[[Any], Any] | None = None,
        team_builder: Any = None,
    ) -> tuple[dict, dict] | None:
        """Unbatched (mirror_frag, my_view) numpy pair, or None when mirror recording is off/broken.

        Auto-disables permanently on the first failure (before anything was recorded) so every frag
        in a batch has identical structure; after the first success, failures degrade to a zero-fill
        mirror with -1 labels (no loss contribution) instead of breaking frag structure.
        """
        if self.enabled is False or (not recording and self.probe is None):
            return None
        try:
            from fog.constants.tables import TABLES
            from fog.encoding.mirror import build_opponent_view_np
            from fog.encoding.reveals import tokens_from_views

            team = getattr(battle, "team", None) or {}
            my_keys = [str(k) for k in team]
            my_species = [TABLES.species_index(_species_id(m)) for m in team.values()]
            my_view_tokens = tokens_from_views(
                ledger.views(my_keys, my_species) if ledger else []
            )
            my_truth_tokens = None
            if private_truth_provider is not None:
                try:
                    my_truth_tokens = private_truth_provider(battle)
                except Exception as exc:  # noqa: BLE001 - provider boundary; mirror records failure
                    self._report_error("mirror_private_truth_provider", exc)
                    my_truth_tokens = None
            if my_truth_tokens is None:
                paste = getattr(team_builder, "last_paste", None)
            else:
                paste = None
            if paste:
                from fog.env.team_truth import truth_tokens_from_paste

                my_truth_tokens = truth_tokens_from_paste(paste, TABLES)
            elif my_truth_tokens is None:
                my_truth_tokens = _own_truth_tokens(battle, TABLES)
            out = build_opponent_view_np(obs, my_view_tokens, my_truth_tokens, TABLES)
            if self.enabled is None:
                self.enabled = True
            return out
        except Exception as exc:  # noqa: BLE001 - optional mirror boundary; error retained below
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.enabled is None:
                self.enabled = False   # never worked: drop the opp_view key entirely
                return None
            return _zero_mirror(obs)         # worked before: keep frag structure, no labels

    def leakage_ce(self, mirror: tuple[dict, dict] | None, device) -> float | None:
        """Frozen-probe self-readability CE; None when labels or a probe are unavailable."""
        if self.probe is None or mirror is None:
            return None
        try:
            import torch

            mirror_frag, my_view = mirror
            # Measuring an unlabelled fallback as CE=0 would fabricate a leakage spike.
            # Leave a gap for the rollout buffer to forward-fill.
            sp_labels = np.asarray((my_view.get("labels") or {}).get("species", -1))
            if not (sp_labels >= 0).any():
                return None
            mb = {
                "entities": _t1(mirror_frag["entities"], torch.float32, device),
                "entity_ids": _t1(mirror_frag["entity_ids"], torch.long, device),
                "field": _t1(mirror_frag["field"], torch.float32, device),
                "history": _t1(mirror_frag["history"], torch.float32, device),
                "history_ids": _t1(mirror_frag["history_ids"], torch.long, device),
                "history_state": _t1(mirror_frag["history_state"], torch.float32, device),
                "history_len": torch.as_tensor(
                    [mirror_frag["history_len"]], dtype=torch.long, device=device
                ),
                "active_idx": _t1(mirror_frag["active_idx"], torch.long, device),
            }
            mv = {
                "tokens": _t1(my_view["tokens"], torch.long, device),
                "masks": {k: _t1(v, torch.bool, device) for k, v in my_view["masks"].items()},
                "labels": {k: _t1(v, torch.long, device) for k, v in my_view["labels"].items()},
            }
            ce = self.probe.leakage_ce(mb, mv, family_weights=self.family_weights)
            return float(ce[0])
        except Exception as exc:  # noqa: BLE001 - optional metric boundary; error is recorded
            self._report_error("leakage_measurement", exc)
            return None  # leakage measurement is never battle-critical


def _species_id(mon) -> str:
    from fog.encoding.events import submitted_species_id

    return submitted_species_id(getattr(mon, "species", ""))


def _own_truth_tokens(battle, tables):
    """My own full team as [6,15] truth tokens for self-belief labels."""

    from fog.constants.schema import TYPE_INDEX
    from fog.encoding.events import submitted_species_id, to_id
    from fog.encoding.reveals import SlotView, tokens_from_views

    views = []
    for mon in list((getattr(battle, "team", None) or {}).values())[:6]:
        v = SlotView()
        sp = getattr(mon, "species", None)
        if sp:
            v.species = tables.species_index(submitted_species_id(sp))
        item = getattr(mon, "item", None)
        if item and to_id(item) != "unknownitem":
            v.item = tables.item_index(to_id(item))
        ability = getattr(mon, "ability", None)
        if ability:
            v.ability = tables.ability_index(to_id(ability), default=None)
        for mid in list(getattr(mon, "moves", None) or {})[:4]:
            idx = tables.move_index(to_id(mid), default=None)
            if idx is not None and idx not in v.moves:
                v.moves.append(idx)
        if sp:
            from fog.encoding.entities import _own_nature_evs

            v.nature, v.evs = _own_nature_evs(mon, to_id(sp), tables)
        tt = getattr(mon, "tera_type", None)
        if tt is not None:
            v.tera = TYPE_INDEX.get(to_id(getattr(tt, "name", tt)), 18)
        views.append(v)
    while len(views) < 6:
        views.append(SlotView())
    return tokens_from_views(views)


def _zero_mirror(obs) -> tuple[dict, dict]:
    """Unlabelled zero observation with the same nested keys as a recorded mirror."""

    from fog.constants.schema import SET_N_ATTRS
    from fog.constants.tables import TABLES
    from fog.encoding.dims import DIMS

    frag = {
        "entities": np.zeros_like(obs.entities),
        "entity_ids": np.zeros_like(obs.entity_ids),
        "field": np.zeros_like(obs.field),
        "history": np.zeros_like(obs.history),
        "history_ids": np.zeros((len(obs.history), DIMS.hist_id_fields), dtype=np.int64),
        "history_state": np.zeros((len(obs.history), DIMS.hist_state), dtype=np.float32),
        "history_len": np.int64(0),
        "active_idx": np.array([0, 6], dtype=np.int64),
        # Opponent request legality is private and unavailable in a mirror fallback.  Keep the
        # structure identical to real mirrors with a constant, non-informative public placeholder.
        "legality_mask": np.ones(14, dtype=bool),
    }
    tokens = np.full((6, SET_N_ATTRS), -1, dtype=np.int64)
    masks = {
        "species": np.ones((6, TABLES.n_species), dtype=bool),
        "ability": np.ones((6, TABLES.n_abilities), dtype=bool),
        "move": np.ones((6, TABLES.n_moves), dtype=bool),
    }
    labels = {
        "species": np.full(6, -1, dtype=np.int64), "item": np.full(6, -1, dtype=np.int64),
        "ability": np.full(6, -1, dtype=np.int64), "move": np.full((6, 4), -1, dtype=np.int64),
        "nature": np.full(6, -1, dtype=np.int64),
        "ev": np.full((6, 6), -1, dtype=np.int64),
        "tera": np.full(6, -1, dtype=np.int64),
    }
    return frag, {
        "tokens": tokens,
        "supervision_tokens": tokens.copy(),
        "supervision_consistent": np.zeros(6, dtype=bool),
        "masks": masks,
        "labels": labels,
    }


def _t1(a, dtype, device):
    import torch

    t = torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)
    return (t.to(device) if device is not None else t).unsqueeze(0)
