"""One canonical conversion from :class:`Observation` objects to model tensor batches.

Live play, engine rollouts, search, and evaluation must agree on padding, optional history
sidecars, active slots, action features, and belief masks.  Keeping that policy here prevents a
new caller from quietly inventing a slightly different observation schema.
"""
from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import torch

from fog.encoding.dims import DIMS


class RevealedBatch(TypedDict):
    tokens: torch.Tensor
    masks: dict[str, torch.Tensor]


class ObservationBatch(TypedDict):
    entities: torch.Tensor
    entity_ids: torch.Tensor
    field: torch.Tensor
    history: torch.Tensor
    history_ids: torch.Tensor
    history_state: torch.Tensor
    history_len: torch.Tensor
    active_idx: torch.Tensor
    legality_mask: torch.Tensor
    action_features: torch.Tensor
    opp_revealed: RevealedBatch


def _tensor(array: Any, dtype: torch.dtype, device: Any) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)


def revealed_to_torch(
    revealed: dict[str, Any] | None,
    *,
    device: Any,
    batched: bool = False,
) -> RevealedBatch:
    """Convert one revealed-team payload, using explicit unknown values when absent."""
    if revealed is None:
        return {
            "tokens": torch.full((1, 6, 15), -1, dtype=torch.long, device=device),
            "masks": {},
        }
    tokens = _tensor(np.asarray(revealed["tokens"]), torch.long, device)
    if not batched and tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    masks: dict[str, torch.Tensor] = {}
    for name, mask in (revealed.get("masks") or {}).items():
        value = _tensor(np.asarray(mask), torch.bool, device)
        masks[name] = value if batched or value.ndim != 2 else value.unsqueeze(0)
    return {"tokens": tokens, "masks": masks}


def collate_observations(observations: list[Any], device: Any) -> ObservationBatch:
    """Stack unbatched observations, front-padding history to this batch's maximum length."""
    from fog.encoding.action_features import ACTION_FEATURE_DIM
    from fog.encoding.history import pad_or_truncate_front

    if not observations:
        raise ValueError("cannot collate an empty observation list")
    max_turns = max(observation.history.shape[0] for observation in observations)

    def history_ids(observation: Any) -> np.ndarray:
        if observation.history_ids is not None:
            return observation.history_ids
        return np.zeros((len(observation.history), DIMS.hist_id_fields), dtype=np.int64)

    def history_state(observation: Any) -> np.ndarray:
        if observation.history_state is not None:
            return observation.history_state
        return np.zeros((len(observation.history), DIMS.hist_state), dtype=np.float32)

    def action_features(observation: Any) -> np.ndarray:
        if observation.action_features is not None:
            return observation.action_features
        return np.zeros((14, ACTION_FEATURE_DIM), dtype=np.float32)

    mask_keys = set((observations[0].opp_revealed or {}).get("masks", {}) or {})
    for observation in observations:
        observed_keys = set((observation.opp_revealed or {}).get("masks", {}) or {})
        if observed_keys != mask_keys:
            raise ValueError("inconsistent opp_revealed mask keys in one observation batch")

    revealed_tokens = np.stack([
        (observation.opp_revealed or {}).get(
            "tokens", np.full((6, 15), -1, dtype=np.int64)
        )
        for observation in observations
    ])
    revealed_masks = {
        key: _tensor(
            np.stack([observation.opp_revealed["masks"][key] for observation in observations]),
            torch.bool,
            device,
        )
        for key in mask_keys
    }
    return {
        "entities": _tensor(np.stack([o.entities for o in observations]), torch.float32, device),
        "entity_ids": _tensor(np.stack([o.entity_ids for o in observations]), torch.long, device),
        "field": _tensor(np.stack([o.field for o in observations]), torch.float32, device),
        "history": _tensor(
            np.stack([pad_or_truncate_front(o.history, max_turns) for o in observations]),
            torch.float32,
            device,
        ),
        "history_ids": _tensor(
            np.stack([pad_or_truncate_front(history_ids(o), max_turns) for o in observations]),
            torch.long,
            device,
        ),
        "history_state": _tensor(
            np.stack([pad_or_truncate_front(history_state(o), max_turns) for o in observations]),
            torch.float32,
            device,
        ),
        "history_len": _tensor(
            np.asarray([min(int(o.history_len), max_turns) for o in observations]),
            torch.long,
            device,
        ),
        "active_idx": _tensor(
            np.stack([
                o.active_idx if o.active_idx is not None else np.array([0, 6], dtype=np.int64)
                for o in observations
            ]),
            torch.long,
            device,
        ),
        "decision_phase": _tensor(
            np.asarray([int(getattr(o, "decision_phase", 0)) for o in observations]),
            torch.long,
            device,
        ),
        "legality_mask": _tensor(
            np.stack([o.legality_mask for o in observations]), torch.bool, device
        ),
        "action_features": _tensor(
            np.stack([action_features(o) for o in observations]), torch.float32, device
        ),
        "opp_revealed": {
            "tokens": _tensor(revealed_tokens, torch.long, device),
            "masks": revealed_masks,
        },
    }


def observation_to_batch(observation: Any, device: Any) -> ObservationBatch:
    """Canonical one-observation specialization used by live players."""
    return collate_observations([observation], device)


__all__ = [
    "ObservationBatch",
    "RevealedBatch",
    "collate_observations",
    "observation_to_batch",
    "revealed_to_torch",
]
