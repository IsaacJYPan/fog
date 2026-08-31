"""Leak-free subset of the optional per-action feature table.

``encode_action_features`` intentionally supports a rule-guided player and therefore includes
the exact SimpleHeuristics choice plus several intermediate rule decisions.  Independent models
must never consume those columns.  This module keeps only facts that can be calculated from the
ordinary current battle state and zeros every rule-decision column before a tensor reaches a
model.

The helper is additive and unused unless a run explicitly enables the public-action-feature
experiment.  Existing configurations and checkpoints are unchanged.
"""
from __future__ import annotations

import numpy as np

from fog.constants.schema import N_ACTIONS
from fog.encoding.action_features import ACTION_FEATURE_DIM, ACTION_FEATURE_NAMES

PUBLIC_ACTION_FEATURE_NAMES = (
    "legal",
    "ordinary_move",
    "tera_move",
    "switch",
    "forced",
    "slot_fraction",
    "move_power",
    "move_accuracy",
    "expected_hits",
    "physical",
    "special",
    "status_move",
    "same_type_bonus",
    "type_effectiveness",
    "physical_stat_ratio",
    "special_stat_ratio",
    "estimated_damage",
    "switch_matchup",
    "positive_switch_matchup",
    "active_matchup",
    "own_team_remaining",
    "opponent_team_remaining",
    "active_full_health",
    "opponent_full_health",
    "own_side_has_hazard",
    "can_tera",
)

ANSWER_DERIVED_ACTION_FEATURE_NAMES = tuple(
    name for name in ACTION_FEATURE_NAMES if name not in PUBLIC_ACTION_FEATURE_NAMES
)

_PUBLIC_INDICES = np.asarray(
    [ACTION_FEATURE_NAMES.index(name) for name in PUBLIC_ACTION_FEATURE_NAMES],
    dtype=np.int64,
)


def public_action_features(features: np.ndarray) -> np.ndarray:
    """Return a finite copy containing only public, answer-independent columns."""
    source = np.asarray(features, dtype=np.float32)
    if source.shape[-2:] != (N_ACTIONS, ACTION_FEATURE_DIM):
        raise ValueError(
            "action features must end with shape "
            f"({N_ACTIONS}, {ACTION_FEATURE_DIM}); got {source.shape}"
        )
    if not np.isfinite(source).all():
        raise ValueError("action features must be finite")
    clean = np.zeros_like(source, dtype=np.float32)
    clean[..., _PUBLIC_INDICES] = source[..., _PUBLIC_INDICES]
    return clean


__all__ = [
    "ANSWER_DERIVED_ACTION_FEATURE_NAMES",
    "PUBLIC_ACTION_FEATURE_NAMES",
    "public_action_features",
]
