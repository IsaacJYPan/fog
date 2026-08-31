"""Explicit player-observation boundary for model inputs.

Only ``PUBLIC`` and ``OWN_PRIVATE`` facts may be encoded directly.  Belief labels may be built
from later revelations or self-play truth, but those values are never observation inputs at the
earlier decision point.
"""
from __future__ import annotations

import enum
from typing import Any


class Provenance(enum.IntEnum):
    PUBLIC = 0
    OWN_PRIVATE = 1
    INFERRED = 2
    SIMULATOR_INTERNAL = 3


class Visibility(enum.IntEnum):
    UNREVEALED = 0
    KNOWN_VALUE = 1
    KNOWN_NONE = 2
    INAPPLICABLE = 3


class DecisionPhase(enum.IntEnum):
    NORMAL_MOVE = 0
    FORCED_SWITCH = 1
    MID_TURN_SWITCH = 2
    TEAM_PREVIEW = 3
    RETRY_AFTER_INVALID_CHOICE = 4
    OTHER_DECISION = 5


DIRECT_MODEL_PROVENANCE = frozenset({Provenance.PUBLIC, Provenance.OWN_PRIVATE})


def decision_phase_from_battle(battle: Any) -> DecisionPhase:
    """Classify only request-visible control state, without consulting simulator truth."""
    request = getattr(battle, "last_request", None)
    if callable(request):
        request = request()
    if not isinstance(request, dict):
        return DecisionPhase.NORMAL_MOVE
    if request.get("teamPreview"):
        return DecisionPhase.TEAM_PREVIEW
    force_switch = request.get("forceSwitch")
    if force_switch is True or (
        isinstance(force_switch, list) and any(bool(value) for value in force_switch)
    ):
        return DecisionPhase.FORCED_SWITCH
    # poke-env/Showdown expose a replacement request after a pivot through forceSwitch in normal
    # play.  Keep a separate enum for collectors that can prove the pivot boundary; do not guess it
    # from hidden engine state here.
    if request.get("wait"):
        return DecisionPhase.OTHER_DECISION
    return DecisionPhase.NORMAL_MOVE


__all__ = [
    "DIRECT_MODEL_PROVENANCE",
    "DecisionPhase",
    "Provenance",
    "Visibility",
    "decision_phase_from_battle",
]
