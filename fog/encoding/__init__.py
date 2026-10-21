"""Encode a player's battle view as entity, field, and history tensors.

Entity slots 0–5 hold the player's team; 6–11 hold the opponent's visible state.
Known/unknown flags distinguish unrevealed attributes from known absences. Nature and
EVs come from the submitted team, with EVs stored as stat-contribution units (EV // 4).
"""

from fog.encoding.dims import DIMS, Dims
from fog.encoding.observation import Observation

__all__ = ["DIMS", "Dims", "Observation"]
