"""Constants and lookup tables (spec §1). Build first — everything depends on them.

Two artifacts per table:
  * versioned JSON under `constants/generated/` (produced by `build.py` from the pinned Showdown
    data files), and
  * frozen numpy arrays exposed by `tables.py` for hot-path use.

The *fixed* enums that never change with the metagame (ACTIONS, STATUS, WEATHER, TERRAIN, categories)
live in `schema.py` and are real. The *generated* tables (SPECIES, MOVES, ITEMS, ABILITIES, SPREADS,
and per-species static features) are produced by `build.py`.
"""

from fog.constants import schema
from fog.constants.schema import (
    ACTIONS,
    N_ACTIONS,
    STATUS,
    TERRAIN,
    WEATHER,
    ActionType,
    MoveCategory,
)

__all__ = [
    "ACTIONS",
    "N_ACTIONS",
    "STATUS",
    "TERRAIN",
    "WEATHER",
    "ActionType",
    "MoveCategory",
    "schema",
]
