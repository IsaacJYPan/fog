"""Dense, non-farmable battle progress used by the reinforcement-learning curriculum.

The final win/loss remains the main objective.  This module supplies intermediate credit for
improving the health-and-Pokemon advantage.  Training uses *changes* in this score, rather than
paying repeatedly for a good position, so passing or extending a battle cannot manufacture reward.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _hp_fraction(mon: Any) -> float:
    if bool(getattr(mon, "fainted", False)):
        return 0.0
    value = getattr(mon, "current_hp_fraction", None)
    if value is None:
        hp = float(getattr(mon, "hp", 0.0) or 0.0)
        max_hp = float(getattr(mon, "maxhp", 0.0) or 0.0)
        value = hp / max_hp if max_hp > 0 else 1.0
    return min(1.0, max(0.0, float(value)))


def _side_score(mons: Iterable[Any], *, missing_slots_are_healthy: bool) -> tuple[float, float]:
    mons = list(mons)
    health = sum(_hp_fraction(mon) for mon in mons)
    alive = sum(_hp_fraction(mon) > 0.0 for mon in mons)
    if missing_slots_are_healthy:
        missing = max(0, 6 - len(mons))
        health += missing
        alive += missing
    return health / 6.0, alive / 6.0


def battle_progress_score(battle: Any, role: str | None = None) -> float:
    """Return a stable score in [-1, 1] from the learner's perspective.

    Internal training battles expose both complete teams.  A network Showdown battle only exposes
    encountered opponents, so unseen slots are conservatively counted as alive at full health.
    Revealing a new opponent therefore does not create a fake reward jump.
    """
    side = getattr(battle, "side", None)
    if role is not None and callable(side):
        opponent_role = "p2" if role == "p1" else "p1"
        mine = side(role).pokemon
        theirs = side(opponent_role).pokemon
        my_health, my_alive = _side_score(mine, missing_slots_are_healthy=False)
        opp_health, opp_alive = _side_score(theirs, missing_slots_are_healthy=False)
    else:
        mine = list((getattr(battle, "team", None) or {}).values())
        theirs = list((getattr(battle, "opponent_team", None) or {}).values())
        my_health, my_alive = _side_score(mine, missing_slots_are_healthy=True)
        opp_health, opp_alive = _side_score(theirs, missing_slots_are_healthy=True)

    score = 0.5 * (my_health - opp_health) + 0.5 * (my_alive - opp_alive)
    return min(1.0, max(-1.0, float(score)))
