"""Engine-side scripted opponents.

League ports of the poke-env scripted baselines for engine self-play. Only behaviors expressible
on engine state are ported:

  * Random       — uniform over legal action indices (mirrors poke_env RandomPlayer)
  * MaxDamage    — highest-base-power available move, else first switch (mirrors
                   MaxBasePowerPlayer: BASE power, deliberately not a full damage calc)

SimpleHeuristics is NOT ported (it leans on poke-env internals); it remains a Showdown-side eval
opponent only — an engine training run configured against it must fail loudly (ppo dispatch).

Policies work directly on engine state + the 14-way mask; their views are never encoded, which
keeps scripted-opponent decisions off the (Python-side) encoding budget entirely.
"""
from __future__ import annotations

import random

import numpy as np


class EngineRandomPolicy:
    """Uniform over legal actions. One RNG for the whole policy (seeded by the runner)."""

    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose(self, battle, role: str) -> int:
        legal = np.flatnonzero(battle.action_mask(role))
        return int(self.rng.choice(legal.tolist()))


class EngineMaxDamagePolicy:
    """MaxBasePowerPlayer semantics: strongest available move by BASE power; first legal switch
    when no move is available (forced switches). Tera is never chosen (matches the poke-env
    baseline, which does not terastallize)."""

    name = "MaxDamage"

    def __init__(self, tables=None, seed: int = 0) -> None:
        if tables is None:
            from fog.constants.tables import TABLES

            tables = TABLES
        self._power = {
            move_id: int(feats.get("base_power") or 0)
            for move_id, feats in tables.moves["features"].items()
        }
        self.rng = random.Random(seed)

    def choose(self, battle, role: str) -> int:
        mask = battle.action_mask(role)
        active = battle.active(role)
        best, best_power = None, -1
        for slot in range(4):
            if not mask[slot]:
                continue
            moves = list(active.moves)
            move_id = str(moves[slot].id).lower() if slot < len(moves) else ""
            power = self._power.get(move_id, 0)
            if power > best_power:
                best, best_power = slot, power
        if best is not None:
            return best
        switches = np.flatnonzero(mask[8:13])
        if len(switches):
            return int(8 + self.rng.choice(switches.tolist()))
        return 13


SCRIPTED_ENGINE_POLICIES = {
    "Random": EngineRandomPolicy,
    "MaxDamage": EngineMaxDamagePolicy,
}
