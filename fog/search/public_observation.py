"""Public transcript identities and observation-conditioned chance likelihoods.

These are identities of the backend's public observation abstraction, not learned embeddings.
The legacy engine's missing mechanics/reveals remain approximation limits.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np


def public_observation_key(battle) -> str:
    """Hash public events only, rounding simulator HP to the spectator percentage display.

    Never include submitted actions, private requests, simulator serialization, exact HP or
    maximum HP. Pending events matter at mid-turn replacement decisions before turn finalization.
    """
    def censor(event):
        fields = list(event)
        if len(fields) > 3 and fields[1] in {"-damage", "-heal", "switch", "drag", "replace"}:
            column = 4 if fields[1] in {"switch", "drag", "replace"} else 3
            if len(fields) > column:
                match = re.match(r"^(\d+)/(\d+)(.*)$", fields[column])
                if match:
                    hp, maximum, suffix = match.groups()
                    percentage = math.ceil(100 * int(hp) / max(1, int(maximum)))
                    fields[column] = f"{percentage}/100{suffix}"
        return fields

    turns = [(int(turn), [censor(event) for event in observation.events])
             for turn, observation in sorted(battle.observations.items())]
    payload = {"turns": turns, "reveals": battle.revealed,
               "pending_events": [censor(event) for event in battle._pending_events],
               "pending_sides": sorted(battle.pending_sides()), "turn": int(battle.turn)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class PublicObservationLikelihood:
    """Aggregate P(public observation | k,j,a,b) without materializing L*K*J*A*B."""

    def __init__(self, grid):
        keys = getattr(grid, "leaf_public_keys", None)
        if keys is None or len(keys) != len(grid.leaf_obs):
            raise ValueError("public-belief conditioning requires explicit public observation keys")
        self.shape = grid.branch_prob.shape[:2]
        self.avail1 = np.asarray(getattr(grid, "avail1", np.ones(
            (self.shape[0], grid.branch_prob.shape[2]), dtype=bool)))
        self.avail2 = np.asarray(getattr(grid, "avail2", np.ones(
            (self.shape[1], grid.branch_prob.shape[3]), dtype=bool)))
        self.keys = list(dict.fromkeys(keys))
        index = {key: i for i, key in enumerate(self.keys)}
        self.leaf_group = np.asarray([index[key] for key in keys], dtype=np.int64)
        evidence = np.asarray(getattr(grid, "leaf_evidence", None))
        self.evidence = np.ones((len(self.keys), *self.shape), dtype=np.float64)
        if evidence.ndim == 3:
            for key, group in index.items():
                selected = evidence[self.leaf_group == group]
                if not np.all(selected == selected[0]):
                    raise ValueError("public observation key conflates different reveal evidence")
                self.evidence[group] = selected[0]
        positions = np.argwhere(grid.leaf_index >= 0)
        self.positions = positions
        self.groups = self.leaf_group[grid.leaf_index[tuple(positions.T)]] if len(positions) else []
        self.probabilities = grid.branch_prob[tuple(positions.T)].copy() if len(positions) else []
        pruned = getattr(grid, "pruned_mass", None)
        if pruned is not None and len(positions):
            self.probabilities *= 1.0 - pruned[tuple(positions[:, :4].T)]

    def likelihoods(self, sigma1, sigma2):
        result = np.zeros((len(self.keys), *self.shape), dtype=np.float64)
        if len(self.positions):
            k, j, a, b, _ = self.positions.T
            weight = self.probabilities * sigma1[k, a] * sigma2[j, b]
            np.add.at(result, (self.groups, k, j), weight)
        return result * self.evidence

    def posteriors(self, rho, sigma1, sigma2):
        """Leaf posteriors; zero denotes a structurally unreachable hypothetical observation.

        An observation produced only by zero-prior joint cells is impossible even under action
        trembles. Its value is multiplied by zero in every root expectation, and it must not be
        logged as a training belief. Actual observation updates still reject this case.
        """
        mass = self.likelihoods(sigma1, sigma2) * np.asarray(rho)[None]
        total = mass.sum(axis=(1, 2), keepdims=True)
        off_path = total.reshape(-1) <= 0
        if off_path.any():
            # Beliefs at zero-reach histories are undefined. Specify a small full-support
            # tremble for hypothetical value queries; actual observation_step never uses it.
            def tremble(strategy, available):
                uniform = available / np.maximum(available.sum(axis=1, keepdims=True), 1)
                return (1 - 1e-8) * strategy + 1e-8 * uniform

            alternative = self.likelihoods(tremble(sigma1, self.avail1),
                                           tremble(sigma2, self.avail2)) * np.asarray(rho)[None]
            mass[off_path] = alternative[off_path]
            total = mass.sum(axis=(1, 2), keepdims=True)
        posterior = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
        return posterior[self.leaf_group]

    def likelihood(self, key, sigma1, sigma2):
        if key not in self.keys:
            raise RuntimeError("observed public history is outside the simulated support")
        return self.likelihoods(sigma1, sigma2)[self.keys.index(key)]

    def transposed(self):
        """The same public histories, with both private/action axes exchanged."""
        import copy

        other = copy.copy(self)
        other.shape = self.shape[::-1]
        other.avail1, other.avail2 = self.avail2, self.avail1
        other.evidence = self.evidence.transpose(0, 2, 1).copy()
        other.positions = self.positions[:, [1, 0, 3, 2, 4]].copy()
        return other
