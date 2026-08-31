"""Two-sided simultaneous-action search on authoritative in-memory Showdown.

This module is the direct-simulator bridge for the pure regret solver.  It intentionally starts
with the one-world-per-side case: both action axes and every finite chance branch are authoritative,
but hidden-team candidate expansion is not claimed here.  That narrower boundary is still useful:
it replaces the old fixed-reply one-ply approximation with a genuine simultaneous zero-sum solve
and makes the remaining public-belief candidate substitution work explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fog.constants.schema import N_ACTIONS
from fog.env.actions import action_legality_mask, map_action_to_order
from fog.env.showdown_local_runner import LocalOrderFactory
from fog.search.two_sided import TwoSidedSolution, two_sided_regret_solve


@dataclass(frozen=True)
class DirectTwoSidedResult:
    """One solved direct-Showdown decision and its simulation accounting."""

    action: int
    action_dist: np.ndarray
    payoff: np.ndarray
    solution: TwoSidedSolution
    simulations: int
    chance_branches: int
    pruned_mass_max: float


def _choice(view: Any, action: int) -> str:
    order = map_action_to_order(view, action, LocalOrderFactory())
    choice = str(getattr(order, "message", order)).strip()
    return choice.removeprefix("/choose ")


def _terminal_value(local: Any, role: str, winner: str | None) -> float:
    if winner is None:
        return 0.0
    username = str(getattr(local.views[role], "player_username", ""))
    return 1.0 if str(winner) == username else -1.0


def _leaf_value(agent: Any, local: Any, role: str, branch: Any, device: Any) -> float:
    import torch

    from fog.encoding.batch import observation_to_batch
    from fog.encoding.observation import Observation

    view = local.branch_view(role, branch)
    obs = Observation.from_battle(view)
    obs.validate()
    with torch.no_grad():
        output = agent(observation_to_batch(obs, device))
    return float(output.value.reshape(-1)[0].item())


def direct_two_sided_action(
    agent: Any,
    battle: Any,
    *,
    role: str,
    device: Any,
    iters: int = 256,
    floor: float = 0.02,
    min_probability: float = 0.0,
    max_branches: int | None = None,
) -> DirectTwoSidedResult:
    """Solve the current direct-Showdown action pair with exact chance enumeration.

    ``battle`` is the acting player's censored poke-env view.  The direct local runner attaches its
    authoritative session as transport context.  No network server is used and no hidden state is
    inserted into the model observation.
    """
    local = getattr(battle, "_fog_local_battle", None)
    if local is None:
        raise RuntimeError("direct two-sided search requires an in-memory LocalShowdownBattle")
    if role not in {"p1", "p2"}:
        raise ValueError(f"role must be p1 or p2, got {role!r}")
    opponent_role = "p2" if role == "p1" else "p1"
    own_actions = np.flatnonzero(np.asarray(action_legality_mask(battle), dtype=bool))
    opponent_view = local.views[opponent_role]
    opponent_request = local.update.requests.get(opponent_role)
    opponent_pending = bool(opponent_request and not opponent_request.get("wait"))
    if opponent_pending:
        opponent_actions = np.flatnonzero(
            np.asarray(action_legality_mask(opponent_view), dtype=bool)
        )
    else:
        opponent_actions = np.asarray([-1], dtype=np.int64)
    if own_actions.size == 0 or opponent_actions.size == 0:
        raise RuntimeError("direct search received an empty legal-action axis")

    payoff = np.zeros((1, 1, len(own_actions), len(opponent_actions)), dtype=np.float64)
    simulations = 0
    chance_branches = 0
    pruned_mass_max = 0.0
    for a_col, own_action in enumerate(own_actions):
        own_choice = _choice(battle, int(own_action))
        for b_col, opponent_action in enumerate(opponent_actions):
            opponent_choice = (
                None if int(opponent_action) < 0 else _choice(opponent_view, int(opponent_action))
            )
            choices = (
                (own_choice, opponent_choice)
                if role == "p1"
                else (opponent_choice, own_choice)
            )
            enumeration = local.client.enumerate(
                local.state,
                choices,
                min_probability=min_probability,
                max_branches=max_branches,
            )
            simulations += 1
            chance_branches += len(enumeration.branches)
            pruned_mass_max = max(pruned_mass_max, enumeration.pruned_mass.decimal)
            value = 0.0
            for branch in enumeration.branches:
                leaf = (
                    _terminal_value(local, role, branch.winner)
                    if branch.ended
                    else _leaf_value(agent, local, role, branch, device)
                )
                value += branch.probability.decimal * leaf
            payoff[0, 0, a_col, b_col] = value

    avail1 = np.ones((1, len(own_actions)), dtype=bool)
    avail2 = np.ones((1, len(opponent_actions)), dtype=bool)
    solution = two_sided_regret_solve(
        lambda _sigma1, _sigma2: payoff,
        np.ones((1, 1), dtype=np.float64),
        avail1,
        avail2,
        iters=iters,
        floor=floor,
    )
    compact = solution.sigma1[0]
    action_dist = np.zeros(N_ACTIONS, dtype=np.float64)
    action_dist[own_actions] = compact
    action = int(own_actions[int(np.argmax(compact))])
    return DirectTwoSidedResult(
        action=action,
        action_dist=action_dist,
        payoff=payoff,
        solution=solution,
        simulations=simulations,
        chance_branches=chance_branches,
        pruned_mass_max=pruned_mass_max,
    )
