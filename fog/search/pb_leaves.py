"""Simulate an action pair for one candidate-team pair in the search grid.

Both actions come from the solver. Chance outcomes are enumerated, and each
branch stops when either player next needs to choose, including forced replacements.
The value network evaluates those leaves; no reply policy selects another action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BranchOutcome:
    """One chance branch of one committed (k, j, a, b) cell.

    Exactly one of `terminal` / `leaf` is set. Probabilities are renormalized
    over kept branches; dropped mass is returned separately as `pruned_mass`.
    """
    probability: float
    terminal: float | None       # game over inside the branch: +1 / -1 / 0 for `role`
    leaf: Any | None             # live EngineBattle fork stopped at the next decision


def commit_pair_branches(
    root: Any,
    role: str,
    my_action_idx: int,
    opp_action_idx: int | None = None,
    *,
    min_prob: float = 0.0,
    max_branches: int | None = None,
) -> tuple[list[BranchOutcome], float]:
    """Commit both public actions on `root` and enumerate the turn's chance branches.

    `root` is a disposable world fork. Its battle state stays unchanged, but its
    random stream seeds a child fork for each chance outcome.

    Supply `opp_action_idx` when the opponent has a pending choice; otherwise use
    None. Finished branches have a terminal payoff (+1 win, -1 loss, 0 tie from
    `role`'s perspective); other branches hold a live battle at the next decision.

    Returns `(outcomes, pruned_mass)`. All chance branches are kept by default.
    With pruning enabled, kept probabilities are renormalized to sum to one and
    callers should record the dropped mass.
    """
    opp_role = "p2" if role == "p1" else "p1"
    pending = set(root.pending_sides())
    if role not in pending:
        raise ValueError(f"{role} is not pending (pending={sorted(pending)}) — this is not "
                         f"{role}'s decision to commit")
    moves = {role: root.engine_action(role, int(my_action_idx))}
    if opp_role in pending:
        if opp_action_idx is None:
            raise ValueError(f"{opp_role} is pending: opp_action_idx is required (the opponent's "
                             "simultaneous action is a solver variable, not a modeled reply)")
        moves[opp_role] = root.engine_action(opp_role, int(opp_action_idx))
    elif opp_action_idx is not None:
        raise ValueError(f"{opp_role} is not pending: opp_action_idx must be None")

    branches, pruned_mass = root.step_branches(moves, min_prob=min_prob,
                                               max_branches=max_branches)
    outcomes: list[BranchOutcome] = []
    for probability, fork in branches:
        if fork.finished:
            terminal = (1.0 if fork.winner == role
                        else -1.0 if fork.winner is not None else 0.0)
            outcomes.append(BranchOutcome(probability=probability, terminal=terminal, leaf=None))
        else:
            outcomes.append(BranchOutcome(probability=probability, terminal=None, leaf=fork))
    return outcomes, pruned_mass
