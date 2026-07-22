"""Branch leaf play for the two-sided public-belief search (PROPOSAL_IMPL_PLAN Phase B(ii);
PROJECT_DESIGN_PROPOSAL.md §10 steps 1-5, §10.4, §10.5).

One cell of the search grid is "our candidate world k, opponent candidate world j, our public
action a, their public action b". This module commits one such simultaneous action pair on a
(k, j) root fork and returns the turn's chance outcomes as exactly-weighted branches
(`EngineBattle.step_branches` — enumerate and branch, never one frozen sample).

The load-bearing rule (§10.5, settled decision 22): **a branch stops at the FIRST state where any
player owes a new decision** — a completed turn, or a mid-turn state with a forced replacement
pending for EITHER side. A forced replacement is a real decision with signaling content; no
scripted stand-in ever chooses it inside a leaf. (The frozen baseline `fog/search/engine_step.py`
resolves opponent replacements with a reply policy — that is the old one-sided interim's
semantics, deliberately NOT reproduced here. The value network owns pending-replacement leaves.)

Both action indices come from the solver's enumeration of each side's public labels (the Phase C
grid builder), never from a policy or script: the opponent's simultaneous action is a solver
variable, not a modeled reply (§21.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BranchOutcome:
    """One chance branch of one committed (k, j, a, b) cell.

    Exactly one of `terminal` / `leaf` is set. `probability` is renormalized over the KEPT
    branches of this cell (the dropped remainder is the `pruned_mass` returned alongside — log
    it, invariant 19.16)."""
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

    `root` is a disposable (k, j) world fork (`fork_for_own_world`); it is never advanced —
    `step_branches` builds one child fork per kept chance outcome, each with a fresh independent
    random future (settled decision 21). Only `root`'s random stream is consumed (fork seeding),
    so results are deterministic given `root`'s rng state.

    `opp_action_idx` is required exactly when the opponent is also pending (a normal simultaneous
    turn). At a one-sided decision — e.g. `role`'s own forced replacement while the opponent
    waits — it must be None. Passing an index for a non-pending opponent raises: silently
    ignoring it would hide grid-builder indexing bugs.

    Classification per kept branch: a finished game becomes `terminal` (+1 win / -1 loss / 0 for
    `role`, the project's standard convention); everything else is a live `leaf` exactly as the
    branch fork stands — a completed turn (both sides pending the next decision) or a pending
    forced replacement (§10.5). No further stepping happens here.

    Returns `(outcomes, pruned_mass)` with kept probabilities summing to one.  The low-level
    primitive is exact by default: it keeps every native engine branch.  Search budgets may pass
    explicit pruning limits (and must record the resulting `pruned_mass`), but merely calling this
    function never turns a rare valid outcome into an undocumented approximation.
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
