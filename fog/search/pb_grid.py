"""The K x J cell grid for two-sided public-belief search (PROPOSAL_IMPL_PLAN Phase C;
PROJECT_DESIGN_PROPOSAL.md §10 table construction, §13 budgets, settled decision 14).

One real decision needs, for every pair (our candidate world k, opponent candidate world j) and
every available public-action pair (a, b), the turn's chance branches simulated ONCE and cached —
regret iterations later re-weight and re-value, they never re-simulate (invariant 19.13).

Structure decisions carried from the proposal:

  * **Shared opponent particles** (settled 14): the J opponent worlds are sampled once by the
    caller and every one of our K candidate worlds faces the SAME J worlds. The old one-sided
    code paired each own-world with its own opponent draw, which confounds private-state effects
    with opponent-draw luck. This module receives `own_tokens [K, 6, 9]` and
    `opp_tokens [J, 6, 9]` and builds the full K x J fork matrix.
  * **Public-action labels for BOTH sides** (proposal §7.3): strategies and posteriors live in
    the space of what the opponent observes (move ids, switch species), not action-slot indices.
    A world that lacks a label cannot play it (`avail` masks), so observing that label eliminates
    the world. Our label availability may depend only on k, the opponent's only on j — the solver
    factorizes on that, so the builder VERIFIES it and raises if the engine ever disagrees.
  * **Branches, not samples** (proposal §10.4/§10.5): each cell is committed with
    `commit_pair_branches`, so its outcomes are exactly-weighted chance branches stopped at the
    next pending decision, with pruned probability mass recorded per cell (invariant 19.16).
  * **Leaves are censored observations** (invariant 19.12): each live leaf is stored as the
    acting player's fog Observation (root reveals plus whatever the simulated turn itself
    revealed — never the sampled world's truth). Discrete branch evidence for §10.6 leaf-belief
    updates is recoverable from those observations' reveal tokens; a richer per-branch evidence
    record is deferred until Phase G defines its consumer (recorded deviation from the original
    plan sketch — the Observation already carries the reveal facts).

The builder handles OUR decision whether or not the opponent is also pending (PROPOSAL_IMPL_PLAN
Phase F2, proposal §10.5):

  * both sides pending  -> a normal simultaneous turn, the full K x J x A x B grid;
  * only `role` pending -> a one-sided forced replacement / pivot (U-turn, Baton Pass, Eject, or a
    faint with several live bench Pokemon) that only WE owe. The opponent has no choice this step,
    so its public-label axis collapses to a single synthetic `<observe>` column (`avail2` all
    True, B = 1) and every cell commits with `opp_action_idx=None`. The B = 1 solve degenerates to
    a plain type-conditioned replacement strategy — but it is still a real solved decision whose
    `sigma1` carries signaling content for the persistent belief table (a forced choice among
    SEVERAL legal replacements DOES carry likelihood, §14), never a scripted stand-in (settled 22).

A decision we do NOT own (only the opponent pending) stays an error: the caller skips the opponent's
solo replacement (its behavioral likelihood is unmodeled there — the honest §7.2 floor) and never
asks this builder to solve a move that is not ours.

Simulation failures are tolerated the way the baseline tolerates them: the cell is marked in
`failed` and gets no branches; the value layer (Phase E) substitutes the root value for failed
cells and reports the count. A flood of failures signals a reconstruction bug, not noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from fog.search.pb_leaves import commit_pair_branches

# The opponent's synthetic single public label at a one-sided (our) decision: it owes no action
# this step, so there is exactly one "column" — the solver sees B = 1 and the belief update takes
# no opponent likelihood (proposal §10.5, F2).
OBSERVE_LABEL = "<observe>"


def public_label(fork, role: str, action_idx: int) -> str:
    """What the opponent observes if `role` plays `action_idx` in this world: kind-prefixed
    engine action string, lowercased. Same convention as the frozen baseline's private helper
    (fog/search/joint.py `_public_label`) — duplicated here so new code never imports from the
    frozen module; the lowercasing bug history lives there."""
    from fog.constants.schema import ACTIONS, ActionType

    kind, _ = ACTIONS[action_idx]
    prefix = "s" if kind is ActionType.SWITCH else ("f" if kind is ActionType.FORCED else "m")
    return f"{prefix}:{fork.engine_action(role, action_idx)}".lower()


@dataclass
class PBGrid:
    """Everything one decision's solver needs, simulated once. Shapes: K own worlds, J opponent
    worlds, A our public labels, B their public labels, M widest branch fan-out.

    For cell (k, j, a, b), branch m exists where `branch_prob[k, j, a, b, m] > 0`; it is terminal
    when `leaf_index[...] < 0` (payoff in `terminal_value`, our perspective), otherwise
    `leaf_obs[leaf_index[...]]` is its censored leaf observation."""
    labels1: list[str]           # our public-action label space (union over k)
    labels2: list[str]           # the opponent's (union over j)
    avail1: np.ndarray           # [K, A] bool — world k can play label a
    avail2: np.ndarray           # [J, B] bool
    local_action1: np.ndarray    # [K, A] int — label column -> real action index in world k; -1
    local_action2: np.ndarray    # [J, B] int
    branch_prob: np.ndarray      # [K, J, A, B, M] float, 0 where no branch
    terminal_value: np.ndarray   # [K, J, A, B, M] float, NaN where not terminal
    leaf_index: np.ndarray       # [K, J, A, B, M] int into leaf_obs, -1 where terminal/absent
    leaf_obs: list               # flat censored Observations for batched value forwards
    pruned_mass: np.ndarray      # [K, J, A, B] float — dropped branch probability per cell
    failed: np.ndarray           # [K, J, A, B] bool — simulation raised; value layer substitutes
    sims_run: int                # committed (k, j, a, b) pairs (budget accounting — report it)


def build_pb_grid(
    battle,
    role: str,
    own_tokens: np.ndarray,
    opp_tokens: np.ndarray,
    *,
    seed: int = 0,
    min_prob: float = 0.005,
    max_branches: int | None = 12,
) -> PBGrid:
    """Simulate the full K x J x A x B branch table for one simultaneous decision on `battle`.

    `own_tokens` are the K candidate versions of OUR team the opponent could believe in (the
    own-state sources' token grids); `opp_tokens` are the J shared opponent candidates (sampled
    once from the belief chain, deduplicated and weighted by the CALLER — this builder treats
    both sets as given identities so the persistent belief table keys stay stable, §7.4).
    Never mutates `battle`."""
    from fog.encoding.observation import Observation

    opp_role = "p2" if role == "p1" else "p1"
    pending = set(battle.pending_sides())
    if role not in pending:
        raise ValueError(
            f"grid solves the decision `role`={role} owns, but pending={sorted(pending)} does not "
            "include it — a decision we do not own (only the opponent pending) is the caller's "
            "belief-update skip, not a grid solve (F2)")
    one_sided = pending == {role}
    if not one_sided and pending != {"p1", "p2"}:
        raise ValueError(f"unexpected pending set {sorted(pending)} for a grid on {role}")
    rng = random.Random(seed)
    own_tokens = np.asarray(own_tokens, dtype=np.int64)
    opp_tokens = np.asarray(opp_tokens, dtype=np.int64)
    n_k, n_j = len(own_tokens), len(opp_tokens)

    # ---- K x J root forks + per-fork label maps for both sides -------------------------------
    forks: list[list] = []
    maps1: list[list[dict]] = []      # [k][j] -> {our label: action idx}
    maps2: list[list[dict]] = []      # [k][j] -> {their label: action idx}
    for k in range(n_k):
        forks.append([])
        maps1.append([])
        maps2.append([])
        for j in range(n_j):
            fork = battle.fork_for_own_world(
                role, own_tokens[k], opp_world_tokens=opp_tokens[j],
                seed=rng.randrange(2**31),
            )
            forks[k].append(fork)
            maps1[k].append({public_label(fork, role, int(x)): int(x)
                             for x in np.flatnonzero(fork.action_mask(role))})
            # one-sided: the opponent owes nothing this step -> a single synthetic column, no real
            # action index (cells commit with opp_action_idx=None). Its local_action stays -1.
            maps2[k].append({OBSERVE_LABEL: -1} if one_sided else
                            {public_label(fork, opp_role, int(x)): int(x)
                             for x in np.flatnonzero(fork.action_mask(opp_role))})

    # ---- factorization checks: our labels depend on k only, theirs on j only -----------------
    for k in range(n_k):
        keys = set(maps1[k][0])
        for j in range(1, n_j):
            if set(maps1[k][j]) != keys:
                raise RuntimeError(
                    f"own-world {k} legal labels vary with the opponent world "
                    f"({sorted(keys ^ set(maps1[k][j]))}) — the [K, A] availability "
                    "factorization the solver relies on does not hold here")
    for j in range(n_j):
        keys = set(maps2[0][j])
        for k in range(1, n_k):
            if set(maps2[k][j]) != keys:
                raise RuntimeError(
                    f"opponent world {j} legal labels vary with our world "
                    f"({sorted(keys ^ set(maps2[k][j]))}) — the [J, B] availability "
                    "factorization the solver relies on does not hold here")

    labels1 = sorted(set().union(*(maps1[k][0] for k in range(n_k))))
    labels2 = sorted(set().union(*(maps2[0][j] for j in range(n_j))))
    col1 = {lab: i for i, lab in enumerate(labels1)}
    col2 = {lab: i for i, lab in enumerate(labels2)}
    n_a, n_b = len(labels1), len(labels2)

    avail1 = np.zeros((n_k, n_a), dtype=bool)
    local_action1 = np.full((n_k, n_a), -1, dtype=np.int64)
    for k in range(n_k):
        for lab, idx in maps1[k][0].items():
            avail1[k, col1[lab]] = True
            local_action1[k, col1[lab]] = idx
    avail2 = np.zeros((n_j, n_b), dtype=bool)
    local_action2 = np.full((n_j, n_b), -1, dtype=np.int64)
    for j in range(n_j):
        for lab, idx in maps2[0][j].items():
            avail2[j, col2[lab]] = True
            local_action2[j, col2[lab]] = idx

    # ---- simulate every available cell once (invariant 19.13) --------------------------------
    cell_outcomes: dict[tuple[int, int, int, int], list] = {}
    pruned_mass = np.zeros((n_k, n_j, n_a, n_b))
    failed = np.zeros((n_k, n_j, n_a, n_b), dtype=bool)
    sims_run = 0
    max_m = 1
    for k in range(n_k):
        for j in range(n_j):
            fork = forks[k][j]
            for a in np.flatnonzero(avail1[k]):
                for b in np.flatnonzero(avail2[j]):
                    sims_run += 1
                    opp_idx = None if one_sided else int(local_action2[j, b])
                    try:
                        outcomes, pruned = commit_pair_branches(
                            fork, role, int(local_action1[k, a]), opp_idx,
                            min_prob=min_prob, max_branches=max_branches,
                        )
                    except Exception:
                        failed[k, j, a, b] = True
                        continue
                    cell_outcomes[(k, j, int(a), int(b))] = outcomes
                    pruned_mass[k, j, a, b] = pruned
                    max_m = max(max_m, len(outcomes))

    # ---- pack branches into padded tensors + flat censored leaf list -------------------------
    branch_prob = np.zeros((n_k, n_j, n_a, n_b, max_m))
    terminal_value = np.full((n_k, n_j, n_a, n_b, max_m), np.nan)
    leaf_index = np.full((n_k, n_j, n_a, n_b, max_m), -1, dtype=np.int64)
    leaf_obs: list = []
    for (k, j, a, b), outcomes in cell_outcomes.items():
        for m, o in enumerate(outcomes):
            branch_prob[k, j, a, b, m] = o.probability
            if o.terminal is not None:
                terminal_value[k, j, a, b, m] = o.terminal
            else:
                leaf_index[k, j, a, b, m] = len(leaf_obs)
                leaf_obs.append(Observation.from_battle(
                    o.leaf.view(role), tracker=o.leaf.trackers[role]))

    return PBGrid(
        labels1=labels1, labels2=labels2, avail1=avail1, avail2=avail2,
        local_action1=local_action1, local_action2=local_action2,
        branch_prob=branch_prob, terminal_value=terminal_value, leaf_index=leaf_index,
        leaf_obs=leaf_obs, pruned_mass=pruned_mass, failed=failed, sims_run=sims_run,
    )
