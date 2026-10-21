"""Cached chance-branch grid for two-sided public-belief search.

The caller supplies own_tokens[K, 6, 15] and opp_tokens[J, 6, 15]. Each candidate
pair faces the same public-action axes. Availability for our actions depends only
on k, and opponent availability only on j; inconsistent engine results raise.

`commit_pair_branches` simulates each cell once to the next decision, recording
weighted outcomes and pruned probability mass. Regret iterations reuse the grid.
Leaf observations omit private team sheets; candidate inputs carry that information
separately. Each leaf has a [K, J] mask excluding pairs contradicted by new reveals.

When only our side is pending, the opponent axis has one `<observe>` action and
commits with opp_action_idx=None. The resulting replacement strategy still carries
behavioral evidence. Calling this builder when only the opponent is pending raises.

Failed simulations remain marked in `failed` with no branches. The value layer
uses the root value for those cells and reports their count."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from fog.search.pb_leaves import commit_pair_branches

# At one-sided decisions, B = 1 and the belief update has no opponent-action likelihood.
OBSERVE_LABEL = "<observe>"


def public_label(fork, role: str, action_idx: int) -> str:
    """Internal chosen-action identity: a kind-prefixed engine action string, lowercased.
    This is not an observation: prevented moves may remain hidden. Same convention as the helper
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
    `leaf_obs[leaf_index[...]]` is its public-only leaf observation."""
    labels1: list[str]           # our public-action label space (union over k)
    labels2: list[str]           # the opponent's (union over j)
    avail1: np.ndarray           # [K, A] bool — world k can play label a
    avail2: np.ndarray           # [J, B] bool
    local_action1: np.ndarray    # [K, A] int — label column -> real action index in world k; -1
    local_action2: np.ndarray    # [J, B] int
    branch_prob: np.ndarray      # [K, J, A, B, M] float, 0 where no branch
    terminal_value: np.ndarray   # [K, J, A, B, M] float, NaN where not terminal
    leaf_index: np.ndarray       # [K, J, A, B, M] int into leaf_obs, -1 where terminal/absent
    leaf_obs: list               # flat public-only Observations for batched value forwards
    leaf_battles: list           # matching live forks; used only by selective depth > 1
    leaf_evidence: np.ndarray    # [L, K, J] bool — candidate pairs compatible with leaf reveals
    pruned_mass: np.ndarray      # [K, J, A, B] float — dropped branch probability per cell
    failed: np.ndarray           # [K, J, A, B] bool — simulation raised; value layer substitutes
    sims_run: int                # committed (k, j, a, b) pairs (budget accounting — report it)
    # What the failed cells actually raised: {"ExcType: msg": count}. A flood of failures signals
    # a reconstruction bug (module docstring) — this makes the flood diagnosable from the grid
    # itself instead of requiring a re-run under a debugger.
    fail_reasons: dict[str, int] = field(default_factory=dict)
    leaf_public_keys: list[str] | None = None


def build_pb_grid(
    battle,
    role: str,
    own_tokens: np.ndarray,
    opp_tokens: np.ndarray,
    *,
    seed: int = 0,
    min_prob: float = 0.0,
    max_branches: int | None = None,
) -> PBGrid:
    """Simulate the full K x J x A x B branch table for one simultaneous decision on `battle`.

    `own_tokens` are the K candidate versions of OUR team the opponent could believe in (the
    own-state sources' token grids); `opp_tokens` are the J shared opponent candidates (sampled
    once from the belief chain, deduplicated and weighted by the CALLER — this builder treats
    both sets as given identities so the persistent belief table keys stay stable).
    Never mutates `battle`."""
    from fog.constants.schema import NO_ITEM
    from fog.encoding.observation import Observation
    from fog.search.evidence import keep_mask_from_reveals
    from fog.search.public_observation import public_observation_key

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
    no_item = int(battle.tables.item_index(NO_ITEM))

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

    # Simulate each available cell once; solver iterations reuse the outcomes.
    cell_outcomes: dict[tuple[int, int, int, int], list] = {}
    pruned_mass = np.zeros((n_k, n_j, n_a, n_b))
    failed = np.zeros((n_k, n_j, n_a, n_b), dtype=bool)
    fail_reasons: dict[str, int] = {}
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
                    except Exception as e:  # noqa: BLE001 - failure is retained in PBGrid
                        failed[k, j, a, b] = True
                        reason = f"{type(e).__name__}: {e}"[:200]
                        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                        continue
                    cell_outcomes[(k, j, int(a), int(b))] = outcomes
                    pruned_mass[k, j, a, b] = pruned
                    max_m = max(max_m, len(outcomes))

    # ---- pack branches into padded tensors + public leaves and evidence masks ----------------
    branch_prob = np.zeros((n_k, n_j, n_a, n_b, max_m))
    terminal_value = np.full((n_k, n_j, n_a, n_b, max_m), np.nan)
    leaf_index = np.full((n_k, n_j, n_a, n_b, max_m), -1, dtype=np.int64)
    leaf_obs: list = []
    leaf_battles: list = []
    leaf_evidence: list[np.ndarray] = []
    leaf_public_keys: list[str] = []
    for (k, j, a, b), outcomes in cell_outcomes.items():
        for m, o in enumerate(outcomes):
            branch_prob[k, j, a, b, m] = o.probability
            if o.terminal is not None:
                terminal_value[k, j, a, b, m] = o.terminal
            else:
                leaf_index[k, j, a, b, m] = len(leaf_obs)
                public_obs = Observation.public_from_engine_battle(o.leaf, role)
                reverse_obs = Observation.from_battle(
                    o.leaf.view(opp_role), tracker=o.leaf.trackers[opp_role]
                )
                keep_own = keep_mask_from_reveals(
                    reverse_obs.opp_revealed["tokens"], own_tokens,
                    no_item_token=no_item,
                )
                keep_opp = keep_mask_from_reveals(
                    public_obs.opp_revealed["tokens"], opp_tokens,
                    no_item_token=no_item,
                )
                if not keep_own[k] or not keep_opp[j]:
                    raise RuntimeError(
                        "a simulated leaf contradicts the candidate pair that produced it; "
                        "the reveal census or item-movement carve-out is incorrect "
                        f"(k={k}, j={j}, a={a} {labels1[a]!r}, b={b} {labels2[b]!r}, "
                        f"branch={m}, keep_own={bool(keep_own[k])}, "
                        f"keep_opp={bool(keep_opp[j])})"
                    )
                leaf_obs.append(public_obs)
                leaf_battles.append(o.leaf)
                leaf_public_keys.append(public_observation_key(o.leaf))
                leaf_evidence.append(np.outer(keep_own, keep_opp))

    return PBGrid(
        labels1=labels1, labels2=labels2, avail1=avail1, avail2=avail2,
        local_action1=local_action1, local_action2=local_action2,
        branch_prob=branch_prob, terminal_value=terminal_value, leaf_index=leaf_index,
        leaf_obs=leaf_obs, leaf_battles=leaf_battles,
        leaf_evidence=(np.stack(leaf_evidence).astype(bool, copy=False)
                       if leaf_evidence else np.ones((0, n_k, n_j), dtype=bool)),
        pruned_mass=pruned_mass, failed=failed, sims_run=sims_run,
        fail_reasons=fail_reasons,
        leaf_public_keys=leaf_public_keys,
    )
