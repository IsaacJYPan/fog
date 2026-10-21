"""Build and solve a two-sided public-belief search grid.

Candidate teams and a joint belief define the simulated action grid. Leaf values
feed the regret solver, then the action is sampled from the true private team's
averaged strategy row. Changing its temperature would change the solved strategy.

With ``opp_tokens``, candidate identities stay fixed and an optional
``JointBeliefTable`` supplies the prior. Without a table, source weights form an
independent product prior. Without ``opp_tokens``, candidates are sampled afresh
and deduplicated. Their count doubles up to ``j_max`` until the action row changes
by at most ``j_tol`` total variation, the sampler saturates, or the cap is reached.
Fresh identities cannot be combined with a persistent belief table.

The runner updates the table after observing the turn; search only reads it.
Results include posterior queries under the final strategies and one uniformly
sampled solver iteration for value training.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fog.search.pb_grid import build_pb_grid
from fog.search.pb_values import make_grid_values
from fog.search.public_observation import PublicObservationLikelihood
from fog.search.two_sided import two_sided_regret_solve

N_ACTIONS = 14


@dataclass
class PBSearchResult:
    action: int                      # real-battle action index (execute this)
    action_dist: np.ndarray          # [14] the true state's averaged row on real action indices
    labels1: list[str]               # our public-label space (columns of sigma1)
    labels2: list[str]               # the opponent's (columns of sigma2)
    sigma1: np.ndarray               # [K, A] averaged solved strategy (belief updates use this)
    sigma2: np.ndarray               # [J, B] averaged solved opponent strategy
    avail1: np.ndarray               # [K, A] bool — needed to floor likelihoods correctly: a
    avail2: np.ndarray               # [J, B]        zero from "unavailable" must stay a zero
    rho: np.ndarray                  # [K, J] the root joint belief the solve used
    true_index: int                  # k of the real private state (its row was sampled)
    iterations: int
    converged: bool
    j_used: int                      # opponent candidates in the final solve
    sims_run: int                    # committed (k, j, a, b) pairs across all solves
    failed_cells: int
    pruned_mass_max: float           # worst per-cell dropped chance mass
    queried_beliefs: list = field(default_factory=list)   # ("public_observation", key, rho_after)
    # Posteriors from one uniformly sampled solver iteration, for value training.
    # sample_iteration is 1-indexed, or -1 on trivial decisions.
    sample_beliefs: list = field(default_factory=list)    # ("public_observation", key, rho_after)
    sample_iteration: int = -1
    # sigma/rho/tokens/true_index use the retained axes. These indices map them
    # back to the full candidate sets; dropped mass is omitted root probability.
    own_kept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    opp_kept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    dropped_own_mass: float = 0.0
    dropped_opp_mass: float = 0.0
    # Candidate teams indexed by rho, sigma, and true_index; [K, 6, 15] / [J, 6, 15].
    own_tokens: np.ndarray = field(default_factory=lambda: np.zeros((0, 6, 15), dtype=np.int64))
    opp_tokens: np.ndarray = field(default_factory=lambda: np.zeros((0, 6, 15), dtype=np.int64))
    search_depth: int = 1
    depth2_expanded: int = 0
    depth2_reach_mass: float = 0.0
    depth2_child_sims: int = 0
    depth2_failures: list[str] = field(default_factory=list)
    observation_model: Any | None = None


def sample_opponent_worlds(agent: Any, battle: Any, role: str, j: int, *, seed: int,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Default `opp_source`: j ancestral samples from the belief chain on `role`'s censored
    observation, deduplicated, weighted by empirical frequency (the documented SampledOwnStates
    recipe applied to the opponent direction). Returns (tokens [J', 6, 15], log_weights [J'])."""
    import torch

    from fog.encoding.observation import Observation
    from fog.env.engine_runner import _device, _stack_obs

    agent.eval()
    device = _device(agent)
    obs = Observation.from_battle(battle.view(role), tracker=battle.trackers[role])
    batch = _stack_obs([obs], device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        belief_context = agent.belief_context(contexts)
        worlds = agent.belief.sample_team(
            belief_context, batch["opp_revealed"], k=max(1, int(j)), generator=gen,
        )[0].cpu().numpy().astype(np.int64)
    index: dict[bytes, int] = {}
    tokens: list[np.ndarray] = []
    counts: list[int] = []
    for w in worlds:
        key = w.tobytes()
        if key in index:
            counts[index[key]] += 1
        else:
            index[key] = len(tokens)
            tokens.append(w)
            counts.append(1)
    weights = np.asarray(counts, dtype=np.float64)
    return np.stack(tokens), np.log(weights / weights.sum())


def _truncate_axes(rho: np.ndarray, true_k: int, top_k: int | None, top_j: int | None,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return retained full-axis indices for persistent search.

    Drop zero-mass candidates before forking: reveals may have made their worlds
    inconsistent. Cap opponent candidates by public marginal mass. Own-side caps
    that remove live types raise, since reinserting the true private team would
    leak it into the calculated opponent strategy.
    """
    row_mass = rho.sum(axis=1)
    col_mass = rho.sum(axis=0)
    if row_mass[true_k] <= 0.0:
        raise RuntimeError("the true own candidate has zero belief mass — upstream evidence bug "
                           "(the true state must never be eliminated by its own reveals)")
    own_alive = np.flatnonzero(row_mass > 0.0)
    opp_alive = np.flatnonzero(col_mass > 0.0)
    if top_k is not None and len(own_alive) > int(top_k):
        raise ValueError(
            "own-side top_k truncation would remove live public types. It is disabled because "
            "forcing the actual private team back into the solve leaks that truth into the "
            "calculated opponent strategy. Use top_k=None until a truth-independent private-row "
            "query is implemented."
        )
    if top_j is not None and len(opp_alive) > int(top_j):
        opp_alive = np.sort(opp_alive[np.argsort(col_mass[opp_alive])[::-1]][: int(top_j)])
    return own_alive.astype(np.int64), opp_alive.astype(np.int64)


def _policy_inits(agent, battle, role: str, grid, own_toks: np.ndarray, opp_toks: np.ndarray,
                  rng: random.Random) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Initialize strategies from the policy's masked distributions in candidate worlds.

    Evaluate each side's censored view with only its own private team swapped,
    in one K+J batch, then map action indices through the grid's label tables.
    Linear iteration weighting makes the initialization's influence decay.
    Returns init1 [K, A] and init2 [J, B], or None for a one-sided decision.
    """
    import torch

    from fog.encoding.observation import Observation
    from fog.env.engine_runner import _device, _stack_obs

    opp_role = "p2" if role == "p1" else "p1"
    one_sided = bool((grid.local_action2 < 0).all())
    obs = [Observation.from_battle(
        battle.fork_for_own_world(role, own_toks[k], seed=rng.randrange(2**31)).view(role),
        tracker=battle.trackers[role]) for k in range(len(own_toks))]
    if not one_sided:
        obs += [Observation.from_battle(
            battle.fork_for_own_world(opp_role, opp_toks[j],
                                      seed=rng.randrange(2**31)).view(opp_role),
            tracker=battle.trackers[opp_role]) for j in range(len(opp_toks))]
    agent.eval()
    with torch.no_grad():
        probs = agent(_stack_obs(obs, _device(agent))).log_probs.exp().cpu().numpy()

    def _map(p: np.ndarray, local: np.ndarray, avail: np.ndarray) -> np.ndarray:
        init = np.zeros(avail.shape, dtype=np.float64)
        for row in range(avail.shape[0]):
            cols = np.flatnonzero(avail[row])
            init[row, cols] = p[row, local[row, cols]]
            z = init[row].sum()
            init[row] = init[row] / z if z > 0 else avail[row] / avail[row].sum()
        return init

    init1 = _map(probs[: len(own_toks)], grid.local_action1, grid.avail1)
    init2 = (None if one_sided
             else _map(probs[len(own_toks):], grid.local_action2, grid.avail2))
    return init1, init2


def _real_action_dist(row: np.ndarray, local_actions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map the true team's label row to action indices and renormalize over legal actions."""
    dist = np.zeros(N_ACTIONS, dtype=np.float64)
    for col in np.flatnonzero(local_actions >= 0):
        dist[local_actions[col]] += row[col]
    dist = np.where(mask, dist, 0.0)
    if dist.sum() <= 0:
        dist = mask.astype(np.float64)
    return dist / dist.sum()


def _trivial(action: int, true_index: int = 0) -> PBSearchResult:
    dist = np.zeros(N_ACTIONS)
    dist[action] = 1.0
    return PBSearchResult(
        action=action, action_dist=dist, labels1=[], labels2=[],
        sigma1=np.zeros((0, 0)), sigma2=np.zeros((0, 0)),
        avail1=np.zeros((0, 0), dtype=bool), avail2=np.zeros((0, 0), dtype=bool),
        rho=np.zeros((0, 0)),
        true_index=true_index, iterations=0, converged=True, j_used=0, sims_run=0,
        failed_cells=0, pruned_mass_max=0.0,
    )


def pb_search_action(
    agent: Any,
    battle: Any,
    role: str = "p1",
    *,
    own_source: Any,
    opp_tokens: np.ndarray | None = None,
    table: Any | None = None,
    opp_source: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    rng: random.Random | None = None,
    seed: int = 0,
    iters: int = 256,
    floor: float = 0.02,
    min_prob: float = 0.0,
    max_branches: int | None = None,
    j_start: int = 8,
    j_max: int = 40,
    j_tol: float = 0.05,
    n_queried: int = 8,
    top_k: int | None = None,
    top_j: int | None = None,
    policy_init: bool = False,
    value_net: Any | None = None,
    value_recompute_every: int = 8,
    search_depth: int = 1,
    depth2_max_nodes: int = 2,
    depth2_min_reach: float = 0.01,
    depth2_child_iters: int = 64,
) -> PBSearchResult:
    """Solve one decision without mutating the belief table or live battle state.

    In persistent mode, `top_j` caps the opponent axis by marginal belief mass;
    omitted mass is reported and the full table remains intact. `top_k` raises
    if it would remove live own types, since preserving the true team alone
    would leak private information. Zero-mass candidates are dropped before forks.

    `policy_init` warm-starts both strategies from per-world policy predictions.
    `value_net` supplies per-pair PBValueNet values, refreshed every
    `value_recompute_every` calls. With None, values use the scalar head's
    posterior features. Choose the refresh interval using `fog.eval.pb_value_drift`.
    """
    if int(search_depth) not in (1, 2):
        raise ValueError("search_depth currently supports 1 or 2; deeper traversal requires the "
                         "outcome-sampling public-belief search phase")
    rng = random.Random(seed) if rng is None else rng
    mask = battle.action_mask(role)
    real_legal = np.flatnonzero(mask)
    # Even a forced action can reveal chance evidence needed by a persistent table.
    # At one-sided decisions, build_pb_grid gives the absent opponent a synthetic
    # label, leaving our replacement strategy conditioned on the candidate team.
    if len(real_legal) == 0:
        return _trivial(13)
    if len(real_legal) == 1 and table is None:
        return _trivial(int(real_legal[0]))
    if table is not None and opp_tokens is None:
        raise ValueError("a persistent belief table indexes stable candidates: pass the "
                         "opp_tokens it was built over (sampled fresh worlds have no stable "
                         "identity for it)")

    own = own_source.states(battle, role)
    kt = int(own.true_index)
    own_w = np.asarray(own.weights, dtype=np.float64)
    own_w = own_w / own_w.sum()

    # root value = the failed-cell fallback (the baseline's neutral convention)
    import torch

    from fog.encoding.observation import Observation
    from fog.env.engine_runner import _device, _stack_obs

    agent.eval()
    root_obs = Observation.from_battle(battle.view(role), tracker=battle.trackers[role])
    with torch.no_grad():
        root_value = float(agent(_stack_obs([root_obs], _device(agent))).value[0])

    def _solve(own_toks: np.ndarray, opp_toks: np.ndarray, rho: np.ndarray):
        grid = build_pb_grid(battle, role, own_toks, opp_toks,
                             seed=rng.randrange(2**31), min_prob=min_prob,
                             max_branches=max_branches)
        init1 = init2 = None
        if policy_init:
            init1, init2 = _policy_inits(agent, battle, role, grid, own_toks, opp_toks, rng)
        def make_values(leaf_pair_values=None):
            if value_net is not None:
                from fog.search.pb_value_fn import make_pb_value_grid_values

                return make_pb_value_grid_values(
                    agent, value_net, grid, rho, own_toks, opp_toks,
                    recompute_every=value_recompute_every, fallback_value=root_value,
                    leaf_pair_values=leaf_pair_values,
                )
            return make_grid_values(
                agent, grid, rho, fallback_value=root_value,
                leaf_pair_values=leaf_pair_values,
            )

        values_fn, vstats = make_values()
        sol = two_sided_regret_solve(values_fn, rho, grid.avail1, grid.avail2,
                                     init1=init1, init2=init2,
                                     iters=iters, floor=floor,
                                     sample_seed=rng.randrange(2**31))
        depth_stats = None
        if int(search_depth) == 2:
            from fog.search.selective_depth import build_depth2_overrides

            overrides, depth_stats = build_depth2_overrides(
                agent, grid, rho, sol.sigma1, sol.sigma2, role, own_toks, opp_toks,
                rng=rng, value_net=value_net,
                value_recompute_every=value_recompute_every,
                child_iters=depth2_child_iters, floor=floor,
                min_prob=min_prob, max_branches=max_branches,
                max_nodes=depth2_max_nodes, min_reach=depth2_min_reach,
                fallback_value=root_value,
            )
            if depth_stats.expanded:
                values_fn, vstats = make_values(overrides)
                sol = two_sided_regret_solve(
                    values_fn, rho, grid.avail1, grid.avail2,
                    init1=sol.sigma1, init2=sol.sigma2,
                    iters=iters, floor=floor,
                    sample_seed=rng.randrange(2**31),
                )
        return grid, vstats, sol, depth_stats

    sims_total = 0
    if opp_tokens is not None:                                  # ---- persistent mode ----
        opp_tokens = np.asarray(opp_tokens, dtype=np.int64)
        if table is not None:
            rho_full = table.normalized()
            if rho_full.shape != (len(own.tokens), len(opp_tokens)):
                raise ValueError(f"table shape {rho_full.shape} does not index the candidate "
                                 f"sets ({len(own.tokens)}, {len(opp_tokens)})")
        else:
            rho_full = np.outer(own_w, np.full(len(opp_tokens), 1.0 / len(opp_tokens)))
        own_kept, opp_kept = _truncate_axes(rho_full, kt, top_k, top_j)
        dropped_own = float(1.0 - rho_full[own_kept, :].sum())
        dropped_opp = float(1.0 - rho_full[:, opp_kept].sum())
        rho = rho_full[np.ix_(own_kept, opp_kept)]
        rho = rho / rho.sum()
        own_toks_used = own.tokens[own_kept]
        opp_toks_used = opp_tokens[opp_kept]
        kt = int(np.flatnonzero(own_kept == own.true_index)[0])
        grid, vstats, sol, depth_stats = _solve(own_toks_used, opp_toks_used, rho)
        sims_total += grid.sims_run + (
            depth_stats.child_sims if depth_stats is not None else 0
        )
    else:                                                       # ---- adaptive sampled mode ----
        if top_k is not None or top_j is not None:
            raise ValueError("posterior-mass truncation (top_k/top_j) is a persistent-mode "
                             "budget feature; adaptive mode already sizes J by the "
                             "stability rule")
        if opp_source is None:
            def _default_opp_source(b, r, j, *, seed):
                return sample_opponent_worlds(agent, b, r, j, seed=seed)
            opp_source = _default_opp_source
        j_req = min(j_start, j_max)
        prev_row, last_count = None, -1
        while True:
            toks, logw = opp_source(battle, role, j_req, seed=rng.randrange(2**31))
            rho = np.outer(own_w, np.exp(logw))
            rho = rho / rho.sum()
            grid, vstats, sol, depth_stats = _solve(own.tokens, toks, rho)
            sims_total += grid.sims_run + (
                depth_stats.child_sims if depth_stats is not None else 0
            )
            row = _real_action_dist(sol.sigma1[kt], grid.local_action1[kt], mask)
            stable = (prev_row is not None
                      and 0.5 * np.abs(row - prev_row).sum() <= j_tol)
            saturated = len(toks) == last_count                 # dedup stopped finding new worlds
            if stable or saturated or len(toks) >= j_max or j_req >= j_max:
                break
            prev_row, last_count = row, len(toks)
            j_req = min(j_req * 2, j_max)
        own_kept = np.arange(len(own.tokens), dtype=np.int64)
        opp_kept = np.arange(len(toks), dtype=np.int64)
        dropped_own = dropped_opp = 0.0
        own_toks_used, opp_toks_used = own.tokens, toks

    action_dist = _real_action_dist(sol.sigma1[kt], grid.local_action1[kt], mask)
    action = int(rng.choices(range(N_ACTIONS), weights=action_dist)[0])

    # Subsample public-observation posteriors for value training.
    queried: list[tuple[str, str, np.ndarray]] = []
    sampled: list[tuple[str, str, np.ndarray]] = []
    observation_model = PublicObservationLikelihood(grid)
    leaves = list(range(len(grid.leaf_obs)))
    if leaves and n_queried > 0:
        post = observation_model.posteriors(rho, sol.sigma1, sol.sigma2)
        reachable = [leaf for leaf in leaves if post[leaf].sum() > 0]
        for leaf in rng.sample(reachable, min(n_queried, len(reachable))):
            queried.append(("public_observation", grid.leaf_public_keys[leaf], post[leaf].copy()))
        # Include mid-solve beliefs as well as final-strategy beliefs. Draw leaves
        # independently to cover more public observations.
        if sol.sample_sigma1 is not None:
            post_s = observation_model.posteriors(rho, sol.sample_sigma1, sol.sample_sigma2)
            reachable_s = [leaf for leaf in leaves if post_s[leaf].sum() > 0]
            for leaf in rng.sample(reachable_s, min(n_queried, len(reachable_s))):
                sampled.append(("public_observation", grid.leaf_public_keys[leaf], post_s[leaf].copy()))

    return PBSearchResult(
        action=action, action_dist=action_dist,
        labels1=grid.labels1, labels2=grid.labels2,
        sigma1=sol.sigma1, sigma2=sol.sigma2,
        avail1=grid.avail1.copy(), avail2=grid.avail2.copy(), rho=rho, true_index=kt,
        iterations=sol.iterations, converged=sol.converged,
        j_used=grid.avail2.shape[0], sims_run=sims_total,
        failed_cells=vstats.failed_cells, pruned_mass_max=float(grid.pruned_mass.max()),
        queried_beliefs=queried,
        observation_model=observation_model,
        sample_beliefs=sampled, sample_iteration=sol.sample_iteration,
        # candidate identities behind rho's axes: the KEPT sets the solve actually ran over
        # (own_kept/opp_kept map them back to the caller's full axes); true_index indexes the
        # kept own set.
        own_tokens=np.asarray(own_toks_used, dtype=np.int64),
        opp_tokens=np.asarray(opp_toks_used, dtype=np.int64),
        own_kept=own_kept, opp_kept=opp_kept,
        dropped_own_mass=dropped_own, dropped_opp_mass=dropped_opp,
        search_depth=int(search_depth),
        depth2_expanded=(depth_stats.expanded if depth_stats is not None else 0),
        depth2_reach_mass=(
            depth_stats.expanded_reach_mass if depth_stats is not None else 0.0
        ),
        depth2_child_sims=(depth_stats.child_sims if depth_stats is not None else 0),
        depth2_failures=(depth_stats.failures if depth_stats is not None else []),
    )
