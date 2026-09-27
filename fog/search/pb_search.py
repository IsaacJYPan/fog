"""One experimental two-sided public-belief decision (ARCHITECTURE.md S1–S3).

Assembles the landed pieces into the claim-bearing decision path: candidate worlds for both
sides -> joint belief rho -> simulated K x J branch grid (pb_grid) -> leaf values (pb_values,
the interim scalar head) -> two-sided regret solve (two_sided) -> **sample exactly
from the averaged strategy row of the true private state**. The API deliberately has no
temperature and no greedy mode (settled decisions 16.2/16.3): the sampled randomness IS the
mixed strategy being executed, and any reshaping would play a strategy the solver never
evaluated. A greedy diagnostic, if ever wanted, lives outside this function.

Two candidate-supply modes, because particle persistence is only half-landed (§7.4):

  * **Persistent mode** (exact pools): the caller passes `opp_tokens` — the same stable J
    candidate identities every turn — and optionally the battle-lifetime `JointBeliefTable`
    whose axes they index. rho comes from the table (the real §14 flow); without a table it is
    the independent outer product of the source weights.
  * **Adaptive sampled mode** (open spaces, interim): no `opp_tokens`; worlds are drawn fresh
    from the belief chain via `opp_source`, deduplicated with empirical-frequency weights, and
    GROWN (j doubling up to `j_max`, §13.1) until the executed action row stops moving by more
    than `j_tol` total variation, the sampler saturates, or the cap is hit. Re-sampling fresh
    worlds each decision is explicitly the §18 interim that particle rejuvenation replaces —
    a persistent table cannot be used in this mode (identities are not stable), so passing one
    raises.

The search NEVER mutates the belief table (the runner conditions on the subsequent public
observation, marginalizing hidden choices under the acting strategies). The result
carries a subsample of the public-observation posteriors the values were computed under
(`queried_beliefs`), plus the same subsampling under one uniformly drawn solver iteration's
CURRENT strategies (`sample_beliefs`) — together the value-training query log.
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
    pruned_mass_max: float           # worst per-cell dropped chance mass (invariant 19.16)
    queried_beliefs: list = field(default_factory=list)   # ("public_observation", key, rho_after)
    # Phase G1 (§9.3 mitigation 2): observation posteriors under the CURRENT strategy pair of one
    # uniformly drawn solver iteration — mid-solve belief queries the final averages never revisit
    # (early near-uniform beliefs especially). Same entry shape as queried_beliefs; the value
    # trainer mixes both sets (plus the root rho) as belief inputs. sample_iteration is the
    # 1-indexed iteration the snapshot came from (-1 on trivial decisions).
    sample_beliefs: list = field(default_factory=list)    # ("public_observation", key, rho_after)
    sample_iteration: int = -1
    # Phase G2 posterior-mass truncation (persistent mode): which FULL-axis candidates the solve
    # actually ran over. sigma/rho/tokens/true_index above all live on the KEPT axes; `own_kept` /
    # `opp_kept` map them back to the passed candidate sets (and the belief table's axes — the
    # persistent observation updates reject truncation of positive-mass candidates).
    # Dropped mass is the root-belief probability the solve did not model, logged per side
    # (§13.1/§13.2 applied to the persistent path). Without truncation kept = arange, dropped = 0.
    own_kept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    opp_kept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    dropped_own_mass: float = 0.0
    dropped_opp_mass: float = 0.0
    # The candidate identities rho / sigma / true_index index (proposal §10 shared particles). The
    # Phase-G value trainer needs the raw token grids to embed each team; k*/j* index into these.
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
    """Posterior-mass truncation for the persistent path (Phase G2, §13.1/§13.2 amendment A3).

    Zero-mass candidates are ALWAYS dropped — exact, not an approximation (a zero-probability
    type's strategy carries no weight in any update), and load-bearing for fork validity: a
    candidate contradicted by reveals cannot be forked coherently. Opponent candidates may then be
    capped by public margin mass. Own-side capping is refused when it would remove any live type:
    forcing the actual private type back into a publicly selected subset makes the opponent model
    depend on information the opponent does not possess. Returns ascending full-axis indices."""
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
    """Phase I.3 (§10.2 initialization-not-anchoring): both sides' solver strategies initialized
    from the CURRENT policy's masked distribution in each candidate world.

    Our side: the policy evaluated on OUR censored view with our private team swapped to
    candidate k (the own-world fork — private info is the only thing that varies with k).
    Opponent side: the policy on the OPPONENT's censored view with ITS private team swapped to
    candidate j. One batched forward over K+J observations; label mapping through the grid's
    local_action tables. Linear iteration weighting in the solver keeps this a warm start whose
    influence vanishes with iterations (pinned by the washout test) — never a payoff bonus.
    Returns (init1 [K, A], init2 [J, B] or None at a one-sided decision)."""
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
    """Map the true state's averaged label row onto real-battle action indices, then defensively
    renormalize over the REAL legality mask (the true-world fork must agree with the real battle;
    trusting that blindly is how indexing bugs stay silent — the frozen baseline's convention)."""
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
    """One real decision under the two-sided solver. See the module docstring for the two
    candidate-supply modes. Never mutates `battle` (beyond fork seeding) or `table`.

    `top_j` (Phase G2, persistent mode only): opponent-axis truncation by public rho margin mass.
    `top_k` is accepted for configuration compatibility but raises if it would remove a live own
    type: forcing the actual private type back into support leaks truth into the opponent strategy.
    Dropped opponent mass is reported and full-axis table entries persist.
    Zero-mass candidates are always dropped in persistent mode regardless (exact — and required
    before forking, see _truncate_axes).

    `policy_init` (Phase I.3, §10.2): initialize both sides' solver strategies from the current
    policy's per-world masked distributions (see _policy_inits) — a warm start whose influence
    washes out under linear averaging. Default off; turning it on is a measured compute nicety
    (iterations-to-stability), never a correctness change.

    `value_net` (Phase G5 switch): a trained `PBValueNet` — leaf values come from
    `make_pb_value_grid_values` (per-pair tables, refreshed every
    `value_recompute_every` calls — the G4-drift-chosen N). Default None keeps the §18 interim
    posterior-feature head (`make_grid_values`) as the baseline arm; the swap is claim-bearing
    only after the G4 gates pass."""
    if int(search_depth) not in (1, 2):
        raise ValueError("search_depth currently supports 1 or 2; deeper traversal requires the "
                         "outcome-sampling public-belief search phase")
    rng = random.Random(seed) if rng is None else rng
    mask = battle.action_mask(role)
    real_legal = np.flatnonzero(mask)
    # A single action has no choice entropy, but its public consequences can still carry chance
    # evidence. Persistent filtering must model that transition. A one-sided forced replacement or
    # pivot with 2+ live choices is NOT trivial: build_pb_grid collapses the absent opponent to a
    # single synthetic label and the B=1 solve yields a real, type-conditioned replacement sigma1
    # (F2, proposal §10.5) — it flows through the normal path below.
    if len(real_legal) == 0:
        return _trivial(13)
    if len(real_legal) == 1 and table is None:
        return _trivial(int(real_legal[0]))
    if table is not None and opp_tokens is None:
        raise ValueError("a persistent belief table indexes stable candidates: pass the "
                         "opp_tokens it was built over (sampled fresh worlds have no stable "
                         "identity for it — proposal §7.4)")

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
                             "budget feature; adaptive mode already sizes J by the §13.1 "
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

    # §9.3 seed data: a subsample of the public-observation posteriors the leaf values saw
    queried: list[tuple[str, str, np.ndarray]] = []
    sampled: list[tuple[str, str, np.ndarray]] = []
    observation_model = PublicObservationLikelihood(grid)
    leaves = list(range(len(grid.leaf_obs)))
    if leaves and n_queried > 0:
        post = observation_model.posteriors(rho, sol.sigma1, sol.sigma2)
        reachable = [leaf for leaf in leaves if post[leaf].sum() > 0]
        for leaf in rng.sample(reachable, min(n_queried, len(reachable))):
            queried.append(("public_observation", grid.leaf_public_keys[leaf], post[leaf].copy()))
        # G1: the same subsampling under the random-iteration snapshot pair — the mid-solve
        # queries (§9.3 mitigation 2). An independent leaf draw widens observation coverage.
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
