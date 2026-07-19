"""pb_search_action — one two-sided public-belief decision (PROPOSAL_IMPL_PLAN Phase F;
PROJECT_DESIGN_PROPOSAL.md §14 end-to-end procedure, §10.3, §13.1).

Assembles the landed pieces into the claim-bearing decision path: candidate worlds for both
sides -> joint belief rho -> simulated K x J branch grid (pb_grid) -> leaf values (pb_values,
the §18 interim head until Phase G) -> two-sided regret solve (two_sided) -> **sample exactly
from the averaged strategy row of the true private state**. The API deliberately has no
temperature and no greedy mode (settled decisions 16.2/16.3): the sampled randomness IS the
mixed equilibrium being executed, and any reshaping would play a strategy the solver never
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

The search NEVER mutates the belief table (§14 step 11 is the runner's job, after both real
actions are public, using this result's averaged strategies — invariant 19.6). The result
carries a subsample of the action-pair posteriors the values were computed under
(`queried_beliefs`) — the seed of the §9.3 value-training query log.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from fog.search.pb_grid import build_pb_grid
from fog.search.pb_values import make_grid_values
from fog.search.two_sided import pair_posteriors, two_sided_regret_solve

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
    queried_beliefs: list = field(default_factory=list)   # (label1, label2, rho_after [K, J])
    # The candidate identities rho / sigma / true_index index (proposal §10 shared particles). The
    # Phase-G value trainer needs the raw token grids to embed each team; k*/j* index into these.
    own_tokens: np.ndarray = field(default_factory=lambda: np.zeros((0, 6, 9), dtype=np.int64))
    opp_tokens: np.ndarray = field(default_factory=lambda: np.zeros((0, 6, 9), dtype=np.int64))


def sample_opponent_worlds(agent: Any, battle: Any, role: str, j: int, *, seed: int,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Default `opp_source`: j ancestral samples from the belief chain on `role`'s censored
    observation, deduplicated, weighted by empirical frequency (the documented SampledOwnStates
    recipe applied to the opponent direction). Returns (tokens [J', 6, 9], log_weights [J'])."""
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
        _, c_dyn = agent._backbone_pass(batch)
        worlds = agent.belief.sample_team(
            c_dyn, batch["opp_revealed"], k=max(1, int(j)), generator=gen,
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
    min_prob: float = 0.005,
    max_branches: int | None = 12,
    j_start: int = 8,
    j_max: int = 40,
    j_tol: float = 0.05,
    n_queried: int = 8,
) -> PBSearchResult:
    """One real decision under the two-sided solver. See the module docstring for the two
    candidate-supply modes. Never mutates `battle` (beyond fork seeding) or `table`."""
    rng = random.Random(seed) if rng is None else rng
    mask = battle.action_mask(role)
    real_legal = np.flatnonzero(mask)
    # Only 0/1 legal actions are trivial (no behavioral content). A one-sided forced replacement or
    # pivot with 2+ live choices is NOT trivial: build_pb_grid collapses the absent opponent to a
    # single synthetic label and the B=1 solve yields a real, type-conditioned replacement sigma1
    # (F2, proposal §10.5) — it flows through the normal path below.
    if len(real_legal) == 0:
        return _trivial(13)
    if len(real_legal) == 1:
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

    def _solve(opp_toks: np.ndarray, rho: np.ndarray):
        grid = build_pb_grid(battle, role, own.tokens, opp_toks,
                             seed=rng.randrange(2**31), min_prob=min_prob,
                             max_branches=max_branches)
        values_fn, vstats = make_grid_values(agent, grid, rho, fallback_value=root_value)
        sol = two_sided_regret_solve(values_fn, rho, grid.avail1, grid.avail2,
                                     iters=iters, floor=floor)
        return grid, vstats, sol

    sims_total = 0
    if opp_tokens is not None:                                  # ---- persistent mode ----
        opp_tokens = np.asarray(opp_tokens, dtype=np.int64)
        if table is not None:
            rho = table.normalized()
            if rho.shape != (len(own.tokens), len(opp_tokens)):
                raise ValueError(f"table shape {rho.shape} does not index the candidate sets "
                                 f"({len(own.tokens)}, {len(opp_tokens)})")
        else:
            rho = np.outer(own_w, np.full(len(opp_tokens), 1.0 / len(opp_tokens)))
        grid, vstats, sol = _solve(opp_tokens, rho)
        sims_total += grid.sims_run
    else:                                                       # ---- adaptive sampled mode ----
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
            grid, vstats, sol = _solve(toks, rho)
            sims_total += grid.sims_run
            row = _real_action_dist(sol.sigma1[kt], grid.local_action1[kt], mask)
            stable = (prev_row is not None
                      and 0.5 * np.abs(row - prev_row).sum() <= j_tol)
            saturated = len(toks) == last_count                 # dedup stopped finding new worlds
            if stable or saturated or len(toks) >= j_max or j_req >= j_max:
                break
            prev_row, last_count = row, len(toks)
            j_req = min(j_req * 2, j_max)

    action_dist = _real_action_dist(sol.sigma1[kt], grid.local_action1[kt], mask)
    action = int(rng.choices(range(N_ACTIONS), weights=action_dist)[0])

    # §9.3 seed data: a subsample of the action-pair posteriors the leaf values saw
    queried: list[tuple[str, str, np.ndarray]] = []
    pairs = [(a, b) for a in range(len(grid.labels1)) for b in range(len(grid.labels2))
             if grid.avail1[:, a].any() and grid.avail2[:, b].any()]
    if pairs and n_queried > 0:
        post = pair_posteriors(rho, sol.sigma1, sol.sigma2)
        for a, b in rng.sample(pairs, min(n_queried, len(pairs))):
            queried.append((grid.labels1[a], grid.labels2[b], post[a, b].copy()))

    return PBSearchResult(
        action=action, action_dist=action_dist,
        labels1=grid.labels1, labels2=grid.labels2,
        sigma1=sol.sigma1, sigma2=sol.sigma2,
        avail1=grid.avail1.copy(), avail2=grid.avail2.copy(), rho=rho, true_index=kt,
        iterations=sol.iterations, converged=sol.converged,
        j_used=grid.avail2.shape[0], sims_run=sims_total,
        failed_cells=vstats.failed_cells, pruned_mass_max=float(grid.pruned_mass.max()),
        queried_beliefs=queried,
        # candidate identities behind rho's axes: persistent mode used `opp_tokens`, adaptive mode
        # the last-sampled `toks`. own.tokens is the K own candidates the true_index kt indexes.
        own_tokens=np.asarray(own.tokens, dtype=np.int64),
        opp_tokens=np.asarray(opp_tokens if opp_tokens is not None else toks, dtype=np.int64),
    )
