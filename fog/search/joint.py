"""Legacy joint sampled counterfactual search. Inference-time only — NEVER in PPO.

Where determinized search (search/determinized.py) samples worlds for MY uncertainty about the
OPPONENT, this module reasons over the opposite direction: K candidate private states the OPPONENT
considers possible for ME (fog/search/own_states.py). The K per-state policies are optimized
TOGETHER, coupled through the opponent's belief update: the value of an action in state k is
computed under the posterior over states that playing it would induce — the defining requirement
of joint counterfactual search. Executed action: sampled from the TRUE state's averaged policy.

Three design decisions that carry the correctness:

  1. **Coupling is over PUBLIC actions, not action-slot indices.** In different candidate worlds
     the same move slot holds different moves; what the opponent observes is the move id (or the
     switch species). Policies, posteriors and regret tables therefore live in a public-label
     space (the `engine_action` string, kind-prefixed) — the union over worlds. A world that
     lacks a label plays it with probability 0, so observing that action correctly ELIMINATES the
     world in the posterior (using an unrevealed move is a hard reveal).
  2. **Simulations run once; iterations re-score.** The expensive parts of a decision — one
     engine ply per (state, public action) via `fork_for_own_world` + `_play_one_ply`, and one
     batched backbone/belief pass over all fog leaves — happen once. Each regret-matching
     iteration recomputes only the posterior features and re-runs the cheap policy/value HEAD
     (`FogAgent.head_inputs` caches the trunk). Cost per extra iteration ≈ one [K·A, in_dim] MLP.
  3. **The coupling needs a consumer.** Leaf values respond to the posterior only through the
     head's posterior-features input (`model.use_posterior_features`, heads.N_POSTERIOR_FEATURES).
     With the flag off the head ignores the features, values are posterior-independent, and the
     joint optimization provably degenerates to K independent searches — that degenerate mode is
     pinned by tests/test_joint_search.py as the contrast case, not a supported configuration.

Two-sided worlds: the searcher doesn't know the OPPONENT's team either, so each own-state k is
paired with one opponent world sampled from the agent's belief net. Opponent replies come from
`opponent_policy` (default EngineMaxDamagePolicy —
reads only its own side, so no engine truth leaks; it also cannot react to beliefs, which is the
depth-one limitation documented by ARCHITECTURE.md invariant S6).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

N_ACTIONS = 14


# ------------------------------------------------------------------------------- pure math core
def action_posteriors(prior: np.ndarray, policies: np.ndarray) -> np.ndarray:
    """[K] prior x [K,A] per-state policies -> [K,A] posterior over states given each action:
    post[k,a] ∝ prior[k]·π_k(a). Columns no state plays are all-zero."""
    unnorm = prior[:, None] * policies
    z = unnorm.sum(axis=0, keepdims=True)
    return np.divide(unnorm, z, out=np.zeros_like(unnorm), where=z > 0)


def _floor_mix(pi: np.ndarray, avail: np.ndarray, floor: float) -> np.ndarray:
    """Mix a row-stochastic policy with a probability floor over available columns (the
    premature-determinism guard: smooth mixed policies keep beliefs stable during search)."""
    n_avail = avail.sum(axis=1, keepdims=True).astype(np.float64)
    f = np.minimum(floor, 0.5 / np.maximum(n_avail, 1.0))  # keep the mix a distribution
    return (1.0 - f * n_avail) * pi + f * avail


def _regret_matched(regret: np.ndarray, avail: np.ndarray, floor: float) -> np.ndarray:
    """Row-wise regret matching over available columns, floor-mixed."""
    pos = np.where(avail, np.maximum(regret, 0.0), 0.0)
    z = pos.sum(axis=1, keepdims=True)
    n_avail = avail.sum(axis=1, keepdims=True).astype(np.float64)
    uniform = np.divide(avail.astype(np.float64), n_avail, out=np.zeros_like(pos),
                        where=n_avail > 0)
    pi = np.where(z > 0, np.divide(pos, z, out=np.zeros_like(pos), where=z > 0), uniform)
    return _floor_mix(pi, avail, floor)


def joint_regret_matching(
    values_fn: Callable[[np.ndarray], np.ndarray],
    prior: np.ndarray,
    pi0: np.ndarray,
    avail: np.ndarray,
    *,
    iters: int = 16,
    floor: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """The joint loop: posterior from current policies -> action values UNDER that posterior ->
    regrets -> regret-matched policies; averaged across iterations (oscillation guard).

    `values_fn(post [K,A]) -> q [K,A]` — action values given the belief update the current joint
    policies induce (entries where ~avail are ignored). Returns (avg_policies [K,A],
    posterior-of-avg [K,A]).
    """
    prior = np.asarray(prior, dtype=np.float64)
    pi = np.asarray(pi0, dtype=np.float64).copy()
    avail = np.asarray(avail, dtype=bool)
    regret = np.zeros_like(pi)
    avg = np.zeros_like(pi)
    for _ in range(max(1, int(iters))):
        post = action_posteriors(prior, pi)
        q = values_fn(post)
        u = (pi * np.where(avail, q, 0.0)).sum(axis=1, keepdims=True)
        regret += np.where(avail, q - u, 0.0)
        pi = _regret_matched(regret, avail, floor)
        avg += pi
    avg /= max(1, int(iters))
    return avg, action_posteriors(prior, avg)


def posterior_features_np(post_col: np.ndarray, k_index: int) -> np.ndarray:
    """heads.N_POSTERIOR_FEATURES features for one (state, action) leaf from the posterior column
    of its action: [normalized entropy, mass on this state, top-1 mass, effective-state fraction]."""
    p = np.asarray(post_col, dtype=np.float64)
    k = len(p)
    nz = p[p > 0]
    h = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    h_norm = h / np.log(k) if k > 1 else 0.0
    return np.array([h_norm, float(p[k_index]), float(p.max()) if k else 0.0,
                     float(np.exp(h)) / k if k else 0.0], dtype=np.float32)


# --------------------------------------------------------------------------- engine-backed search
@dataclass
class JointSearchResult:
    action: int                      # action index on the REAL battle (execute this)
    labels: list[str]                # public-action label space (columns of the arrays below)
    policies: np.ndarray             # [K, A] averaged joint policies
    posterior: np.ndarray            # [K, A] opponent posterior under the averaged policies
    prior: np.ndarray                # [K] state weights going in
    true_index: int
    action_dist: np.ndarray          # [14] executed-state policy on real-battle action indices
    failures: int                    # (state, action) simulations that errored (root-value fallback)
    failure_reasons: dict[str, int] = field(default_factory=dict)


def joint_search_action(
    agent: Any,
    battle: Any,
    role: str = "p1",
    *,
    own_source: Any,
    opponent_policy: Any | None = None,
    iters: int = 16,
    floor: float = 0.02,
    temperature: float = 0.0,
    max_forced_steps: int = 8,
    max_batch_forward: int = 256,
    seed: int = 0,
) -> JointSearchResult:
    """One legacy joint-search decision on a live EngineBattle. See the module docstring.

    `own_source` is an own-states provider (`EnumeratedPoolStates` / `SampledOwnStates`).
    `temperature=0` plays the argmax of the true state's averaged policy; >0 samples from it.
    Never mutates `battle`.
    """
    import random as _random

    import torch

    from fog.encoding.observation import Observation
    from fog.env.engine_runner import _device, _stack_obs
    from fog.search.engine_step import _play_one_ply

    rng = _random.Random(seed)
    device = _device(agent)
    agent.eval()
    if getattr(agent, "action_oracle", None) is not None:
        raise ValueError("joint search does not support oracle-mode agents (§3.4 eval ceilings)")

    if opponent_policy is None:
        from fog.env.engine_scripted import EngineMaxDamagePolicy

        opponent_policy = EngineMaxDamagePolicy(tables=battle.tables, seed=seed)

    real_legal = np.flatnonzero(battle.action_mask(role))
    if len(real_legal) == 0:
        return _trivial_result(13)
    if len(real_legal) == 1:
        return _trivial_result(int(real_legal[0]))

    # ---- states: K own-worlds (opponent's belief about me) + K opponent worlds (my belief) ----
    own = own_source.states(battle, role)
    k_states = own.k
    my_obs = Observation.from_battle(battle.view(role), tracker=battle.trackers[role])
    my_batch = _stack_obs([my_obs], device)
    gen = torch.Generator(device=device)  # must live where sample_team's multinomial runs
    gen.manual_seed(seed)
    with torch.no_grad():
        contexts = agent.encode_contexts(my_batch)
        belief_context = agent.belief_context(contexts)
        opp_worlds = agent.belief.sample_team(
            belief_context, my_batch["opp_revealed"], k=k_states, generator=gen
        )[0].cpu().numpy().astype(np.int64)                                  # [K, 6, 15]

    # ---- root forks + the public-action label space (design decision 1) ----
    roots, per_state = [], []
    for k in range(k_states):
        fork = battle.fork_for_own_world(
            role, own.tokens[k], opp_world_tokens=opp_worlds[k], seed=rng.randrange(2**31)
        )
        legal_k = np.flatnonzero(fork.action_mask(role))
        labels_k = {_public_label(fork, role, int(a)): int(a) for a in legal_k}
        roots.append(fork)
        per_state.append(labels_k)
    labels = sorted(set().union(*per_state))
    n_labels = len(labels)
    col = {lab: j for j, lab in enumerate(labels)}
    avail = np.zeros((k_states, n_labels), dtype=bool)
    local_action = np.full((k_states, n_labels), -1, dtype=np.int64)
    for k, labels_k in enumerate(per_state):
        for lab, a in labels_k.items():
            avail[k, col[lab]] = True
            local_action[k, col[lab]] = a

    # ---- one batched root forward: per-state policy priors pi0 + root-value fallbacks ----
    root_obs = [Observation.from_battle(f.view(role), tracker=f.trackers[role]) for f in roots]
    with torch.no_grad():
        root_out = agent(_stack_obs(root_obs, device))
    root_values = root_out.value.cpu().numpy().astype(np.float64)            # [K]
    root_logp = root_out.log_probs.cpu().numpy().astype(np.float64)          # [K, 14]
    pi0 = np.zeros((k_states, n_labels), dtype=np.float64)
    for k in range(k_states):
        for j in np.flatnonzero(avail[k]):
            pi0[k, j] = np.exp(root_logp[k, local_action[k, j]])
        z = pi0[k].sum()
        pi0[k] = pi0[k] / z if z > 0 else avail[k] / max(1, avail[k].sum())
    pi0 = _floor_mix(pi0, avail, floor)  # never start with zero-support actions

    # ---- simulate each available (state, public action) ONCE (design decision 2) ----
    terminal = np.full((k_states, n_labels), np.nan)      # outcome floats land here
    leaf_of = np.full((k_states, n_labels), -1, dtype=np.int64)
    leaf_obs, leaf_state = [], []
    failures = 0
    failure_reasons: dict[str, int] = {}
    for k in range(k_states):
        for j in np.flatnonzero(avail[k]):
            fork = roots[k].clone_for_step(seed=rng.randrange(2**31))
            try:
                r = _play_one_ply(fork, role, int(local_action[k, j]), opponent_policy,
                                  max_forced_steps=max_forced_steps)
            except Exception as exc:  # noqa: BLE001 - legacy search reports every failed cell
                reason = f"{type(exc).__name__}: {exc}"[:240]
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                r = None
            if r is None:
                failures += 1
                terminal[k, j] = root_values[k]          # neutral fallback (as determinized.py)
            elif isinstance(r, (int, float)):
                terminal[k, j] = float(r)
            else:
                leaf_of[k, j] = len(leaf_obs)
                leaf_obs.append(r)
                leaf_state.append(k)

    # ---- cache leaf trunks once; values_fn re-runs only the head with fresh posterior features ----
    head_cache = []
    for lo in range(0, len(leaf_obs), max_batch_forward):
        chunk = _stack_obs(leaf_obs[lo : lo + max_batch_forward], device)
        with torch.no_grad():
            s_board_leaf, c_dyn_leaf = agent.head_inputs(chunk)
        head_cache.append((s_board_leaf, c_dyn_leaf, chunk["legality_mask"]))
    posterior_dim = int(getattr(agent.policy_value, "posterior_dim", 0))
    leaf_cols = np.full(len(leaf_obs), -1, dtype=np.int64)
    for k in range(k_states):
        for j in np.flatnonzero(leaf_of[k] >= 0):
            leaf_cols[leaf_of[k, j]] = j

    static_leaf_values: np.ndarray | None = None  # posterior_dim==0: values never change

    def _leaf_values(post: np.ndarray) -> np.ndarray:
        nonlocal static_leaf_values
        if posterior_dim == 0 and static_leaf_values is not None:
            return static_leaf_values
        feats = None
        if posterior_dim:
            feats = np.stack([
                posterior_features_np(post[:, leaf_cols[i]], leaf_state[i])
                for i in range(len(leaf_obs))
            ]) if leaf_obs else np.zeros((0, posterior_dim), dtype=np.float32)
        values = np.zeros(len(leaf_obs), dtype=np.float64)
        pos = 0
        for s_board_leaf, c_dyn_leaf, mask in head_cache:
            n = s_board_leaf.shape[0]
            pf = (torch.as_tensor(feats[pos : pos + n], device=device)
                  if feats is not None else None)
            with torch.no_grad():
                _, v = agent.policy_value(s_board_leaf, mask, c_dyn_leaf, posterior_features=pf)
            values[pos : pos + n] = v.cpu().numpy()
            pos += n
        if posterior_dim == 0:
            static_leaf_values = values
        return values

    def values_fn(post: np.ndarray) -> np.ndarray:
        lv = _leaf_values(post)
        q = np.where(np.isnan(terminal), 0.0, terminal)
        has_leaf = leaf_of >= 0
        if len(lv):
            q = np.where(has_leaf, lv[np.maximum(leaf_of, 0)], q)
        return q

    # ---- the joint loop (design decision 3 lives inside values_fn) ----
    avg_pi, posterior = joint_regret_matching(
        values_fn, own.weights, pi0, avail, iters=iters, floor=floor
    )

    # ---- execute the TRUE state's averaged policy, mapped back to real-battle action indices ----
    kt = own.true_index
    action_dist = np.zeros(N_ACTIONS, dtype=np.float64)
    for j in np.flatnonzero(avail[kt]):
        action_dist[local_action[kt, j]] += avg_pi[kt, j]
    # true-world fork action indexing must agree with the real battle (same moves, same species);
    # renormalize over the REAL mask defensively rather than trusting it blindly
    mask = battle.action_mask(role)
    action_dist = np.where(mask, action_dist, 0.0)
    if action_dist.sum() <= 0:
        action_dist = mask.astype(np.float64)
    action_dist /= action_dist.sum()
    if temperature <= 0:
        action = int(action_dist.argmax())
    else:
        p = action_dist ** (1.0 / max(temperature, 1e-6))
        p /= p.sum()
        action = int(rng.choices(range(N_ACTIONS), weights=p)[0])

    return JointSearchResult(
        action=action, labels=labels, policies=avg_pi, posterior=posterior,
        prior=own.weights.copy(), true_index=kt, action_dist=action_dist, failures=failures,
        failure_reasons=failure_reasons,
    )


def _public_label(fork: Any, role: str, action_idx: int) -> str:
    """What the opponent observes if `role` plays `action_idx` in this world: kind-prefixed engine
    move string (move id / '-tera' variant, or switch-target species). Slot indices are NOT public.

    Lowercased: sides built from pastes carry poke-engine's own casing while world-fork sides are
    rebuilt from lowercase table ids, so without normalization a label computed on the live battle
    (fog/search/posterior_log.py) never matches the fork-built label space."""
    from fog.constants.schema import ACTIONS, ActionType

    kind, _ = ACTIONS[action_idx]
    prefix = "s" if kind is ActionType.SWITCH else ("f" if kind is ActionType.FORCED else "m")
    return f"{prefix}:{fork.engine_action(role, action_idx)}".lower()


def _trivial_result(action: int) -> JointSearchResult:
    dist = np.zeros(N_ACTIONS)
    dist[action] = 1.0
    return JointSearchResult(
        action=action, labels=[], policies=np.zeros((0, 0)), posterior=np.zeros((0, 0)),
        prior=np.ones(1), true_index=0, action_dist=dist, failures=0,
    )
