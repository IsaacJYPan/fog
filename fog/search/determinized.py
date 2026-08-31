"""Root-level determinized evaluation (spec §8). Inference-time only — NEVER in the training loop.

For each legal action a (≤14): simulate one ply in each of k opponent worlds sampled from the
belief model (clamped to reveals), evaluate the resulting position, average over worlds, and blend
with the policy prior. Play the argmax.

Three properties are load-bearing (each fixes a defect of the original root determinization):

  1. `step_fn` is REQUIRED. Without a simulator nothing downstream of the root depends on the
     action, every action ties, and argmax silently picks the lowest legal index — the old
     `step_fn=None` fallback was exactly that bug. No simulator => no search; use the policy.
  2. Leaves are evaluated UNDER FOG. `step_fn` returns the searcher's censored `Observation`
     after the ply — root reveals plus whatever the simulated ply itself revealed, hidden
     attributes still hidden. The sampled world drives the simulator only. This keeps the value
     net on its training distribution AND prices information: scouting actions produce sharper
     leaf beliefs, which the belief-conditioned value net can genuinely score higher. (Clamping
     the full sampled team into the leaf belief input — what this module originally did — erases
     information value and hands the value net an input it never trained on.)
  3. The search score is blended with the policy prior: score(a) = mean leaf value + λ·log π(a).
     Determinization is structurally blind to concealment (it assumes worlds resolve); the prior
     is trained under the §7 leakage objective, so it carries what search cannot see. λ=0
     recovers pure search; large λ recovers the policy.

`step_fn(action_idx, world_tokens) -> Observation | float | None`
  * Observation — the searcher's fog observation at the leaf (numpy, unbatched)
  * float      — the simulated game ENDED: terminal outcome in [-1, 1] from the searcher's side
  * None       — this (action, world) simulation failed; the sample is skipped

`fog.search.engine_step.EngineWorldStepper` is the poke-engine implementation of this contract.

All Observation leaves are scored in ONE batched value-net forward (chunked at
`max_batch_forward`), not k×|legal| single-item calls. Actions whose every sample failed fall
back to the root state's value (neutral: search neither favors nor buries them).
"""
from __future__ import annotations

import torch


def root_belief_context(agent, batch: dict, contexts=None):
    """Build the belief context through the same shared-trunk route as ``FogAgent.forward``.

    Search predates the typed shared latent and used to call ``belief_context(contexts)``
    directly.  That is valid for the legacy encoder, but it either fails or bypasses the learned
    common state for current typed agents.  Keep this helper inference-only and derive everything
    from the same censored root batch used by the acting policy.
    """
    if contexts is None:
        contexts = agent.encode_contexts(batch)
    shared_state = None
    if getattr(agent, "shared_latent", None) is not None:
        shared_state = agent.shared_latent(batch, contexts, agent.backbone.embeds)
        if isinstance(shared_state, tuple):
            shared_state = shared_state[0]
    return agent.belief_context(contexts, shared_state=shared_state)


@torch.no_grad()
def determinized_action(
    agent,
    batch: dict,
    *,
    step_fn,
    k: int = 16,
    prior_weight: float = 0.5,
    max_batch_forward: int = 256,
    generator: torch.Generator | None = None,
    min_override_margin: float = 0.0,
    uncertainty_z: float = 0.0,
) -> int:
    """Return the argmax action under root-level determinized rollout for a single decision.

    `batch`: a batched (B=1) FogAgent input dict built from the searcher's FOG view (as FogPlayer /
    EngineRunner build). `step_fn`: see module docstring — required. `prior_weight` is λ in
    score(a) = mean_w value(leaf(a, w)) + λ·log π(a|root); leaf values and terminal outcomes live
    in [-1, 1].
    """
    if step_fn is None:
        raise ValueError(
            "determinized_action requires a step_fn: without a simulator the leaf value cannot "
            "depend on the action and every action ties (see module docstring). Use the policy "
            "directly when no simulator is available."
        )
    agent.eval()
    legal = batch["legality_mask"][0].nonzero(as_tuple=True)[0].tolist()
    if len(legal) <= 1:
        return legal[0] if legal else 13

    out = agent(batch)  # root fog forward: policy prior + neutral fallback value
    log_probs = out.log_probs[0]
    root_value = float(out.value[0])

    contexts = agent.encode_contexts(batch)
    belief_context = root_belief_context(agent, batch, contexts)
    worlds = agent.belief.sample_team(
        belief_context, batch["opp_revealed"], k=k, generator=generator
    )[0].cpu().numpy()                                                 # [k, 6, 15]

    # simulate every (action, world) ply; terminal outcomes score directly, fog leaves batch up
    scores_by_action: dict[int, list[float]] = {a: [] for a in legal}
    leaf_obs, leaf_action = [], []
    for a in legal:
        for w in range(k):
            r = step_fn(a, worlds[w])
            if r is None:
                continue
            if isinstance(r, (int, float)):
                scores_by_action[a].append(float(r))
            else:
                leaf_obs.append(r)
                leaf_action.append(a)

    device = batch["legality_mask"].device
    for lo in range(0, len(leaf_obs), max_batch_forward):
        chunk = leaf_obs[lo : lo + max_batch_forward]
        values = agent(_stack_observations(chunk, device)).value       # [len(chunk)]
        for a, v in zip(leaf_action[lo : lo + max_batch_forward], values.tolist()):
            scores_by_action[a].append(v)

    scores = torch.full((14,), float("-inf"))
    standard_errors: dict[int, float] = {}
    for a in legal:
        vals = scores_by_action[a]
        search = sum(vals) / len(vals) if vals else root_value
        scores[a] = search + prior_weight * float(log_probs[a])
        if len(vals) > 1:
            standard_errors[a] = float(torch.tensor(vals).std(unbiased=True) / len(vals) ** 0.5)
        else:
            standard_errors[a] = 0.0
    candidate = int(scores.argmax())
    policy_action = max(legal, key=lambda action: float(log_probs[action]))
    if candidate == policy_action:
        return candidate
    uncertainty = float(uncertainty_z) * (
        standard_errors[candidate] ** 2 + standard_errors[policy_action] ** 2
    ) ** 0.5
    required = float(min_override_margin) + uncertainty
    return candidate if float(scores[candidate] - scores[policy_action]) >= required else policy_action


def _stack_observations(obs_list: list, device) -> dict:
    """Numpy Observations -> one batched FogAgent input dict (same layout the runner builds)."""
    from fog.env.engine_runner import _stack_obs

    return _stack_obs(obs_list, device)
