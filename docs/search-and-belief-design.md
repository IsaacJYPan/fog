# Search and belief design

This document fixes the theory decisions for search under hidden teams, so that later work does
not reopen them without new evidence. [ROADMAP.md](../ROADMAP.md) says when each part is built.
Sections 1–3 are reasoning; sections 4–5 are specifications to implement; section 6 explains
what is deliberately not being built.

## 1. Setting and notation

- `I`: our information state. Our own team sheet, the public board and the public history.
- `w`: a world. A complete assignment of the opponent's hidden team sheet (and symmetric
  hidden facts such as their remaining PP).
- `b(w | I)`: our belief over worlds.
- `O_opp(w, I)`: what the opponent would observe in world `w`. It contains their own sheet and
  the public facts, and **only the publicly revealed part of our team**.
- `π`: the blueprint policy, a network mapping an information state to action probabilities.
  It is used for both players, each from its own censored observation.
- `V^π(I)`: expected outcome in `[-1, 1]` from information state `I` when both players follow
  `π` from here. It is a function of an information state, never of a full world.

Turns are simultaneous. Chance is damage rolls, accuracy, secondary effects, speed ties.

## 2. Strategy fusion: what it is here and what removes it

Determinized search samples worlds and searches each as a perfect-information game. Three
distinct errors hide under the name "strategy fusion". They have different costs and different
fixes, so keep them apart.

**(F1) Fusion at the root.** Choosing the best action separately in each world and then voting
lets the searcher "play a different move in each world", which it cannot do. Fix: score each
root action once across all worlds, `Q(I, a) = Σ_w b(w | I) Q_w(a)`, and pick from `Q`. Every
search in this repository must aggregate at the root this way. Aggregating visit fractions
(as EngineMCTS does) is the same idea with visits as the score.

**(F2) Fusion below the root.** Inside a per-world tree, our own later decisions are optimised
knowing `w`. The search therefore assumes we will play perfectly against the true hidden set
later, undervalues scouting and safe plays, and overvalues lines that only work if we guess
right. Fix: either keep the tree shallow so few own decisions are inside it, or share one tree
across worlds keyed by what we would observe (section 5).

**(F3) An omniscient opponent.** In a per-world tree the opponent's nodes also know our full
sheet. The search then assumes every hidden item and move of ours is already known, so it never
values concealment and is too pessimistic about surprise sets. Fix: generate the opponent's
action distribution from `π(· | O_opp(w, I))`, the policy applied to the opponent's censored
view. This is the role of "keeping partial information in search" and it is correct. It fixes
F3. It does not fix F2.

The value function matters for the same reason. A leaf evaluated as `V^π(I')`, with `I'` our
censored observation at the leaf, already averages over what we still do not know and over an
opponent who does not know our sheet. A leaf evaluated with full knowledge of `w` reintroduces
F2 and F3 beyond the leaf. So: **leaf values are always computed from censored observations**,
which `fog/search/determinized.py` and `fog/search/engine_step.py` already do.

What this buys formally. Fix the opponent to `π`. From our side the game is then a
single-agent partially observed decision problem, and

```text
Q^π(I, a) = E_{w ~ b(·|I)}  E_{a' ~ π(·|O_opp(w,I))}  E_{chance}  [ V^π(I') ]
```

is the exact one-step action value. Acting greedily on it is a policy improvement step against
`π`: it cannot do worse than `π` against `π`, provided the three ingredients are right (worlds
drawn from the true posterior, the opponent's actions drawn from `π` on its own view, `V^π`
accurate). This holds for depth one. It is a best-response step, not an equilibrium, which is
the next section.

## 3. What happens when the policy shifts

"Search uses the policy, so if the policy changes the search was computed against the wrong
thing." There are four separate dependencies. Three are harmless if handled as below; one is a
real limitation that is managed, not solved.

### 3.1 Search targets are relative to the current policy

This is expert iteration working as intended. Search output is "something better than `π_n`";
distilling it gives `π_{n+1}`; the targets for `π_n` are then stale. Rules:

- Regenerate targets every generation. Train on the two most recent generations only.
- Never mix targets produced by different searchers in one generation without recording it.
- Promotion is by match result against the previous generation and the fixed anchors
  (ROADMAP P5.1), not by training loss.

### 3.2 The value function is relative to a policy pair

`V^π` is stale as soon as `π` changes. Train the value head on outcomes of games played by the
searcher of the same generation (and, in generation 0, on human game outcomes), blended with
the search value at the root as in ROADMAP P3.1. Check calibration each generation.

### 3.3 Beliefs depend on the opponent's policy

Write the posterior as three factors:

```text
b(w | I)  ∝  prior(w)  ×  1[w consistent with every revealed fact and mechanical evidence]
                        ×  L(opponent's past actions | w, π_opp)
```

The first two factors do not depend on anyone's policy. They come from usage statistics, team
structure, revealed moves, items and abilities, speed order and damage ranges. In this game
most hidden information becomes known through those mechanical channels. The third factor is
the only policy-dependent one, and it is also the dangerous one: with a misspecified opponent
model, multiplying action likelihoods over twenty turns produces confident wrong beliefs.

Decision: **all search up to gate G4 uses only the first two factors.** The belief model and
the usage sampler are both policy-independent, so nothing about beliefs has to be recomputed
when the policy changes, and the set model can be trained once on human data and exact-truth
logs. Behaviour likelihoods stay in the parking lot. If they are ever tried, use the floor and
temperature already implemented in `JointBeliefTable.behavior_step`, and accept them only if
the posterior log-probability of the true sheet improves on held-out human games, not just on
self-play.

Dropping the third factor makes `b` an approximation of the true posterior, so the improvement
property in section 2 holds only approximately. That trade is accepted: a stationary, slightly
blunt belief is better than a sharp one that silently depends on a moving policy.

### 3.4 Best response is exploitable

A greedy improvement against `π` can be deterministic where a mixed strategy is needed (the
classic two-option predictions). An opponent who knows the agent can exploit that. Full
equilibrium search would address it and is out of scope (section 6). The managed approach:

- Do not play the argmax of `Q`. Play from `π'(a) ∝ π(a) · exp(Q(I, a) / τ)`, the maximiser of
  `⟨π', Q⟩ − τ · KL(π' ‖ π)`. Start with `τ` such that the best action gets about 80% in
  positions where the top two `Q` values differ by 0.05; tune by match result. For visit-based
  searches the equivalent is sampling from `visits^(1/T)` with `T` around 0.5 instead of 0.
- Train against a population: previous generations and the fixed anchors, not only the latest.
- Measure exploitability empirically with the exploiter check in ROADMAP P5.2.

This gives no convergence guarantee. The claim made is only: each generation is at least as
strong as the last against the population it was tested on, and a fixed-budget exploiter has
been tried against it.

## 4. Specification: FogMCTS (ROADMAP P4.2)

Decoupled PUCT on the Rust engine with learned prior and value, one tree per sampled world,
aggregated at the root. It accepts F2 inside each tree (bounded by a small simulation count)
and removes F1 and F3.

Per decision:

1. Sample `K` worlds `w_1..w_K` with weights `ρ_k` (usage sampler, or the belief model after
   P4.1). Reconstruct an `EngineBattle` for each (`fog.env.state_recon.reconstruct_battle`).
2. For each world keep a tree. A node is an engine state reached by a joint action and a sampled
   chance outcome. Each node stores, for each player separately, per-action `N`, `W`, and prior
   `P`.
3. One simulation in world `k`:
   - **Select.** At a node, each player independently picks
     `argmax_a  W/N + c_puct · P(a) · sqrt(ΣN) / (1 + N(a))` over its own legal actions
     (`c_puct` 1.5 to start; unvisited actions use the parent's mean value). The pair is the
     joint action.
   - **Step.** `poke_engine.generate_instructions(state, a1, a2)` returns weighted outcome
     branches. Sample one branch by weight (do not expand all; chance widths are large). Apply.
     Resolve forced replacements with the policy's top choice, as `_play_one_ply` does.
   - **Expand and evaluate.** At a new node build **two censored observations**, ours and the
     opponent's, through the same encoder used in play (`EngineBattle.view(role)` then
     `Observation.from_battle`). The opponent's observation must be built from its view object,
     never from the raw state. One batched forward gives `P` for each player from its own
     observation and the leaf value `v` from ours. Terminal states use the true result.
   - **Back up** `v` for us and `−v` for the opponent along the path.
4. Run simulations in waves: collect one pending leaf from each of the `K` trees (virtual loss
   1 on the path), evaluate all leaves in one forward, back up, repeat until the time budget.
5. Root output: `p(a) = Σ_k ρ_k · N_k(a) / Σ_a N_k(a)` over our actions, and root value
   `Σ_k ρ_k · W_k / N_k`. Play by sampling `p^(1/T)`, `T = 0.5` (section 3.4). Export `p` and
   the root value as training targets.

Implementation notes:

- History inside the tree: simulated turns must be appended to a cloned tracker so the leaf
  observation's history matches what play would produce. `ShowdownWorldStepper` shows how the
  one-ply case does it.
- Budget arithmetic: the engine step is cheap (about 25,000 calls/s from Python); observation
  encoding and the forward dominate. Expect a few hundred to a few thousand simulations per
  second per core. Compare against EngineMCTS at equal wall-clock, not equal simulations.
- Keep the code backend-agnostic behind three callables (`legal_actions`, `step`,
  `observations`), so it can be unit-tested on a toy game without the engine.

Tests that must exist before any win rate is reported:

1. **Information isolation.** Changing a hidden field of our own sheet that the opponent has not
   seen must not change the opponent prior at the root. Changing a hidden field of the sampled
   world must not change our leaf observation except through simulated reveals.
2. **Toy exactness.** On a small matrix game with a pure-strategy saddle point, root values
   converge to the game value within 0.05 with the uniform prior. Do not use a game whose only
   equilibrium is mixed for this test: decoupled selection is not guaranteed to converge there.
   That is a known limitation of this family and one reason play samples from visit counts.
3. **Agreement at depth one.** With one simulation per joint action and chance sampling
   disabled, root `Q` matches `fog.search.determinized.determinized_action` on the same worlds.
4. **Determinism.** Same seed, same output.
5. **Failure accounting.** Worlds that fail reconstruction are dropped and counted; the search
   never returns an illegal action.

## 5. Specification: fusion diagnostic and the shared-tree variant (ROADMAP P4.3)

Build this only if the diagnostic shows a cost. The diagnostic is cheap:

- **Oracle gap.** Run FogMCTS twice on the same positions: once normally, once with `K = 1` and
  the true world. Record how often the chosen action differs and the difference in root value.
  A large optimistic gap on normal search (it expects much more than it gets in real outcomes,
  measured by comparing root values with final results over 1,000 games) is the signature of
  F2. Root values that are well calibrated against outcomes mean F2 is not costing much at this
  depth; stop here.

If it does cost: implement multiple-observer information-set search. One tree per player,
shared across worlds. A node in our tree is keyed by our observation history from the root
(our action plus the public observation key from `fog/search/public_observation.py`); a node in
the opponent's tree is keyed by theirs. Each simulation samples a world, then descends both
trees in lockstep, each player selecting from its own tree's statistics. Because our statistics
are shared across worlds, our deeper choices can depend only on what we would have observed.
Everything else (priors, leaf values, backup, root output) is as in section 4. Accept it only
on match result against the per-world variant at equal time.

## 6. What is not being built, and why

**Public-belief-state equilibrium search.** A sound solve needs the common-knowledge belief
over both players' hidden states and a value function over that belief. With six-member teams
and a continuous set space per member, the public belief state cannot be enumerated and a
sampled approximation needs a value network conditioned on a belief summary. The repository's
`pb_*` modules implement a sampled version; its value model failed the public-only ablation and
its second distillation generation lost strength. The modules stay frozen and tested. They may
be reopened after gate G4 for small fixed team pools, as a study of exploitability, where exact
enumeration is possible.

**Opponent modelling by exact Bayesian action inference.** See section 3.3.

**Deep search.** Past a few plies, engine approximation error, belief error and F2 all compound.
Depth comes from the value head and from distillation across generations, not from the tree.

## 7. Information-leak checklist

Every search or training change must be checked against this list.

- The policy input never contains opponent truth. Truth is allowed only as a supervised target,
  as a simulation world, or in an explicitly named oracle evaluation.
- An opponent policy inside search reads its own censored view object, never the engine state.
- Leaf observations contain root reveals plus reveals produced by the simulated moves. They
  never contain the sampled world's unrevealed fields.
- Value targets and value inputs for generation `n` come from generation `n` play.
- Evaluation teams are disjoint from training teams by composition.
- A gain measured on the engine is not a gain until it is measured on pinned Showdown.
