# FOG architecture

This document describes the intended model, search target, and training flow, and identifies
differences between the intended design and the current implementation.

## Problem boundary

At a decision on turn `t`, the agent has three information sources:

1. `private_state`: the acting player's submitted six sets and still-private mechanics such as PP.
2. `public_state`: the current public board, including both parties, field state, revealed facts,
   HP, status, boosts, and legal public identities.
3. `history`: the ordered sequence of completed public turns, including actions, reveals, damage,
   speed order, switches, and start-of-turn state.

The current board and history are both public, but they are not interchangeable. The board is an
exact fixed-width snapshot at one turn. History is a capped sequence whose value comes from order
and cross-turn evidence. Keeping them separate gives the board a short, non-sequential path and
lets the history computation stay bounded by the configured turn cap.

Opponent truth is never an ordinary model input. It is allowed only as a supervised target, as a
simulator-owned search world, or in an explicitly labeled oracle evaluation.

## Encoders and heads

`fog.models.three_encoder.Backbone` owns three distinct encoders:

- `CurrentStateEncoder` removes the acting player's private set fields, embeds categorical public
  identities, encodes each entity independently, and pools the public board and field.
- `TurnHistoryEncoder` combines numerical turn records with categorical sidecars and start-of-turn
  state. The default is a causal transformer with learned positions; a GRU remains an ablation.
- `PrivateStateEncoder` reads only the acting player's six complete private records. Current public
  HP/status information stays in the current-state encoder rather than being duplicated here.

The checkpoint-compatible path fuses current-state and history into `public_context`. Its consumers
are:

- Policy and value: `private_state + public_context`.
- Opponent belief: `public_context + private_to_belief(private_state)`. Our set is relevant evidence
  when interpreting damage, speed order, and action constraints.
- Self-belief: the same construction from the opponent's censored perspective, using the
  opponent-view private state. It estimates what the opponent can infer about us without receiving
  our inaccessible private tensor through that perspective.

All policy, value, and belief consumers therefore use the three conceptual information sources.
The belief heads do not feed a pooled belief vector back into the ordinary policy. Search supplies
belief-dependent strategy information outside the base forward pass.

### Revised typed-token branch (active experiment)

The active experimental replacement is configured by
`configs/gen9ou_typed_token_trunk.yaml`. It contains 24 learned 128-wide tokens: six complete
own-private Pokemon, six opponent-public Pokemon, field, request/control, eight recent
information-set history records, one long-history summary, and one learned global workspace.
Every fact crosses a learned encoder; there is no raw-value or exact-state bypass. Four layers of
self-attention contextualize all tokens without averaging Pokemon together.

History is cut at the decision request. Recent history remains available as eight separate
chronological tokens while the causal history encoder supplies the long summary. Own and opponent
Pokemon share a sub-encoder but receive different type and side embeddings. Opponent inputs contain
only public reveals; missing values retain unknown sentinels and visibility flags.

Belief, value, and policy can query the complete contextual sequence. Candidate actions carry
their own move or destination-Pokemon reference and use one shared action-query scorer. No learned
absolute candidate-index embedding is used, so a consistent candidate permutation produces the
same permutation of logits. Masked visible-state prediction is a modest regularizer rather than a
95-percent promotion gate.

The following flat and sixteen-token branches are preserved only as historical diagnostics, not as
the basis of new training:

The active replacement branch was deliberately simpler:
`SharedBattleLatent(flat_autoencoder=true)`. It concatenates every legitimate current input and its
learned semantic embeddings, then sends that complete input through a global MLP into one
unpartitioned 2,048-number vector. There is no transformer, token partition, raw-input bypass, or
analytically reversible section. The representation is therefore a genuine learned compression,
not a structured copy of the input. A training-only global decoder must reconstruct the complete
current state from only that one vector and is discarded after pretraining.

The earlier sixteen-token contextual branch is retained only as an experiment. Its strict
reconstruction runs plateaued around 34–36%, so it is not the basis for belief, value, or policy
training. The flat branch first has to memorize one fixed legitimate batch at the 95% per-field
gate. This isolates architecture and objective bugs before any broad-data generalization claim.

The required current-state content is 1,961 normalized scalar/code values, excluding history as
agreed. This count makes 2,048 a reasonable starting width, but it is not proof of learned
retention. Learned trunks pass only when a decoder recovers at least 95% of every required semantic
field on both untouched whole-team and separate-battle splits. The one-batch test must pass first;
the untouched splits remain the actual generalization gate.

High-cardinality categorical fields use embeddings. Native set truth is represented as a `[6, 15]`
integer grid:

```text
species, item, ability, move0..move3,
nature, ev_hp, ev_atk, ev_def, ev_spa, ev_spd, ev_spe, tera
```

## Belief model

`fog.models.belief.BeliefModel` is a set-autoregressive model over the opponent's complete team.
Known facts are clamped before sampling. Hidden fields are generated through a mechanically legal
route, with generated facts added back to the set context. Move order and team storage order are not
intended to carry semantic meaning.

The autoregressive factorization is useful because plausible complete teams occupy a sparse subset
of the combinatorial team space. Search consumes complete sampled candidates, not independent field
marginals. Candidate scoring must account for generation-route probability rather than treating one
arbitrary route as the probability of an unordered team.

Public replays do not reveal exact nature, EVs, or every submitted set field. Unrevealed facts remain
censored belief targets. Complete team pastes and simulator-owned self-play truth are the sources for
complete-team supervision.

## Search target and current status

The intended expert is a two-sided public-belief search. It maintains candidate private teams for
both players, evaluates their legal joint-action cells and chance outcomes, and solves for a mixed
strategy over the supported private types. The executed policy is the averaged strategy row for the
agent's actual private type. It is not the argmax against one assumed opponent response.

This construction is the mechanism intended to avoid strategy fusion: private types are solved
together in one Bayesian game, so the opponent strategy and public belief updates are shared across
types. It only provides that protection when the two-sided solver is actually used and its candidate
support is adequate.

Current implementation status:

- `fog.search.pb_search` and `fog.search.two_sided` implement the experimental two-sided candidate
  grid and regret-matching solve on the engine-backed research path.
- `fog.search.direct_two_sided` and `FogPlayer`'s explicit `direct_two_sided` mode connect the same
  regret solver to the authoritative in-memory Showdown action axes and exact chance branches. The
  bridge currently has one private world per player; it is a simultaneous complete-information
  action solve, not yet a hidden-team public-belief claim.
- The older one-ply determinized search remains available only as a labeled legacy diagnostic. It
  samples opponent worlds and does not solve the two-sided equilibrium.
- Substituting multiple hidden-team candidates into a mid-battle direct Showdown state remains an
  experimental research problem. Strict transcript reconstruction is diagnostic because a
  candidate can consume a different sequence of irrelevant random calls; it must not silently
  remove belief support.
- The learned public-belief value network has not passed its public-only ablation and is not a
  promoted default.

Consequently, all search modes are experimental data/evaluation paths, not production inference
claims. The direct complete-information bridge now covers legality, simultaneous actions, and
chance enumeration. Search-generated policy targets still require multi-candidate information
isolation, search-on/off, and held-out strength gates before promotion.

### Search design invariants

These identifiers replace references in source comments to untracked historical planning files:

- **S1 — Two-sided private-state support.** Search represents plausible private teams for both
  players. Exact enumeration is a validation rung; sampled candidates are the scalable path.
- **S2 — Posterior-aware leaf scoring.** Optional posterior features summarize what the opponent
  may infer about our private state. Shared board encodings may be cached, while leaf heads are
  rescored when those posterior features change.
- **S3 — Information isolation.** Each simulated player and reply policy receives only information
  available from its own perspective. Simulator-owned truth never becomes an ordinary policy input.
- **S4 — Current-policy consistency.** When policy changes materially, belief training data should
  be refreshed from the changed policy before belief-dependent search or distillation is treated as
  consistent with that policy.
- **S5 — Evaluation requirements.** Exact-versus-sampled agreement and search strength must be
  measured on held-out team pools. Results confined to a small memorized pool are diagnostic only.
- **S6 — Experimental one-ply limitation.** The legacy joint-search path pairs sampled worlds with
  a fixed reply policy. It is a documented depth-one approximation, not the intended two-sided
  equilibrium solver.

## Training strategy

The environment reward is terminal and extremely sparse. The active typed-token branch is staged:

1. Pretrain the flat shared trunk with exact current-state reconstruction and masked/denoising
   reconstruction. A fresh held-out decoder must reach the per-field 95% gate. History remains an
   auxiliary semantic input rather than an exact-copy target.
2. Pretrain opponent belief on replay-observable censored facts, and on exact complete-team truth
   from published team pastes or self-play. Belief cross-entropy supplies dense representation
   learning without pretending that guessing information is an environment reward. Keep the
   reconstruction objective active so information retention cannot silently regress.
3. Train value from legitimate final outcomes with belief and reconstruction retention active.
4. Train policy last. High-rated replay actions can initialize the new head, but their unknown
   private sheets remain provenance-tracked imputations rather than truth.
5. Fine-tune with self-play PPO against a mixture of scripted, current, and frozen past policies.
   Optional progress shaping may help early optimization, but terminal win rate remains the target
   and must be evaluated separately.
6. Once the search expert is validated, distill its full move distribution into policy and refresh
   belief models on play from the changed policy. Search may still run at inference time, analogous
   to policy-guided search systems; distillation is not a reason to remove it.

One PPO or behavior-cloning run can be controlled through `fog.train.TrainingSession`:

```python
from fog.train import TrainingSession

session = TrainingSession.from_file("configs/bc_ou.yaml")
session.config["bc"]["lr"] = 2e-4
metrics = session.run()
```

The same interface is available as `python -m fog.train --config ...`. The automated block driver
in `fog.scripts.automated_training` remains an orchestration layer around PPO; it is not a second
training algorithm.

## Simulator authority

Pinned Pokémon Showdown is authoritative for mechanics, censored observations, final evaluation,
and future search evaluations. The direct worker avoids a server without changing the observation
protocol. Poke-engine remains useful for fast diagnostics and explicitly labeled approximation
experiments. Cross-simulator agreement is a gate, not an assumption.
