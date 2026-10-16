# Kakuna belief adaptation

The working hypothesis is that useful hidden-team information can be learned in
Kakuna's shared representation while its ordinary decisions remain close to the
protected checkpoint 34. This is a new, bounded attempt, separate from the closed
native fine-tuning recipe and the independently initialized FOG experiments.
Neither architectural saturation nor a playing-strength gain has been established.

## Why change the experiment

Earlier native training combined a large offline actor/critic objective with
small auxiliary losses. The recorded weighted auxiliary backbone gradient averaged
0.548% of the primary gradient. That does not establish ineffective supervision,
but a maximum gradient ratio of 25% did not ensure useful representation learning.
Correct-prefix and ancestral reconstruction also remained different problems.

This experiment makes exact-team belief supervision the learning objective and
protects policy behavior directly. It first fits the existing fact-set/AR decoder
with the native network frozen. Two subsequent conditions start from the same
fitted heads and use the same observations, labels, query RNG and head-update
budget: frozen native features, and an adapted final native Transformer block
plus final normalization. Actor, critic and all earlier native modules remain
frozen. Both actor and belief decoder read the adapted shared representation.
Outcome supervision has identical weight in both conditions; a strength benefit
cannot automatically be attributed to belief supervision without a further ablation.

## Context and information contracts

Use trace-verified fresh games with submitted teams and composition-disjoint
training/development pools. The earlier 999 accepted training games and 32
development games are a starting corpus, not a claim of adequate diversity or an
independent final test. Repeated training draws are not new decisions.

The lower nine Transformer blocks and observation encoder are frozen in evaluation
mode, including normalization and SigmaReparam buffers. Cache their input to the
last block using complete chronological histories, never cropped full-network
histories. Cache targets separately. The last block has a 96-step causal window;
its last-token output can therefore be recomputed from at most 97 cached lower
features. Earlier cached features already encode earlier history. This shortcut
requires an exact-source/window check and numerical comparison with full native
and cached inference. It must not be generalized to a different upstream model.
Right padding cannot influence an earlier valid causal output. Preserve absolute
positions for rotary embeddings where applicable.

Complete opposing teams are labels only. They cannot enter cached native features,
policy-retention targets or ordinary actions. Reject unverified/impossible labels,
unexpected request corrections, changed input hashes, cropped histories and
composition overlap. Preserve original checkpoints and all previous results.

## Training and retention

Warm the decoder/projection/outcome heads on original Kakuna features. Any reuse
of older auxiliary weights must declare and verify their original native pairing,
then explicitly rebase them through this frozen-feature phase. Reset between
experimental phases only when the recipe declares it; interrupted runs require
complete optimizer, sampler, route and CPU/CUDA RNG state.

During adaptation, keep native modules in evaluation mode while enabling gradients
only in the final block and final normalization. This prevents frozen dropout,
running statistics or power-iteration buffers from drifting. Train the new heads
normally. Keep the original frozen native actor and critic as behavior references.
Use legal-action KL and native shaped-action-value mean consistency penalties; the
native critic is not a win-probability target. Do not suppress belief gradients
merely because an initially exact policy-retention gradient is zero.

Each native update must pass finite-value and policy/value drift bounds on its
sampled contexts. Rejected native updates restore their parameters and Adam state;
head updates and exposure counters remain explicit. A separate training-only
sentinel bank checks cumulative drift. Development observations do not select
training updates. These are sampled safeguards, not guarantees outside those
histories, and they do not prove a win-rate bound. Record KL tails, argmax changes,
gradient norms, accepted/rejected native updates and all learning rates.

## Decision gates and compute

1. Complete local unit tests and a real-checkpoint trial before cloud execution.
   Verify the feature shortcut, label isolation, frozen buffers, exact resume,
   illegal-action masking and both successful and rejected native updates.
2. Run the full feature extraction and environment check on Kaggle CPU. This keeps
   the laptop's work small and avoids spending GPU time on deterministic extraction.
3. Freeze a new GPU identity using the returned cache hashes. Start with a bounded
   paired pilot, reserving setup/export time. Check actual CUDA probabilities and
   interrupted/resumed training before main updates. Fix equal head-update budgets
   and query schedules before either condition trains. Use measured throughput to
   choose subsequent budgets, not the entire remaining weekly quota at once.
4. Evaluate belief marginals, correct-prefix conditionals, ancestral joint samples
   and context sensitivity separately. Compare against existing training-only set
   inference; a nicer training loss is insufficient. Keep per-game/group evidence.
5. Run a separate private Kaggle CPU play comparison against original Kakuna,
   SimpleHeuristics and strong opponents on crossed development schedules. Small
   mean KL or an interval containing 50% does not establish non-inferiority. Use
   the existing predeclared regression margins and preserve uncertain candidates.

The first success criterion is useful belief improvement with no detected material
loss of ordinary play. The eventual project criterion remains improvement over
unchanged Kakuna. Search should use learned beliefs only after reconstruction and
matched fixed-search comparisons support them. The separate native search bridge
still has an unresolved hard-deadline/history-copying problem. Offline endgame
analysis can supply stronger policy/value targets before online search is ready;
optimal-continuation values must not silently replace current-policy PPO values.

Each stage receives hashed inputs, bounded runtime, periodic full-state exports,
explicit failure evidence and a handoff. Private Kaggle jobs and credentials remain
outside Git. Final confirmation cohorts stay untouched during these diagnostics.
