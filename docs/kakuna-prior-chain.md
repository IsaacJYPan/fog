# Prior-conditioned autoregressive belief pilot

The immediate goal is useful hidden-team sampling while preserving original
Kakuna34's ordinary play. The retention-replay pilot completed but its learned
sampler remained weak: item modal accuracy 12.47% versus 48.26% for the matched
small donor prior, and zero encoded-set recall@8. Identical observed game outcomes
were a preservation screen, not evidence that the belief milestone was reached.

## Diagnosis and design review

The old bank contains all 1,836 distinct verified training roster/set pairs seen
in the cache, but only 3 of 96 development pairs. Only 82 training sets remain
supported after excluding their own roster. The independent 20,000-team bank
supports 585 training pairs (552 without their own roster) and 29 development
pairs. These are support counts, not independent games or a claim that every
supported set can be inferred. The expanded bank is potentially reconstructed
prior evidence, not exact historical supervision.

Use that larger bank as an explicit conditional proposal inside the existing
unordered fact-set AR chain. Retrieve public-compatible complete sets, exclude
the query's public roster, weight teammate overlap, aggregate identical sets,
and retain at most 128 donors per slot. The cap follows public conditioning, so a
rare revealed move can retrieve a set outside the unconditional top candidates.
Each insertion filters donors against the actual generated fact prefix. Moves
are unordered, without replacement. Add the resulting conditional log prior to
the neural residual logits before sampling. A 0.001 uniform-legal mixture keeps
all legal targets reachable; inconsistent/absent donors back off explicitly.

These conditionals define a normalized route/value AR sampler and retain its
probability traces. They are not the original donor sampler, exact unordered-team
likelihoods or a calibrated Bayesian posterior. Smoothing, a finite donor cap and
set independence across slots are practical approximations. A matched prior-only
AR control is necessary; improvement over the preceding standalone neural chain
alone would mostly establish the effect of adding prior information.

Keep the native network completely frozen for this pilot. Train only the same
projection, fact encoder, decoder and outcome heads from verified cached native
features. Retain the previous frozen-control fact encoders, but reset every output
head to zero so the initial prediction is exactly the smoothed prior. Disclose the
2,032 historical head updates and the reset; no old checkpoint is overwritten.
There is no native optimizer, actor update, new native checkpoint, search or
ordinary-policy dependency on these heads. This protects ordinary policy behavior
by construction instead of asking a finite KL bank to guarantee win-rate retention.

One existing AR update predicts only one addressed fact per context. Use eight
independently sampled correct-prefix queries per context and compute only their
selected outputs. The optimized path is tested for matching conditional values
and mathematical gradients. Unused parameters may have absent rather than zero
gradients, so identical AdamW trajectories with the old full-output path are not
claimed. The paired new conditions use the same optimized path and exposure.

## Matched experiment and evidence gates

Compare the prior-only AR model with two equally trained residual conditions:
public facts plus a zero latent, and public facts plus the real frozen Kakuna
latent. Both trained conditions receive identical context schedules, teacher
queries, update counts and outcome targets. The outcome head receives real frozen
features in both conditions and clips separately; a shared clipping norm would
confound its updates with the belief-context ablation. Require identical fitted
outcome heads as an audit. This experiment measures what frozen native features
add; it does not claim to have improved the shared backbone itself.

The bounded pilot uses 2,000 updates per arm, effective batch 32, microbatch 8,
eight AR queries per context, head LR 2e-4, 100 warmup updates, cosine decay,
gradient clipping 1.5 per independent head group and outcome weight 0.1. Marginal
and AR losses each use weight 1 after their vocabulary/field normalization.
The first two-hour notebook budget failed its throughput gate before main
training: actual CUDA updates took about 3.7 seconds, implying five hours for the
sequential pair with a 20% margin. CUDA resume and probability agreement passed.
Kaggle provided two T4s, so the recovery runs one independent arm per device,
without changing the model, data, losses or per-arm update counts. Its notebook
wall-time cap is four hours, with three hours allowed per arm and at most eight
physical GPU-hours reserved. Incomplete arms fail the matched-exposure gate.
Periodic saves include heads, Adam, initial state, data/query/global RNG,
phase/update counts and immutable input identities. Main training starts from
the sealed initialization, never from discarded proof updates.
Keep a separate final full-state checkpoint for each arm so its Adam and random
streams remain available after the other arm finishes.
Each device runs four discarded resume-proof updates and eight discarded
production-shape updates. Measure throughput concurrently and stop before main
training if either device's 90th-percentile update time, with a 20% margin, cannot
fit its 2,000 updates. The 24 discarded updates are separate from 4,000 main updates.
Cancel the sibling process if either condition fails. Audit real per-arm Adam/RNG
states and matched query histories before joining results; there is no invented
combined optimizer state. The independent-process implementation must reproduce
the sequential reference, including identical outcome heads. Public-only donor retrieval
uses a bounded deterministic cache; cold and warm retrieval produce identical
probabilities and no cache state changes the optimizer or random streams.

The recovery's standalone timing process did not explicitly configure the main
training numerical flags. Its startup timing is an approximate forecast; actual
completed-run durations and update counts remain required evidence. Subsequent
packages share explicit runtime setup between timing and main training. This
runner correction does not alter the already sealed recovery or its evaluation.

Run ancestral evaluation separately on Kaggle CPU after training. Compare the
same k=8 samples for the prior-only chain, both fitted conditions, the expanded
existing donor sampler and the preserved preceding standalone decoder. Report
every field family, Brier/ECE, encoded-set accuracy/recall, correct-prefix and
public-marginal diagnostics, outcome error, runtime and failures. Frozen original
policy identity is an explicit invariant, not a new playing-strength claim.
Encoded-set metrics refer to a complete individual Pokemon set, not exact recovery
of all six members together. Primary reconstruction diagnostics are set recall@8
and hidden-item marginal NLL; other fields and calibration expose tradeoffs rather
than being discarded when their direction is inconvenient.

Retain the reused 32-game development cohort only as a diagnostic. Before claiming
generalization, collect a fresh development cohort from previously unused
development rosters, keeping the final pool untouched. Pair disjoint rosters into
whole-game groups and retain all failures. No complete opponent team enters the
player or retrieval API. The known sheets provide offline labels only. Resample
whole roster groups, not fields, turns or individual draws. Do not change the
sampler/search settings or select a recipe silently after looking at results.

Local evidence so far is a mechanism check: on eight development contexts, the
prior-only chain's sampled item accuracy was about 39%, versus about 12% for the
old neural sampler. Forty small fitting updates reduced training losses but did
not improve that small development sample. This supports a bounded matched pilot,
not a generalization or strength claim. Do not scale an unchanged recipe merely
because its training loss falls.

If useful beliefs emerge, the next stage can test small shared-backbone changes
with performance taking priority, followed by fixed-search comparisons and
self-play with exact-team belief/value supervision. If the contextual residual
does not help, diagnose prior support, likelihood and prefix errors before
allocating more GPU hours. A failed pilot is not evidence of architectural
saturation. The end goal still requires better decisions against unchanged
Kakuna and strong opponents, then independent confirmation and rated games.
