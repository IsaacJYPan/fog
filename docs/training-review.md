# Reviewing training results

Every completed training experiment receives a written review before another large
training run is chosen. This includes experiments that complete successfully but do
not establish a playing-strength improvement. Keep the review with the frozen
inputs, returned outputs, and handoff, using hashes to identify its evidence.

The review should answer:

1. **What changed, and what was actually trained?** Record initialization, objective,
   data, completed optimizer updates, learning-rate schedule, sampled positions,
   unique data exposure where available, and successful training time. Failed setup
   time or reserved GPU hours do not measure the model's training exposure.
2. **What did evaluation establish?** Compare against the unchanged reference under
   matched teams, seats, seeds, and inference settings. Report scores, uncertainty,
   failures, and data reuse. Reproducing the same games on another machine is a
   correctness check, not independent evidence of strength.
3. **Where did learning occur?** Compare training and held-out losses, intermediate
   checkpoints, prediction calibration, and behavior. For the belief decoder,
   distinguish public marginals, conditionals given correct prefixes, and ancestral
   samples. Mask unverified labels. Save per-game measurements because fields and
   contexts from one battle are correlated.
4. **Which explanations remain plausible?** Consider undertraining, overfitting,
   incorrect inputs or targets, weak or conflicting gradients, data distribution
   shifts, inadequate search support, and an evaluation too small to resolve the
   expected gain. Separate observations from causal explanations. Gradient norms
   alone do not measure Adam's parameter displacement.
5. **What next experiment would distinguish them?** Choose a bounded change with a
   predicted measurable effect. Continue training when held-out progress supports
   it. If held-out progress stalls or reverses, investigate the recipe, supervision,
   inputs, or optimization rather than extending the run solely because it used
   substantial compute. No noisy development screen establishes architectural
   saturation.

Development retention follows the [native evaluation policy](native-kakuna.md):
keep a valid checkpoint unless the comparison supports a regression. Retention is
separate from claiming improvement, connecting learned beliefs to search, or
passing final confirmation. Preserve previous checkpoints in either case.

## Current native attempt

The matched 1,250-update runs completed on identical sampled data schedules. Each
sampled 489,710 decision positions; these are not 489,710 unique decisions. Ordinary
training took about 41 minutes and joint training about 82 minutes, including saves.
The 500-update warmup leaves 750 updates after warmup. The auxiliary heads began
with independent random weights.

On 256 crossed development games per condition against unchanged Kakuna, ordinary
fine-tuning scored 48.44% (exploratory 95% interval 43.75–53.13%) and joint training
50.39% (44.92–55.47%). The joint-minus-ordinary estimate was +1.95 percentage points
(−1.95 to +5.86). All games completed without action failures. The subsequent CPU
cloud run reproduced the same schedules and outcomes; it does not add a separate
evaluation cohort. These results establish neither a gain nor a regression.

The 10.6% learned versus 50.1% existing-inference result measures **hidden-item
prediction**, not battle win rate. It compares the modal item from 32 sampled sets
with verified submitted-team labels across 32 development games. The existing
comparator uses FOG's training-pool donor/prior inference. It is not Metamon's full
replay reconstruction. Ordinary policy inference does not consume either sampler.

Joint marginal and chain training losses fell roughly in half, while held-out
ancestral set prediction remained weak. Weighted auxiliary shared gradients averaged
about 0.26% of the primary norm; the 25% cap was never active. These observations
justify testing held-out checkpoint progression and the decoder's conditional
predictions. They do not establish that more updates, larger auxiliary weights, or
a different architecture will improve play. A fixed CPU benchmark against
SimpleHeuristics and MaxDamage measures basic competence; it does not certify
competitive ladder strength.

The next decision should use those diagnostics. More training remains a plausible
hypothesis, especially for the new heads, but has not yet been demonstrated to be
the missing ingredient.
