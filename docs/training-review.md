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

## Completed CPU diagnostics, October 11

The fixed CPU diagnostic completed 1,536 main games and 64 separate smoke games.
All games resolved without failures or fallbacks. Three CPU probability contracts
passed, and the slowest recorded decision took 1.18 seconds. This job performed no
training and did not use final confirmation cohorts.

Each table entry is wins out of 256 matched development games:

| Model | SimpleHeuristics | MaxDamage |
| --- | --- | --- |
| Unchanged Kakuna | 230 (89.84%) | 254 (99.22%) |
| Ordinary fine-tuning | 229 (89.45%) | 251 (98.05%) |
| Joint fine-tuning | 230 (89.84%) | 252 (98.44%) |

The joint model comfortably clears the heuristic competence target on this pool.
Its paired differences from unchanged Kakuna were 0.00 percentage points
(exploratory 95% block interval −4.69 to +5.08) against SimpleHeuristics and
−0.78 points (−2.34 to +0.78) against MaxDamage. These results support retaining
the working checkpoint; they establish neither a playing-strength improvement
nor non-inferiority against strong opponents. The independent FOG baseline was
not retrained or evaluated by this job.

Belief diagnostics used the same 32 training and 32 development games at updates
313, 625, 938 and 1,250, with matched hidden-field queries across checkpoints.
Held-out public-marginal item NLL fell from 5.27 to 3.28, while ability top-1
accuracy rose from 44.58% to 81.29%. Between the last two checkpoints,
correct-prefix held-out NLL improved for items, abilities, moves and EV amounts.
Tera NLL worsened, and nature NLL changed little. Per-game paired resampling
supports examining this continued progress, but these exploratory intervals have
no correction for multiple comparisons and games may share teams.

Item accuracy is still only 11.04% from public marginals and 10.85% on sampled
correct-prefix queries. On those identical queries, the marginal score is 10.49%.
Thus inaccurate sampled prefixes alone cannot explain weak item reconstruction.
The earlier 10.63% ancestral item accuracy is a reference on these same games,
not a new independent benchmark. Move-query accuracy is not exact moveset
accuracy, and outcome MSE remains an observed-behavior target rather than a
comparison against Kakuna's native critic.

These curves make further auxiliary optimization a concrete hypothesis. A
controlled continuation should test a separate head learning rate while keeping
the native learning rate small. It must explicitly load the retained native and
auxiliary weights under a new immutable recipe, record whether optimizer state
is retained or reset, and preserve strict full-state resume within that recipe.
Continue to evaluate ordinary and joint training under matched data and update
budgets. Keep learned sampling out of search until its held-out reconstruction
and fixed-search play comparisons support using it.

## Completed continuation evaluation, October 12

The next matched recipe loaded each arm's native and auxiliary update-1,250
weights, explicitly reset optimizer and sampling state, and added 1,800 updates.
Native learning rate stayed at 2e-5; auxiliary heads used 2e-4 with 100 warmup
updates. Each arm sampled 707,233 decision positions, including repeated draws.
The complete model, optimizer, RNG and sampler exports passed finite-state and
counter checks. All 1,800 data schedules and supervised counts matched.

The CPU evaluation completed 2,048 main games and 64 smoke games without failures
or fallbacks. Each trained arm played 256 games against unchanged Kakuna:

| Condition | Wins | Score | Exploratory 95% block interval |
| --- | --- | --- | --- |
| Ordinary continuation | 122 | 47.66% | 41.80–53.52% |
| Joint continuation | 127 | 49.61% | 44.14–55.08% |

The joint-minus-ordinary difference was +1.95 percentage points
(−2.73 to +6.25). Neither improvement nor significant regression was established.
The joint checkpoint is retained for development, with previous models preserved.
It is not a confirmed champion. Parent play used a different seed, so comparing
their old scores with these scores is not an exact paired parent comparison.

On matched basic-opponent games, unchanged / ordinary / joint scored
94.14 / 91.41 / 92.58% against SimpleHeuristics and 99.61 / 99.22 / 99.61% against
MaxDamage. These point estimates use a new play seed. They support basic
competence, not ladder strength or non-regression against Foul Play.

On the reused 32-game belief cohort, learned modal item accuracy rose from
10.63% to 14.21%, while existing training-pool inference remains at 50.10%.
Observed-outcome MSE fell from 0.5396 to 0.4763. Learned exact-set recall@32 remains
zero, compared with 2.97% for existing inference. These aggregate sampling reports
do not save per-game scores, so correlated fields cannot supply an independent
confidence interval. Lower Monte Carlo ECE alone is not evidence of better
inference. Outcome MSE still does not compare the native critic. Learned sampling
remains disconnected from search.

More optimization produced some auxiliary progress without a demonstrated play
gain. Native aggregate training loss stayed nearly flat, and the mean weighted
auxiliary shared gradient was 0.548% of primary. The tenfold head learning rate
does not ensure that the shared representation learned useful hidden-state
information. An offline CPU diagnostic now supports zeroing or permuting the
latent across different games while preserving POV and decision phase. Intact and
perturbed contexts use identical verified queries, with per-game scores. The
perturbations can be out of distribution and are not causal tests of playing
strength; they help distinguish latent information from revealed-fact priors.

The bounded pretrained attempt has not demonstrated improvement under this
recipe and budget. Much of its charged compute was reserved or failed time;
successful joint training totaled about 82 plus 111 minutes. This result is not
evidence of architectural saturation or proof that longer training cannot help.
The next neural experiment follows the independent-weight FOG fallback, with
mechanics-derived public action features, strong public-information teachers and
exact-team belief/outcome supervision. Its data path and short local trial must
pass before a new private GPU pilot. Both final confirmation cohorts remain unused.

## Shared-latent diagnostic, October 12

The CPU diagnostic completed in 74.82 minutes. All 54 exported file hashes,
three probability contracts and 16 smoke games passed, with no failures or
fallbacks. Both checkpoints stayed unchanged. No new GPU training or playing
strength benchmark ran.

The parent and continued joint checkpoints used identical inputs, revealed
facts and verified targets across 32 training and 32 development games. Each
split contributed 192 contexts and 64 correct-prefix queries per context.
The decoder was evaluated with intact latent conditioning, zero conditioning,
and the same perspective and game phase from a different game's latent.

On development inputs, continued outcome-head MSE was 0.4763, versus 0.5396
for the parent, 1.0037 with zero conditioning, and 2.0724 with permuted
conditioning. The outcome head uses useful state information in this diagnostic.
This remains an observed-outcome prediction test, separate from comparing the
native critic or proving that outcome supervision improves policy play.

The continued dedicated marginal item head scored 21.37% top1 accuracy, versus
11.04% for its parent; item NLL fell from 3.2825 to 3.0451. These direct
distribution scores differ from the earlier ancestral sample modes of 14.21%
and 10.63%. They must not be presented as another training gain. Permuting the
continued latent changes marginal item top1 to 20.35% and NLL to 3.0766; move
NLL changes from 3.0813 to 3.0833. Latent sensitivity is modest compared with
the gains across checkpoints. Species and revealed-fact priors may explain
much of the predictive structure, but this single perturbation does not prove
that explanation or establish a failed shared representation.

On identical queried fields, exact teacher prefixes improve move and EV
prediction but worsen item prediction relative to the dedicated marginal head:
item NLL rises from 3.0888 to 3.1598 and top1 falls from 18.90% to 14.63%.
An approximate conditional can perform worse despite having more information.
This indicates an unresolved training or conditioning issue; it does not
invalidate autoregressive factorization. Retain the AR design and track direct
marginal, correct-prefix and ancestral scores separately by field.

The development cohort contains eight four-game composition groups with 16
distinct teams. Paired uncertainty estimates resample entire groups and retain
field counts as weights, using 20,000 resamples. The continued-minus-parent
outcome MSE difference is -0.0633 (exploratory 95% interval -0.1052 to -0.0226),
and marginal item NLL difference is -0.2374 (-0.3471 to -0.1359). These intervals
are unadjusted, conditional on reused pools, and based on few groups. Permuted
donor contexts introduce further dependence across groups that those intervals
do not model. Perturbation estimates remain descriptive; zero context may also
be out of distribution. No final-cohort or playing-strength claim follows.

The continued joint checkpoint remains retained. The next neural recipe remains
the approved independently initialized FOG fallback. Its immediate prerequisite
is authoritative strong-teacher data: native fresh records preserve teams and
native action indices but do not contain FOG fragments or per-decision requests.
Validate deterministic engine replay or collect fresh live requests before
training; do not guess PP, trapping or disabled-move legality. Keep complete-team
labels and teacher-answer features outside player-visible policy inputs. Preserve
the native and original FOG baselines and validate a short local training trial
before freezing the bounded fallback GPU pilot.
