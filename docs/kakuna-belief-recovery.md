# Recover a useful shared belief signal

The completed native pilot preserved observed play but demonstrated no useful
belief improvement. This next attempt retains the real Kakuna checkpoint 34,
its actor/critic architecture and FOG's fact-set autoregressive decoder. It
does not resume independent FOG policy training or deploy learned beliefs to search.

Two local measurements identify actionable weaknesses. They use eight fixed
training contexts; they are mechanism diagnostics, not generalization evidence.

* A 1e-4 native step moves the latent about 100 times farther in RMS than 1e-6,
  while passing the same sampled policy/value bounds. A 1e-3 step fails those
  bounds. There is no frozen-weight-cache bug in upstream SigmaReparam.
* With the nominal outcome loss weight of 0.1, its actual native gradient norm
  was 0.240, versus about 0.070 for the two combined belief objectives. A scalar
  loss weight did not ensure deliberate shared belief learning.

The initial donor bank also contains only 3,000 sets from 500 teams, representing
314 species. Only 4.17% of the 600 development sets are exactly present in that
bank. The existing candidate sampler still outperforms the learned chain, but
neither this small bank nor its encoded-set coverage constitutes a good joint
prior for every unseen team.

## Corrected shared-learning pilot

Keep the complete-history feature cache and exact submitted-team labels. Train
only the last native block and final normalization; actor, critic and all lower
weights/buffers stay frozen. Use a native learning rate of 1e-4 as a bounded
pilot hypothesis, retaining the existing legal-action KL, shaped-value consistency,
training-only sentinel, update rejection and full-state recovery checks.

Accumulate belief and weighted outcome gradients over the effective batch, then
cap the outcome contribution on the actual trainable native parameters at 25%
of the belief gradient norm. Retention gradients and all head gradients remain
unchanged. This enforces the intended priority in parameter space rather than
assuming loss magnitudes or latent-gradient norms imply a backbone-gradient ratio.
Record the pre/post-cap norms, cap scale and retention-gradient norm.

Before retention gradients and Adam are applied, this cap prevents the outcome
term from cancelling more than 25% of the belief gradient's projection onto
itself. It does not guarantee a loss decrease after momentum, preconditioning or
nonlinear updates, nor a win-rate bound. Those still require measurement.

Compare frozen features with adapted features, using identical fitted-head
initialization, query/data schedules, head optimizer budgets and outcome targets.
The combined change tests meaningful, belief-led shared adaptation; it does not
identify the separate causal effects of learning rate and gradient balancing.
An actual CUDA numerical/resume gate precedes main updates. Budget at most two
GPU hours for this pilot; incompleteness never counts as a completed matched pair.

## Independent prior-data comparison

Build a bounded bank from the pinned High Ladder May 2026 Gen9OU archive. Do not
mix it with its General Ladder parent and count duplicate evidence twice. Check
current legality with the pinned Showdown validator, deduplicate encoded teams,
and exclude every development/final species roster before allowing donors in.
Target 20,000 teams, require at least 5,000, and inspect at most 40,000 candidates.
Retain archive/member provenance and protected-roster exclusion evidence.

Archive completions are prior evidence. They are not verified historical labels
and cannot become exact-team supervision for a historical replay. Their dates
remain explicitly synthetic availability ordering. The builder rejects incomplete
or unrepresented modeled sets rather than inventing their missing categories.

Run this construction and comparison on Kaggle CPU. Evaluate the original and
expanded priors with identical k=8 draws and the same frozen public contexts.
Require exact reproduction of the previous baseline metrics, with the known-item
mask matching the actual decoder. Truth is scoring-only. Save per-game sufficient
statistics for connected-roster resampling; 32 reused games are eight
composition groups, not thousands of independent fields.

## Decisions and eventual contribution

These two jobs address different hypotheses and can run independently. Neither
changes protected checkpoints, consumes final confirmation games, or establishes
improved playing strength. Collect full artifacts and review before changing a
recipe or scaling its compute.

If shared adaptation still offers no useful reconstruction gain, inspect context
sensitivity and the static/conditional prior mismatch rather than merely extending
training. If the larger prior helps, test a prior-informed residual autoregressive
decoder against the same stronger prior. It must preserve support for unfamiliar
sets and keep its probabilities distinct from a calibrated posterior claim.

The eventual contribution still requires better decisions against unchanged
Kakuna, through measured policy learning, search or endgame teaching. The separate
search bridge's hard deadline must be repaired before online search is enabled.
Do not interpret a representational or imputation improvement alone as a stronger bot.

## Observed failure and retention replay

The stronger-step pilot stopped after 101 accepted native updates. A newly drawn
training context already exceeded the original-policy KL limits before the next
step. Rejecting that proposal correctly restored the pre-step weights, but could
not repair drift left by earlier updates on other contexts. The fixed sentinel
remained within its limits. This was a limitation of sampled coverage, not a
failed checkpoint resume or measured playing-strength regression.

Local reproduction recovered the same mean KL of 0.002886 and maximum 0.09105.
Contracting total upper-weight displacement halfway toward protected Kakuna34
passed the unchanged limits on the failing batch, sentinel and 64 additional
training contexts. Retention gradients were a median 43 times the belief-gradient
norm during the stopped pilot. More updates with that objective do not by
themselves resolve either problem.

The next bounded pilot uses a deterministic 512-context training-only retention
bank, including the offending public context. Each update checks its current
batch, fixed sentinel and 32 rotating bank contexts; all bank groups are checked
every 50 updates and before export. Rotating public contexts also contribute a
retention objective. Private labels never enter the actor/critic readouts, and
development contexts do not decide acceptance.

A dead zone leaves per-context/action changes below one quarter of the original
mean KL/value budgets unpenalized. Hard acceptance bounds remain unchanged. If
earlier updates left a newly detected violation, backtrack the total displacement
toward immutable original weights and check every retention group. Copy and test
the original exactly as the last bounded fallback. Reset native Adam moments
after a successful repair, reduce the native learning rate, and record the repair;
head weights, head Adam, query exposure and random streams remain intact. Failure
to validate the original, excessive repairs, or incomplete matched exposure stops
the run. This is an explicit changed optimization algorithm, not a global
preservation theorem. A finite training bank can miss unseen-history drift.

Use native LR 2e-5 and keep the 25% outcome-to-belief native-gradient cap. Reuse
the completed warm heads already paired with original Kakuna, disclose their
historical 1,000 updates, and perform 32 common warm updates plus 1,000 matched
updates per condition. Require an actual CUDA regression of the stopped case,
complete-state resume including repair progress, and the probability contracts.
Keep the two-hour GPU limit and separate CPU playing-strength evaluation.

The independent expanded-prior audit admitted 20,000 legal, supported training
teams and improved development item modal accuracy from 48.26% to 52.25%, and
encoded-set recall@8 from 1.22% to 3.85%. Calibration did not improve uniformly.
These are eight reused development roster groups, with reconstructed archive
evidence and no downstream-play measurement. Keep this bank separate from the
matched shared-learning pilot. A prior-informed AR proposal and fixed-search
comparison are subsequent hypotheses, not demonstrated strength improvements.
