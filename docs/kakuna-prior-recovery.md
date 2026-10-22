# Diagnosing prior-conditioned belief sampling

The frozen-policy pilot reduced correct-prefix loss without improving ancestral
reconstruction over its donor prior. A lower teacher-forced loss therefore does
not establish that the proposal used by search is better.

`fog.scripts.audit_prior_prefixes` observes the existing production calls under
public-only, correct and generated prefixes. It records exhausted donor support,
target support, retrieval truncation and prefix compatibility with verified labels.
The observer receives labels only after the production conditional has returned;
it must leave logits, gradients, RNG, sampled teams and probability traces unchanged.
Move targets are unordered remaining-set membership events in this diagnostic,
not the ordered move NLL of the training objective.

A small development diagnostic found substantially more fallback queries with
correct prefixes than with generated prefixes. It also found matching sets lost
to the 128-donor cap. These are hypotheses to check over the full saved cohorts,
not a new held-out result. Most generated prefixes contradict the actual submitted
set; blindly teaching the original answer under those assumptions is not valid
conditional supervision and can conflict with move and EV constraints.

`DonorPrior(..., residual_mode="fallback_only")` provides a conservative comparison.
When compatible donors give positive mass to a legal query value, the decoder
uses their smoothed conditional exactly. When support is exhausted, it uses the
learned decoder under the same legality masks. Sampling, named queries and path
scoring share this rule. The default `"all"` preserves the preceding behavior.
The mode propagates through repeated and selected prior batches.

The gate prevents learned corrections from disturbing supported donors. It also
prevents the model from correcting a supported but wrong prior. More donors may
retain rare sets but may add inaccurate or obsolete sets. Neither change guarantees
better reconstruction, calibration or stronger decisions.

`fog.scripts.compare_prior_recovery` compares the original 128-donor prior and
native residual with a 512-donor prior, fallback-only native residuals at both caps,
and a fallback-only facts residual at 512. It also reproduces the existing complete-
set donor sampler. The comparison uses the same k=8 draws and seed as its saved
controls, checks their exact sampling metrics and numerical density agreement,
and reports paired whole connected-roster intervals. Both cohorts are consumed
development evidence; final teams remain untouched. Multiple exploratory
comparisons are not a confirmation test.

The private CPU capsule fits no weights and contains no original native-policy
weights. It checks local/Kaggle CPU probability agreement, the categorical schema,
sealed inputs, source hashes and complete result inventories. CUDA certification
of the new inference mode remains a separate requirement before CUDA use.
Standalone sampler timings include projection but exclude retrieval, policy and
search, so they cannot certify a combined five-second move deadline.

The next GPU recipe depends on the diagnostic result. The retained project goal
is stronger Gen9OU play with useful belief and value information. Better auxiliary
metrics alone do not satisfy that goal; a later fixed-search playing-strength
comparison must demonstrate the benefit.
