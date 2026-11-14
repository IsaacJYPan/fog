# Brief 07 evidence (smoke)

Stage 1 only. No model, value-loss, mechanical-feature, preview-training or play result is
implied by this directory. Checkpoints stay under ignored `runs/order_blind/`.

| Evidence | Current files |
| --- | --- |
| Original policy | `R0-metrics.csv`, `R0-slot-check-extended.json`, `R0-holdout.json`, `R0-execution.json` |
| Corrected rows, first seed | `R1-metrics.csv`, `R1-slot-check.json`, `R1-holdout.json` |
| Corrected rows, second seed | `R1-seed2-metrics.csv`, `R1-seed2-slot-check.json`, `R1-seed2-holdout.json` |
| Completed comparison and seed differences | `stage1-summary.json`, `stage1-complete.json`, `stage1-execution.json` |
| Final conversion and fixed game IDs | `converter-after-quality.json`, `fixed-split.json`, `row-count-changes-final.json` |
| Before/after order diagnostics | `rows-before-extended.json`, `rows-after.json`, `row-intervals.json`, `lead-order-before.json`, `lead-order-after.json` |
| Tests and compatibility | `tests-final.txt`, `tests-shared-imputer.txt`, `ruff-final.txt`, `checkpoint-compatibility.json`, `trained-checkpoint-compatibility.json`, `digests-summary.json` |
| CPU scheduling equivalence and interrupted attempt | `runtime-equivalence.json`, `R0-default-attempt-execution.json` |
| Conversion recovery | `conversion-resume.json` |

Use the extended R0 slot check for the comparison: it applies the same corrected diagnostic
as R1 and includes game-resampled intervals. The original `R0-slot-check.json` is retained
as the historical diagnostic output. `not_yet_moved` means no earlier move by the acting
species in the **stored history window**, not necessarily its first move in the game.
Policy metrics use eligible policy rows; value metrics include every retained row.

All intervals resample games with both perspectives together. The R0/R1 comparison uses
the shuffled policy columns because R0's stored holdout is leaky. Filtering also differs:
the corrected converter rejects stale action-to-snapshot identity mismatches.

Files containing `pre-guard`, `pre-empty-packing` or `partial-attempt` are superseded
conversion attempts, retained as negative evidence. The unsuffixed `row-count-changes.json`
is the first, superseded corpus comparison; use `row-count-changes-final.json` for final rows.
No R1 training used those superseded corpora. The converter recovery check predates the
final paired-imputation rebuild; its exact retained paths are recorded in its JSON.

The historical rollout hashes were not reproduced, including on unchanged source.
`digests-summary.json` distinguishes that failure from the matching before/after hashes
on this runtime and records the negative CPU-build and kernel probes. No reference hash
was replaced. Script copies ending in `-driver.txt` record the executed orchestration.
