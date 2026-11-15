# Brief 09: order-blind model (stage 2 in progress)

**Answer.** The order-blind representation passes the required permutation test on 64 fixed-holdout
and 64 live Showdown observations with every parameter perturbed first. Both required negative
controls fail with the key off. R1 is training; R2 has not started, so no accuracy or value
comparison is available. Value targets, mechanical features, rare-move/play comparisons and leads
are not implemented. Everything measured here is smoke.

The branch is `work/order-blind-model`, from `70cc5ed`; the implementation is `b824430`.
CPU only: AMD EPYC 9V74, four-core quota (five logical CPUs visible), PyTorch 2.13.0+cu130,
no available GPU. Pinned Showdown is `e440c4a`. Runtime details are in `09-artifacts/runtime.json`.

## Data and checks

| Check | Measured result | Evidence in `09-artifacts/` |
| --- | --- | --- |
| Corrected conversion | 22,000 games, zero quarantines, 1,126,792 rows | `conversion-quality.json` |
| Fixed split | Same 2,000 holdout IDs / 103,944 rows; 1,000 training games / 52,955 rows | `data-counts.json`, `preparation-execution.json` |
| Permutations | All four required cases, plus other moves on both sides, pass at 1e-5; key-off move/bench controls fail | `permutations.json`, `tests-stage2-acceptance.txt` |
| Worst holdout / live log-probability error | 7.153e-7 / 7.153e-7 | `permutations-summary.json` |
| Focused tests | 10 acceptance tests pass on synthetic, holdout and live data; 40 related model/config tests pass | `tests-stage2-acceptance.txt`, `tests-stage2-development-final.txt` (the latter predates the available holdout and has three explicit skips) |
| Compatibility | Original checkpoint loads strictly; log probabilities and values bit-identical on 64 live observations | `original-checkpoint.json`, `compatibility-stage2.json` |
| New key off | Sequential / batched / MaxDamage digests equal before and after on this CPU | `digests-summary.json` |
| Ruff | Clean | `ruff-stage2.txt` |

The corpus conversion was interrupted while correcting the isolated baseline launch; all 9,227
completed game checkpoints were reused. `preparation-restart.json` records the interruption. The
completed preparation command and fixed baseline command are retained in the execution evidence.

## Implementation and defects found

`model.order_blind_entities` defaults to false. Enabled, a shared per-move network reads the move
embedding, PP and known flag; symmetric pooling replaces ordered move fields. Candidates reuse
that network. Creature tokens lose position tags and normalized numeric IDs; history keeps the
action kind and existing identity sidecars. The shared control token also needed symmetric legality
counts instead of ordered bits and active indices. Optional flattened global input is pooled by
side. Unsupported ordered architectures are rejected explicitly.

`hist_opp_action` stores the public move-table identity (`events.py`), not an opponent-private
move-slot index, and is unchanged. Belief heads and their slot-indexed queries are unchanged;
policy and value do not read their predictions. No search or PPO source changed.

## Not verified

R1 and R2 training comparisons are pending. Stages 3–5, second-seed R4, rare-move and play
comparisons are not run. No phase gate or strength claim is complete.
