# Brief 09: order-blind model (stages 2–3 complete)

**Answer.** The order-blind model passes the perturbed-weight permutation test and matches stored/shuffled results to four decimals. R2 shuffled NLL/top-1 is 1.8799/28.24% versus R1's 1.9253/26.64%, beyond twice the historical R1 seed gaps. The value head still carries no useful outcome information: R3 best-value Brier 0.2500, epoch 0; its best-policy NLL/top-1 is 1.8806/28.51%. Stages 2–3 are complete; move features, rare-move/play comparisons and leads are pending. All results are smoke.

CPU: AMD EPYC 9V74, four-core quota, five logical CPUs visible; PyTorch 2.13.0+cu130,
CUDA unavailable. Pinned Showdown `e440c4a`. Branch `work/order-blind-model` from `70cc5ed`;
R1 source `70cc5ed`, R2 `c7007c8` (model `b824430`), R3 `69b9d19`.

## Fixed data and recipe

22,000 corrected games, zero quarantines, 1,126,792 rows. The exact fixed holdout has
2,000 games / 103,944 rows (103,933 policy-valid); training has 1,000 games / 52,955 rows.
All three runs use eight CPU epochs, batch 256, cosine 3e-4 to 3e-5, no early stop, seed 20261201.
R1 reproduced the 1.9239 reference within 0.0016 NLL. The initial conversion resumed 9,227
per-game checkpoints after a dispatcher interruption.

## Policy at the best policy epoch

Shuffled-slot numbers below; 95% intervals resample 2,000 whole games, keeping both sides.
“Unmoved” means no earlier move in the retained history, not the entire game.

| Run / group | NLL [95%] | Top-1 % [95%] |
| --- | --- | --- |
| R1 e8 / all | 1.9253 [1.9189, 1.9316] | 26.64 [26.34, 26.95] |
| R1 e8 / move | 1.7664 [1.7607, 1.7720] | 29.16 [28.80, 29.53] |
| R1 e8 / unmoved | 1.6697 [1.6599, 1.6805] | 33.46 [32.79, 34.10] |
| R2 e8 / all | 1.8799 [1.8735, 1.8860] | 28.24 [27.92, 28.57] |
| R2 e8 / move | 1.7280 [1.7223, 1.7334] | 30.26 [29.90, 30.65] |
| R2 e8 / unmoved | 1.6281 [1.6184, 1.6386] | 33.98 [33.31, 34.64] |
| R3 e8 / all | 1.8806 [1.8744, 1.8869] | 28.51 [28.16, 28.84] |
| R3 e8 / move | 1.7321 [1.7264, 1.7379] | 30.42 [30.01, 30.81] |
| R3 e8 / unmoved | 1.6332 [1.6232, 1.6441] | 34.46 [33.80, 35.10] |

R2 and R3 stored and shuffled NLL/top-1 match to four decimals in all three groups.
Historical R1 seed thresholds and differences are in `stage2-summary.json`; R1 was rerun once.

## Value checkpoint selection

| Run / selector / epoch | Brier [95%] | ECE [95%] |
| --- | --- | --- |
| R1 / policy / 8 | 0.4038 [0.3887, 0.4176] | 0.3808 [0.3653, 0.3956] |
| R1 / value / 0 | 0.2533 [0.2532, 0.2534] | 0.0573 [0.0567, 0.0579] |
| R2 / policy / 8 | 0.3978 [0.3839, 0.4117] | 0.3744 [0.3596, 0.3893] |
| R2 / value / 0 | 0.2500 [0.2500, 0.2500] | 0.0015 [0.0009, 0.0021] |
| R3 / policy / 8 | 0.4065 [0.3928, 0.4201] | 0.3841 [0.3702, 0.3981] |
| R3 / value / 0 | 0.2500 [0.2500, 0.2500] | 0.0015 [0.0009, 0.0021] |

Brier at or above 0.25 carries no useful outcome information here. Value selection includes epoch 0. R1/R2 initial checkpoints were reconstructed and
verified against all initial tensor digests; R3 saves value and policy selections natively.

## Public terminal targets

R3 uses stable cross-entropy on the unsaturated win logit, plus coefficient-0.1 survival and
HP-difference auxiliary heads. Labels are training-only and never enter an encoder.
Per-epoch Brier, global-bin ECE and both auxiliary losses are in `R3-metrics.csv`.

Two label attempts were rejected: 84/500 games failed a cross-view check; the first form fix
then missed non-Ogerpon forms, so a two-epoch R3 run was discarded. Targets now use public
opponent views and unique base-species identities, rejecting unmatched forms. All 22,000 games
pass identity/coherence checks; old arrays are identical in the 3,000 fixed-split games.
Ninety-five affected tests and Ruff pass (`tests-stage3-general-forms.txt`).

Extreme predictions below use decision-weighted game outcomes; intervals resample games.
Each cell is prediction share % → observed win %. Empty groups are n/a.

| R3 checkpoint | p < 0.1 | p > 0.9 |
| --- | --- | --- |
| policy | 39.24 → 44.51 | 41.12 → 56.46 |
| value | 0.00 → n/a | 0.00 → n/a |

Exact extreme-frequency intervals are in `R3-best-{policy,value}-slot-check.json`.

## Implementation, checks and limits

`model.order_blind_entities` defaults off. Shared move vectors pool symmetrically; creature
positions and numeric IDs are removed, and history retains kind plus identity. Control legality
counts and optional global pooling are symmetric. Ordered architecture combinations are rejected.
`hist_opp_action` is a public move-table ID, not a private slot. Belief decoders and their
slot-indexed queries are unchanged; policy and value do not read their predictions.

Every parameter was perturbed before testing all four required permutations on 64 holdout,
64 live and 64 synthetic observations. Ten acceptance tests pass, including key-off negative
controls; 40 related model/config tests pass. Worst log-probability error is 7.153e-7.
Ruff is clean. The original checkpoint loads strictly with bit-identical policy/value on 64
live observations (`compatibility-stage3.json`). Three same-machine digests matched at stage 2;
they will be repeated after the remaining changes.

Evidence is under `09-artifacts/`: `data-counts.json`, `permutations.json`, test logs,
`compatibility-stage3.json`, `digests-summary.json`, per-run metrics/selection/slot-check files
and command execution records. The slot-check JSON contains every group interval.

**Not verified:** stages 4–5, R4 second seed, rare-move and play comparisons. No strength or
phase gate is claimed. No search, PPO or belief source changed.
