# Brief 09: order-blind model (stage 2 complete)

**Answer.** The order-blind model passes the perturbed-weight permutation test and gives identical stored/shuffled results to four decimals. R2 shuffled holdout NLL is 1.8799 versus R1's 1.9253; the 0.0453 difference exceeds twice the historical R1 seed gap. Top-1 is 28.24% versus 26.64%; its difference exceeds that metric's seed threshold. Neither value head beats constant 0.5. Stage 2 is complete; value targets, move features, play and leads are pending. All results are smoke.

CPU: AMD EPYC 9V74, four-core quota, five logical CPUs visible; PyTorch 2.13.0+cu130,
CUDA unavailable. Pinned Showdown `e440c4a`. Branch `work/order-blind-model` from `70cc5ed`;
R1 source `70cc5ed`, R2 source `c7007c8` (model implementation `b824430`).

## Fixed data and recipe

22,000 corrected games, zero quarantines, 1,126,792 rows. The exact fixed holdout has
2,000 games / 103,944 rows (103,933 policy-valid); training has 1,000 games / 52,955 rows.
Both runs use eight CPU epochs, batch 256, cosine 3e-4 to 3e-5, no early stop, seed 20261201.
R1 reproduced the 1.9239 reference within 0.0016 NLL. Conversion resumed 9,227 completed games
after an interrupted dispatcher correction; no results were discarded.

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

R2 stored and shuffled NLL/top-1 match to four decimals in all three groups.
The historical two-seed R1 thresholds and observed differences are in `stage2-summary.json`;
this session reran R1 once, as brief 09 requires.

## Value checkpoint selection

| Run / selector / epoch | Brier [95%] | ECE [95%] |
| --- | --- | --- |
| R1 / policy / 8 | 0.4038 [0.3887, 0.4176] | 0.3808 [0.3653, 0.3956] |
| R1 / value / 0 | 0.2533 [0.2532, 0.2534] | 0.0573 [0.0567, 0.0579] |
| R2 / policy / 8 | 0.3978 [0.3839, 0.4117] | 0.3744 [0.3596, 0.3893] |
| R2 / value / 0 | 0.2500 [0.2500, 0.2500] | 0.0015 [0.0009, 0.0021] |

Brier at or above 0.25 carries no useful outcome information here. Best value selection
includes the untrained epoch 0. Its checkpoint was reconstructed from the fixed seed and
verified against every initial tensor digest in the resumable trainer state.

## Implementation, checks and limits

`model.order_blind_entities` defaults off. Shared move vectors pool symmetrically; PP and known
flags travel with moves. Creature positions, party one-hots and numeric IDs are removed.
History retains action kind plus categorical identity. The control token uses symmetric action
counts; optional global input pools by side. Ordered architecture combinations are rejected.
`hist_opp_action` is a public move-table ID, not a private slot. Belief decoders and their
slot-indexed queries are unchanged; policy and value do not read their predictions.

Every parameter was perturbed before testing all four required permutations on 64 holdout,
64 live and 64 synthetic observations. Ten acceptance tests pass, including key-off negative
controls; 40 related model/config tests pass. Worst log-probability error is 7.153e-7.
Ruff is clean. An original checkpoint loads strictly and gives bit-identical policy/value on
64 live observations; all three same-machine new-key-off digests match.

Evidence is under `09-artifacts/`: `data-counts.json`, `permutations.json`, test logs,
`compatibility-stage2.json`, `digests-summary.json`, per-run metrics/selection/slot-check files
and command execution records. The slot-check JSON contains every group interval.

**Not verified:** stages 3–5, R4 second seed, rare-move and play comparisons. No strength or
phase gate is claimed. No search, PPO or belief source changed.
