# Brief 09: order-blind model (stages 2–4 complete)

**Answer.** The order-blind model passes perturbed-weight permutation checks and improves shuffled NLL/top-1 from R1’s 1.9253/26.64% to R2’s 1.8799/28.24%. Value still fails the constant-0.5 Brier reference. Mechanics improve moves seen 1–19 times (NLL 2.0192→1.8193), but show no unseen-move gain (130 rows). R4’s play scores were 38.50%/77.00% against SimpleHeuristics/MaxDamage versus R1’s 14.75%/39.75%. Leads remain pending. All results are smoke.

CPU: AMD EPYC 9V74, four-core quota, five logical CPUs; PyTorch 2.13.0+cu130, no CUDA.
Pinned Showdown `e440c4a`. Sources: R1 `70cc5ed`, R2 `c7007c8` (model `b824430`),
R3 `69b9d19`, both R4 seeds `4066edc`. Branch `work/order-blind-model` from `70cc5ed`.

22,000 games, zero quarantines, 1,126,792 rows; fixed training 1,000 games/52,955 rows,
holdout 2,000 games/103,944 rows (103,933 policy-valid). Every run uses eight epochs, batch
256, cosine 3e-4→3e-5, no early stop. Seeds 20261201 and, for R4b, 20261202.
R1 reproduced the 1.9239 reference within 0.0016 NLL.

## Policy at best policy epoch

Each cell is shuffled NLL / top-1 %. All 95% intervals resample whole games, keeping both
perspectives; full intervals are in each `R*-best-policy-slot-check.json`.
“Unmoved” means no earlier move in retained history, not the entire game.

| Run / epoch | All | Move | Unmoved |
| --- | --- | --- | --- |
| R1 / 8 | 1.9253 / 26.64 | 1.7664 / 29.16 | 1.6697 / 33.46 |
| R2 / 8 | 1.8799 / 28.24 | 1.7280 / 30.26 | 1.6281 / 33.98 |
| R3 / 8 | 1.8806 / 28.51 | 1.7321 / 30.42 | 1.6332 / 34.46 |
| R4 / 7 | 1.8041 / 33.25 | 1.6804 / 36.36 | 1.5554 / 40.84 |
| R4b / 6 | 1.8139 / 32.80 | 1.6776 / 36.66 | 1.5763 / 40.51 |

R2/R3/both R4 seeds match stored/shuffled NLL and top-1 to four decimals. R2’s all-row
gain exceeds twice the historical R1 seed gap. R4’s all/move/unmoved gains exceed twice
its seed gap (`stage2-summary.json`, `stage4-summary.json`).

## Value selection

| Run / checkpoint / epoch | Brier [95%] | ECE [95%] |
| --- | --- | --- |
| R1 / policy / 8 | 0.4038 [0.3887, 0.4176] | 0.3808 [0.3653, 0.3956] |
| R1 / value / 0 | 0.2533 [0.2532, 0.2534] | 0.0573 [0.0567, 0.0579] |
| R2 / policy / 8 | 0.3978 [0.3839, 0.4117] | 0.3744 [0.3596, 0.3893] |
| R2 / value / 0 | 0.2500 [0.2500, 0.2500] | 0.0015 [0.0009, 0.0021] |
| R3 / policy / 8 | 0.4065 [0.3928, 0.4201] | 0.3841 [0.3702, 0.3981] |
| R3 / value / 0 | 0.2500 [0.2500, 0.2500] | 0.0015 [0.0009, 0.0021] |
| R4 / policy / 7 | 0.4017 [0.3877, 0.4162] | 0.3731 [0.3586, 0.3880] |
| R4 / value / 0 | 0.2567 [0.2566, 0.2568] | 0.0819 [0.0813, 0.0825] |
| R4b / policy / 6 | 0.3878 [0.3731, 0.4020] | 0.3601 [0.3449, 0.3747] |
| R4b / value / 0 | 0.2707 [0.2705, 0.2708] | 0.1438 [0.1432, 0.1444] |

Brier ≥0.25 carries no useful outcome information here. Native independent value selection
includes epoch 0; R1/R2 epoch-zero checkpoints were reconstructed and tensor-verified.
For R3 policy: p<0.1 share 39.24%, observed wins 44.51%; p>0.9 share 41.12%, wins 56.46%.
R3 value selection has no extreme predictions. All extreme statistics/intervals are in both checkpoint JSONs (decision-weighted).

## Move frequency and play

Move groups count policy-valid training labels; ordinary/Tera share identity. Cells are
NLL / top-1 %; game intervals: `rare-moves.json`. Gains for 1–19 and 20+ exceed twice
their R4 seed gaps. Unseen-move NLL worsens; its 130-row intervals are broad.

| Training count / rows | R3 | R4 | R4b |
| --- | --- | --- | --- |
| never / 130 | 1.8840 / 25.38 | 2.0163 / 15.38 | 2.0197 / 22.31 |
| 1_to_19 / 2171 | 2.0192 / 11.93 | 1.8193 / 29.25 | 1.7940 / 31.74 |
| 20_or_more / 72552 | 1.7233 / 30.98 | 1.6757 / 36.62 | 1.6735 / 36.84 |

Greedy play uses brief 03’s `pool_b`, 400 games/cell, shared matched seed 0, both teams
and sides crossed. Timing overlaps R4b training; it does not measure isolated throughput.
Only primary-seed best-policy checkpoints play; pool B is not held out.

| Policy / opponent | W–L / draws / unresolved / failures | Score and interval |
| --- | --- | --- |
| R1 vs simpleheuristics | 59–341 / 0 / 0 / 0 | 14.75 [11.00, 18.50]% block bootstrap |
| R1 vs maxdamage | 159–241 / 0 / 0 / 0 | 39.75 [34.75, 44.75]% block bootstrap |
| R4 vs simpleheuristics | 154–246 / 0 / 0 / 0 | 38.50 [33.75, 43.25]% block bootstrap |
| R4 vs maxdamage | 308–92 / 0 / 0 / 0 | 77.00 [72.75, 81.50]% block bootstrap |

Primary score treats genuine draws as 0.5 and truncations as unresolved; full native
scores/action statistics are in `play.json`; block intervals and paired differences are in
`play-block-intervals.json`. No strength claim.

## Defects, implementation and limits

Two dense-label attempts were rejected: 84/500 games disagreed across views, then a narrow
form fix missed other forms and its R3 run stopped after two epochs. Final public-only
base-species matching passes all 22,000 games; old arrays match in all 3,000 fixed-split games.
Initial conversion resumed 9,227 completed games after interruption.

All new behavior defaults off. Shared move pooling removes arbitrary order; public history
keeps kind and identity (`hist_opp_action` is a public global move ID). Value targets never
enter encoders. Mechanics use generated table facts and exactly eleven named candidate
columns; excluded columns, including NaNs, cannot affect outputs or gradients. Belief
decoders and their slot-indexed queries are unchanged.

All parameters are perturbed before four permutation checks on 64 synthetic, 64 holdout and
64 live observations; key-off move/bench controls fail as required. Mechanical-on tests and
89 affected tests pass; Ruff is clean. The old checkpoint was strictly loaded with identical
outputs at stage 3. Final compatibility and all-off digests will repeat after leads.

Evidence: all commands, metrics and checks are under `09-artifacts/`.

**Not verified:** stage 5, final all-off regression checks, playing-strength or phase gates.
No search, PPO or belief source changed.
