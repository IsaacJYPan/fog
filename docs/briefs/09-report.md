# Brief 09: order-blind model — complete

**Answer.** The order-blind model passes perturbed-weight permutation checks and improves shuffled NLL/top-1 from R1’s 1.9253/26.64% to R2’s 1.8799/28.24%. Value still fails the constant-0.5 Brier reference. Mechanics improve moves seen 1–19 times (NLL 2.0192→1.8193), with no unseen-move gain. R4’s play scores were 38.50%/77.00% against SimpleHeuristics/MaxDamage versus R1’s 14.75%/39.75%. Lead top-1 is 34.90% versus 16.67% chance and 23.93% for training frequency. All stages are complete; all results are smoke.

AMD EPYC 9V74, four-core quota; PyTorch 2.13.0+cu130, CPU only. Showdown `e440c4a`.
Sources: R1 `70cc5ed`, R2 `c7007c8`, R3 `69b9d19`, R4/R4b `4066edc`, leads `c87b519`.
22,000 games, zero quarantines, 1,126,792 decisions. Fixed train: 1,000 games/52,955 rows;
holdout: 2,000/103,944 (103,933 policy-valid). Eight epochs, batch 256, cosine 3e-4→3e-5,
no early stop. Seeds 20261201 and 20261202 (b). R1 passes the reference 0.01-NLL guard.

## Policy and value

Policy: shuffled NLL / top-1 %. Unmoved refers to retained history. Intervals resample
whole games, both sides together (`R*-best-policy-slot-check.json`).

| Run / epoch | All | Move | Unmoved |
| --- | --- | --- | --- |
| R1 / 8 | 1.9253 / 26.64 | 1.7664 / 29.16 | 1.6697 / 33.46 |
| R2 / 8 | 1.8799 / 28.24 | 1.7280 / 30.26 | 1.6281 / 33.98 |
| R3 / 8 | 1.8806 / 28.51 | 1.7321 / 30.42 | 1.6332 / 34.46 |
| R4 / 7 | 1.8041 / 33.25 | 1.6804 / 36.36 | 1.5554 / 40.84 |
| R4b / 6 | 1.8139 / 32.80 | 1.6776 / 36.66 | 1.5763 / 40.51 |

R2/R3/both R4 seeds match stored/shuffled metrics to four decimals. R2’s all-row gains
exceed twice historical R1 seed gaps; R4’s all/move/unmoved gains exceed twice its seed gaps
(`stage2-summary.json`, `stage4-summary.json`).

| Run / selector / epoch | Brier [95%] | ECE [95%] |
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

Brier ≥0.25 carries no useful outcome information here. Value selection includes epoch 0;
R1/R2 initial weights were reconstructed and verified. Both checkpoint JSONs contain
calibration/extreme-probability shares, conditional wins and game intervals.
R3 policy: p<0.1 share 39.24%, wins 44.51%; p>0.9 share 41.12%, wins 56.46%.
Its best-value checkpoint has no extreme predictions.

## Move frequency and play

Groups count training labels, merging ordinary/Tera identity; cells: NLL / top-1 %.

| Training count / rows | R3 | R4 | R4b |
| --- | --- | --- | --- |
| never / 130 | 1.8840 / 25.38 | 2.0163 / 15.38 | 2.0197 / 22.31 |
| 1_to_19 / 2171 | 2.0192 / 11.93 | 1.8193 / 29.25 | 1.7940 / 31.74 |
| 20_or_more / 72552 | 1.7233 / 30.98 | 1.6757 / 36.62 | 1.6735 / 36.84 |

The 1–19 and 20+ gains exceed twice their R4 seed gaps. Unseen NLL worsens; its 130-row
game intervals are broad (`rare-moves.json`).

Greedy play: brief 03’s `pool_b`, 400 games/cell, shared seed-0 matched teams/sides.
Only primary-seed best-policy weights play. No draws, truncations or failures.

| Policy / opponent | W–L | Score % [95% matched-block bootstrap] |
| --- | --- | --- |
| R1 vs simpleheuristics | 59–341 | 14.75 [11.00, 18.50] |
| R1 vs maxdamage | 159–241 | 39.75 [34.75, 44.75] |
| R4 vs simpleheuristics | 154–246 | 38.50 [33.75, 43.25] |
| R4 vs maxdamage | 308–92 | 77.00 [72.75, 81.50] |

`play.json` retains action statistics; `play-block-intervals.json` includes paired differences.
Pool B is not held out. Play overlapped training; no isolated timing claim.

## Leads

44,000 previews; fixed train/holdout have 1,985/3,986 valid labels. Unresolved Illusion
identities and incomplete previews are excluded (`preview-data-counts.json`). Position shares
remain near one sixth (`preview-holdout.json`). Inputs contain own private sheets and public
preview only, before send-out. R5 warms R4 and trains eight epochs; no play.

| Preview checkpoint / epoch | Top-1 % [95% game bootstrap] |
| --- | --- |
| R5 / 8 | 34.90 [33.43, 36.44] |
| R5b / 8 | 34.95 [33.51, 36.44] |
| Chance | 16.67 |
| Training-frequency baseline | 23.93 [22.58, 25.17] |

Frequency chooses the most-led training species present, breaking ties alphabetically.
Twice the lead seed gap is 0.10 points; `stage5-summary.json` qualifies comparisons.
Initial pooled-preview metrics and paired baseline intervals are also recorded.

## Defects and checks

Two label attempts were rejected: 84/500 cross-view mismatches, then missed forms and a
discarded two-epoch R3. Final public/base-species labels pass all 22,000 coherence checks.
Preview conversion preserves every old array in all 3,000 fixed-split games. Initial conversion
resumed 9,227 games; preview validation resumed after correcting an empty-slot assertion.

All new behavior defaults off. History keeps identity (`hist_opp_action` is a public move
ID); belief decoders/queries are unchanged. Mechanics use eleven allowed candidate columns.
Noisy-weight tests cover four permutations on 64 holdout/live/synthetic rows, key-off
negative controls, preview isolation and exact resume.
166 affected tests and Ruff pass. Old checkpoint loads strictly with identical policy/value
on 64 live rows; all three original same-machine digests match (`compatibility-final.json`,
`digests-final-summary.json`). All commands/results are under `09-artifacts/`.

**Not verified:** held-out playing strength or phase gates. P2.3/G2 remain open.
No search, PPO or belief source changed.
