# Replay blueprint on corrected rows

**The 20,000-game policy reached 47.32% holdout top-1 accuracy and 1.4578 negative log-likelihood at epoch 14.** The independently selected value checkpoint peaked at epoch 3 with Brier 0.2129 and expected calibration error 0.0083. It meets both numerical value thresholds on the trainer's fixed holdout. The policy checkpoint's value estimates deteriorated over the later epochs.

| Selection | Epoch | Holdout NLL | Top-1 | Brier | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Policy | 14 | 1.4578 | 47.32% | 0.2448 | 0.1336 |
| Value | 3 | 1.5871 | 42.32% | 0.2129 | 0.0083 |

The 989,494 training rows come from 20,000 games. The fixed 2,000-game holdout has 103,944 rows. Filtering acting-side validation compositions removed 36,563 candidate rows, including 28 whole games. Holdout move labels occupy the four slots in proportions of 25.09%, 24.83%, 24.82% and 25.25%. The stored outcome sign agrees with the protocol winner on every holdout row.

Training finished all 18 scheduled epochs. Calibration and continuation consumed **1.97 Kaggle GPU-hours** together. The account showed **24.47 hours remaining** afterward. Both exported checkpoints match their copies in the final optimizer archive tensor for tensor. Their SHA-256 values and the 19 metric rows, including initialization, are in [the training summary](13-artifacts/G20-training.json) and [the epoch measurements](13-artifacts/G20-metrics.csv).

**Gate G2 passed for G20.** The greedy policy won 730/1,000 against SimpleHeuristics (passed, threshold 65%) and 921/1,000 against A3 (passed, threshold above 55%). The separately selected value checkpoint achieved ECE 0.00832 (passed, threshold below 0.05) and Brier 0.21286 (passed, threshold below 0.24).

| Opponent | Greedy | Temperature 0.5 |
| --- | --- | --- |
| SimpleHeuristics | 730/1,000 (73.0%, 70.2–75.7%) | 637/1,000 (63.7%, 60.7–66.6%) |
| MaxDamage | 918/1,000 (91.8%, 89.9–93.3%) | 892/1,000 (89.2%, 87.1–91.0%) |
| Retained imitation policy (A3) | 921/1,000 (92.1%, 90.3–93.6%) | 900/1,000 (90.0%, 88.0–91.7%) |

The short lead-training stage raised holdout action NLL by 0.02160, exceeding the 0.005 retention threshold. It raised the 400-game SimpleHeuristics score by 7.5 percentage points (paired 95% interval 2.5 to 12.8). All six validation rows used the original policy checkpoint. Each cell used the same 1,000 four-way team-and-side-matched games across 141 held-out teams, seed 0, with a 200-turn limit. The table gives wins per scheduled game, decisive win rates and 95% Wilson intervals. Across the six cells, 0 games reached the limit and no simulator failures occurred.

The larger training set uses the converter with the replay/live parity fixes. The 20k-to-100k data comparison is pending. [Full stage results](13-artifacts/G20-stage.json)
