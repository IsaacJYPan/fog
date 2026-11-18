# Replay blueprint on corrected rows

**The 20,000-game policy reached 47.32% holdout top-1 accuracy and 1.4578 negative log-likelihood at epoch 14.** The independently selected value checkpoint peaked at epoch 3 with Brier 0.2129 and expected calibration error 0.0083. It meets both numerical value thresholds on the trainer's fixed holdout. The policy checkpoint's value estimates deteriorated over the later epochs.

| Selection | Epoch | Holdout NLL | Top-1 | Brier | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Policy | 14 | 1.4578 | 47.32% | 0.2448 | 0.1336 |
| Value | 3 | 1.5871 | 42.32% | 0.2129 | 0.0083 |

The 989,494 training rows come from 20,000 games. The fixed 2,000-game holdout has 103,944 rows. Filtering acting-side validation compositions removed 36,563 candidate rows, including 28 whole games. Holdout move labels occupy the four slots in proportions of 25.09%, 24.83%, 24.82% and 25.25%. The stored outcome sign agrees with the protocol winner on every holdout row.

Training finished all 18 scheduled epochs. Calibration and continuation consumed **1.97 Kaggle GPU-hours** together. The account showed **24.47 hours remaining** afterward. Both exported checkpoints match their copies in the final optimizer archive tensor for tensor. Their SHA-256 values and the 19 metric rows, including initialization, are in [the training summary](13-artifacts/G20-training.json) and [the epoch measurements](13-artifacts/G20-metrics.csv).

The G2 play clauses await the lead comparison and validation matches against SimpleHeuristics and A3. The larger training set and the 20k-to-100k data comparison follow publication of the G20 play results. The larger set uses the converter with the replay/live parity fixes.
