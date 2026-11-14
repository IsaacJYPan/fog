# Replay blueprint GPU scale smoke — 2026-11-14

**Answer (in progress).** CPU preparation and the two default-off options are implemented;
71 affected tests pass and lint is clean. The 22,000-game set is verified: 20,000 training games / 1,023,525 rows and the fixed
2,000-game holdout / 104,037 rows. The 112,000-game conversion is running. No GPU job has been submitted yet, so GPU throughput, the new learning
curve, P2.4's decision branch, the G100 value reading and new play rates are unmeasured. No gate
is claimed. This checkpoint records work in progress, not completion of brief 05.

| Run (all smoke) | Policy epoch | Holdout top-1 | Policy NLL | Value Brier | Value ECE |
| --- | --- | --- | --- | --- | --- |
| B, historical CPU | 8 | 0.440802 | 1.579018 | 0.344501 | 0.298756 |
| G3 | pending | — | — | — | — |
| G20 | pending | — | — | — | — |
| G100 | pending | — | — | — | — |

| Run/checkpoint (smoke) | Train value MSE | Holdout value MSE | Early/middle/late Brier | Early/middle/late ECE |
| --- | --- | --- | --- | --- |
| B, policy epoch 8 | 0.272015 | 1.378004 | 0.343345 / 0.363723 / 0.327324 | 0.278211 / 0.328123 / 0.291107 |
| B, value epoch 4 | 0.746103 | 0.956552 | unavailable | unavailable |
| G3 / G20 / G100 | pending | pending | pending | pending |

B's ten-bin reliability table and extremes are in `05-artifacts/B-reference.json` and
`B-holdout-best.json`: 56.2588% of rows have predictions below 0.1 or above 0.9.
The historical best-value weights/report are absent from this checkout; only its per-epoch
MSE is available. These are retained brief 03 measurements, not new executions.

Implementation: `--prepare-only` imports, converts and splits without requiring CUDA; removing
the flag skips completed data stages. `bc.keep_best_value=true` independently retains
`bc_best_value.pt`, includes initialization as epoch zero, and restores its selection and bytes
from completed epoch state. It changes no loss, schedule or update. Unit tests check recovery
and identical metrics and final parameters with selection enabled and disabled. Code commit:
`bd6db4161cb3d6d563ae4a98f9415771601d4d8c`.

Validation: `05-artifacts/tests.txt`, `lint.txt`, `validation-attempts.json`. The command sandbox
hangs during the dependency's asynchronous shutdown; the same real pipeline passes outside
it. Earlier failed attempts are recorded separately. No full suite was run.

GPU budget: **0 of 20 GPU-hours used**, no jobs submitted; `05-artifacts/gpu-hours.json`.
Official client authentication and job-list access succeeded. Prepared data and every job will
be private; account identity stays outside Git. `gpu_job.py` is the job template, not evidence
of a completed GPU execution. No new checkpoint exists (`checkpoints.json`).

Stage 0 first set: import 92.066 s, conversion 1,000.359 s, split 3.294 s; these CPU walls
overlapped setup/tests and the larger import. All 22,000 games accepted, none quarantined.
The 1,000/3,000 subsets reproduce 52,983/156,597 rows. Repeating the prepare command skips
all three completed stages (`prepare-22000-skip.txt`).

Next: upload the verified first set privately; run G3 and its deliberate cross-job stop/continue; choose batch size using
the specified 0.031 NLL comparison; finish G20, its value tables and CPU play smoke before G100.
Choose G100 epochs only from measured GPU rate and remaining budget, reserving two hours.

Not verified: hosted install/training/validation speed or memory, cross-job recovery, the
20k/100k learning curve, G100 value learning/sign/leakage checks, durable GPU checkpoints and
new Showdown play results. P2.1–P2.5 remain unchecked.
