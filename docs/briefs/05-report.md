# Replay blueprint GPU scale smoke — 2026-11-14

**Answer (in progress).** CPU preparation and the two default-off options are implemented;
71 affected tests pass and lint is clean. The 22,000-game set is verified: 20,000 training games / 1,023,525 rows and the fixed
2,000-game holdout / 104,037 rows. The 112,000-game conversion is running. The first hosted setup reached a Tesla T4 but failed before training because the dataset
archive was expanded by Kaggle. The corrected retry was capped without returning training output. An opaque archive and
graceful export deadline are tested; a private opaque-input pilot has been accepted. GPU throughput, the new learning
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

GPU budget: **20 GPU-hours maximum**; first setup used 0.038575 quota-hours, two attempts ended, **0.208297 quota-hours total**; `05-artifacts/gpu-hours.json`.
Official client authentication and job-list access succeeded. Prepared data and every job will
be private; account identity stays outside Git. `gpu_job.py` is the job template; the exact submitted pilot source and accepted response
are retained. No completed GPU execution is claimed. No new checkpoint exists (`checkpoints.json`).

Stage 0 first set: import 92.066 s, conversion 1,000.359 s, split 3.294 s; these CPU walls
overlapped setup/tests and the larger import. All 22,000 games accepted, none quarantined.
The 1,000/3,000 subsets reproduce 52,983/156,597 rows. Repeating the prepare command skips
all three completed stages (`prepare-22000-skip.txt`).

The private first dataset upload succeeded. Its separate read endpoints return `datasets.get`
permission denied, but the owned dataset listing confirms privacy
(`prepare-22000-private-check.json`); full kernel metadata confirms the job is private. Private GPU submission accepted with
no invalid dataset sources (`fog-p2-g3-stop-20261114-submission.json`).

Hosted setup: Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, Tesla T4; installation
126.478 s (`G3-stop-setup-log.txt`, `G3-stop-job-runtime.json`). The initial missing-archive
failure is retained. Three tests cover expanded read-only inputs and intact archives; fix
commits `6dcbf51` and `530ae08` change only the job restore path, not training semantics.

The corrected retry ended `CANCEL_ACKNOWLEDGED`; the final billed total is 749.868 s,
including 610.999 s for that attempt. No returned training log/checkpoint exists. The last
visible event was setup at 98.617 s; the subsequent stage is unverified. Five local tests
now cover opaque/expanded inputs, writable restoration and deadline preservation of saved
output. The existing private dataset was versioned with the identical archive under a binary
filename, retaining the old version. A new destination was rejected by automatic approval
review; updating the existing destination was permitted.

The existing dataset now lists exactly one `replay-work.bin`, privacy confirmed. Pilot
`fog-p2-g3-stop3-20261114` was accepted at `106034c`, with an internal 480-second stop
for export and 600-second platform cap; no other GPU job is active.

The opaque-input job exported successfully at its deadline, with no completed training
epoch. Setup was 92.656 s and input restored in 21.109 s. Settled quota use is 0.359879 h;
a longer private measurement now stops after epoch 1, capped at 30 min within the remaining
stage-1 allowance. No epoch throughput or peak-memory number exists yet.

Next: obtain one completed epoch, then complete the cross-job continuation within budget; choose batch size using
the specified 0.031 NLL comparison; finish G20, its value tables and CPU play smoke before G100.
Choose G100 epochs only from measured GPU rate and remaining budget, reserving two hours.

Not verified: hosted training/validation speed or memory, cross-job recovery, the
20k/100k learning curve, G100 value learning/sign/leakage checks, durable GPU checkpoints and
new Showdown play results. P2.1–P2.5 remain unchecked.
