# Replay blueprint GPU scale — 2026-11-14 (smoke, in progress)

**Answer.** The two completed GPU epochs run at **604–622 rows/s** (**613 aggregate**) on **Tesla T4**,
with **150–154 s** validation and **2.44 GiB** peak allocation. Full G3 projects to over two hours,
so it cannot complete inside stage 1's one-hour limit. The partial G3 cross-job continuation
is verified; G20 is next at the brief's default batch 1,024. The 20k/100k curve, P2.4 branch,
100k value reading and new play rates are unmeasured. **No gate is claimed.**

| Run (smoke) | Epochs completed / target | Batch | Policy epoch | Top-1 | NLL | Value Brier | Value ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B, historical CPU | 18 / 18 | 256 | 8 | 0.440802 | 1.579018 | 0.344501 | 0.298756 |
| G3, partial GPU | 2 / 18 | 1024 | 2 | 0.300391 | 1.895671 | 0.244778 | 0.033124 |
| G20 | pending | 1024 | — | — | — | — | — |
| G100 | pending budget selection | pending | — | — | — | — | — |

| Run/checkpoint (smoke) | Train value MSE | Holdout value MSE | Early/middle/late Brier | Early/middle/late ECE |
| --- | --- | --- | --- | --- |
| B, policy epoch 8 | 0.272015 | 1.378004 | 0.343345 / 0.363723 / 0.327324 | 0.278211 / 0.328123 / 0.291107 |
| B, value epoch 4 | 0.746103 | 0.956552 | unavailable | unavailable |
| G3, policy/value epoch 2 | 0.988902 | 0.979114 | 0.248064 / 0.244514 / 0.241777 | 0.020288 / 0.027854 / 0.050854 |
| G20 / G100 | pending | pending | pending | pending |

B's ten-bin reliability table and extremes are retained in `05-artifacts/B-reference.json`:
56.2588% of rows lie below 0.1 or above 0.9. Its historical best-value weights/report are absent;
only that epoch's MSE is available. G3's best of two epochs cannot be compared with B's best of 18 to attribute a
batch/device effect. No complete GPU batch-256 repeat or batch-quality decision is claimed.

Both CPU datasets are prepared: train **20,000 / 1,023,525 rows** and **101,928 / 5,217,022
rows**, plus the fixed **2,000 / 104,037** holdout. Every holdout shard is byte-identical, and
G20 is contained in the larger training set. All 112,000 games converted without quarantine;
8,072 newer non-holdout games were excluded. Rating floor stays 1500. Preparation/counts,
private input verification and timers are in `prepare-22000*` / `prepare-112000*`. Both private
datasets are ready and verified. Import/conversion/splitting used CPU.

GPU evidence: `G3-stop4-{metrics.csv,training-runtime.json,job-runtime.json,tail.txt}` and
`G3-first-epoch-summary.json`. Initial validation 160.716 s, subsequent pass 154.323 s;
peak `torch.cuda.max_memory_allocated` is 2,619,046,912 bytes. Runtime: Python 3.13.16,
PyTorch 2.11.0+cu130, CUDA 13.0; CUDA device 0 used, two T4s visible. Installation was
137.037 s in this attempt. The one-epoch training command and exact submitted source are retained.

Budget: **0.879015 quota-hours** across all calibration jobs, including failed attempts.
Continuation preserved the original `[0,1]` CSV bytes exactly as the prefix of `[0,1,2]`.
Calibration ended below one hour;
total allowance remains 20 hours. Earlier missing-archive and capped attempts are retained as
failed/partial, not passes (`gpu-hours.json`, ledger). The archive is now kept opaque. A redundant
new dataset destination was rejected by automatic approval review; the permitted version update
of the existing private dataset resolved it. Every dataset/job privacy check stays in the artifacts.

The two public options default off: `--prepare-only` and `bc.keep_best_value`. They change no
model, loss, schedule or filters. Validation: **71 affected application tests passed** outside
the command sandbox; **6 job-layout/deadline/logging tests passed**; `ruff check fog tests` clean.
Earlier sandbox shutdown timeouts and corrected mock-metric failures are recorded separately.

Both partial G3 weights are now in the one private checkpoint dataset; the one-command
download in `05-artifacts/checkpoints.json` was tested and both hashes/sizes match. G20/G100
weights are still pending. Both holdout reports and detailed before/after metrics are retained.

Next: G20 projects to 9.6 h including overhead allowance (`G20-plan.json`), with an 11-hour
platform cap and export deadline. Select G100 epochs/subset from actual G20 rate and remaining
allowance, reserving two hours. Full G3 calibration/batch comparison, G20/G100 results, the
strict 100k value reading, new play results and gates remain unverified. P2.1–P2.5 remain unchecked.
