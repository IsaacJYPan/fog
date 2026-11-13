# Brief 03: replay measurements

## Answer

Training measures 194 rows/s in the profile and 187 rows/s across R's training
epochs; model computation accounts for 96% of the profiled step. The original
22,000-game split and real 600-game recovery test pass. R differs from the
historical curve, while the required flag-off repeat has identical metrics and
final weights, and the original/current trainers agree on 20 real updates; no data-versus-steps
conclusion exists yet. The scale pipeline works on real data; hosted GPU
throughput and memory remain untested.

## Stage 2 table

| Run (games × epochs) | Steps | Train wall / rows/s | Last NLL / top-1 | Best report NLL / top-1 (epoch) | Final train NLL | Value MSE min / last |
| --- | ---: | --- | --- | --- | ---: | --- |
| R (1,000 × 6) | 1,286 | 2,799.8 s / 187.06 | 1.810490 / 0.340484 | 1.810496 / 0.340476 (6) | 1.737690 | 0.993827 / 1.247020 |
| A (1,000 × 18) | — | running; resumed epoch 3 completed | — | — | — | — |
| S (1,000 × 6) | — | not started | — | — | — | — |
| B (3,000 × 18) | — | projection pending | — | — | — | — |

Last-epoch fields come from `R-metrics.csv`; best fields are the standalone
holdout report. `R-summary.json` records every source and formula; the trainer
command/commit/attempts are in `R-execution.json` (commit `247a1e1`). R does not
reproduce the old curve to three decimals: baseline/epochs 1–6 are
2.306/2.045/1.992/1.917/1.850/1.826/1.810, against
2.306/2.043/1.990/1.909/1.841/1.819/1.804. The required six-epoch repeat with
resume/selection omitted matches every non-timing metric and all 376 final
parameter/buffer tensors exactly (`R-default-summary.json`). Default total-loss
selection picks epoch 5 (standalone policy NLL 1.825620), whereas R picks epoch
6 (1.810496). The options preserve this full training trajectory. Original and
current trainers also produce identical first 20 real-input losses and all 376
parameter/buffer tensors after those updates (`real20-comparison.json`);
their losses match the profiler exactly (`real20-profile-equality.json`).
The historical CPU differs and its thread count was not recorded, so hardware
versus thread/kernel contributions remain unisolated. No data-versus-steps or seed-noise conclusion
exists yet; R's value MSE barely beats constant zero at its best and rises above
1.0 afterward. Per-epoch value/policy curves are retained in `R-summary.json`.

## Per stage

- Setup: locked install and pinned Showdown build complete; four cores of CPU quota,
  16 GiB memory, PyTorch 2.13.0+cu130, no CUDA device. Import: 22,000 games in 194 s,
  with brief 01's rejection counts. Conversion: 22,000 games / 1,127,562 rows in 2,225 s wall, including the 758 s pause. The holdout has 2,000 games / 104,037 rows, uploaded 2026-02-20 04:17:33 through 2026-05-19 04:52:01 UTC. Training sets 1k/3k/20k have 52,983/156,597/1,023,525 rows. Fixed-ID splitting drops zero games and matches chronological holdout/train membership (`verify-split-output.txt`, `split-fixed-output.txt`).
- Stage 1: done, with no qualifying source changes. Before/repeat: 1.314/1.274 s per step, 188.123/194.040 rows/s; batch construction 0.039762/0.038087 s (3.03%/2.99%); forward 0.608237/0.589191 s; backward+optimizer 0.654632/0.634710 s; remainder 0.011513/0.012080 s. Belief heads: 46.11%/45.76% of forward. Batch-only: 7,182.28/7,100.46 rows/s. Both profiles used 20 warm-ups and 200 measured steps; separate cProfile top-15 files are retained. The first 20 losses match exactly (`profile-comparison.json`). No loader/prefetch criterion qualifies. Thread1/2/4 screens used 5 warm-ups and 30 measured steps: 2.417/1.561/1.235 s, slower or only 6.05% faster than the five-thread baseline, below 10%; alternatives change losses, so preserve five threads. Profile walls 330/321 s; screens 95.067/62.085/49.680 s (`profile-*-timing.json`, `profile-*-cprofile.txt`, `thread-benchmarks.json`).
- Stages 2–3: R completed using the unchanged five-thread recipe: 2,799.785 s
  trainer wall plus 108.494/110.127 s best/final reports, 3,018.406 s combined.
  Mean validation time is 154.662 s per pass, calculated for each epoch as
  `seconds_per_epoch - 52983/train_rows_per_second`; aggregate pure-training
  throughput is 187.060 rows/s. The saved state records 1,286 actual updates;
  the brief's step counts are approximate (streaming shard buffers end in partial
  batches). `R-summary.json` and the two report JSONs retain the measurements.
  Its projection is 1,638 s training plus seven 269 s holdout passes, 3,522 s
  through training/evaluation, or 4,060 s with two standalone reports
  (`R-projection.json`). The evaluation projection uses profile forward/batch/loss
  time per row; it is not a measured evaluation. R's epoch-1 policy NLL is
  2.045422 versus brief 01's 2.042711. The required flag-off repeat took
  2,737.795 s trainer wall and 107.419/125.688 s reports (2,970.903 s combined),
  matching R's non-timing metrics and all final tensors exactly. The saved
  pre-launch projection was 2,939.569 s (`R-default-projection.json`). A/S/B
  can now continue with the failed historical guard explicitly recorded. The
  original/current real-20 diagnostic passed in 426 s wall (222.034/201.122 s
  trainer calls, each including full baseline evaluation), with exact losses
  and all tensors equal (`real20-{comparison.json,output.txt,wall.txt}`).
  Baseline normalized metrics agree
  within 2.49e-8. Current CPU is AMD EPYC 9V74; brief 01 recorded Intel Xeon
  2.10 GHz. This platform difference is observed, not yet an isolated cause
  (`reproduction-platform.json`).
  A started at 06:47:15 UTC after the diagnosis was pushed. Its pre-launch
  projection is 4,914.936 s training plus 19 validation passes and two reports,
  8,072.140 s total (`A-projection.json`); its required kill/restart is planned
  after at least two completed epochs. That real interruption/recovery now
  passed: SIGTERM at completed epoch 2 exited −15 after 1,197.492 s; the identical
  command resumed at epoch 2 and completed epoch 3, with all earlier CSV rows
  (including timing fields) exactly unchanged (`A-resume-evidence.json`,
  `A-resume-metrics.csv`, `A-resume-tail.txt`). A continues; S/B remain unstarted.
- Stage 4: done on real replays. 600 games / 29,508 rows; train 500 / 24,278,
  holdout 100 / 5,230. After the first completion, only `training/` and `results/`
  were deleted. The identical command skipped import/conversion/split, was killed
  after epoch one (exit 143; 147.248 s), and resumed epoch two (exit 0; 151.991 s).
  Metric epochs went `[0,1]` to `[0,1,2]`, with earlier rows exactly unchanged.
  All numeric metrics match the first completed run exactly; elapsed-time columns
  are excluded. Post-completion step walls: train attempts combined 284.380 s;
  best/untrained scores 7.216/7.133 s; import/conversion/split skipped. Sources:
  `runner-smoke/` and `runner-first-completion/` in `03-artifacts/`, script
  `check_real_runner.py`, measured at `fa35867`. The first completion's initial
  evaluation competed with conversion and was unusually slow; conversion was
  paused from 04:02:31 to 04:15:09 UTC (758 s) to serialize CPU work.
- Stage 5: done: public clone, included fixed IDs, numbered cells, restart/download
  steps and measured CPU time/disk arithmetic (`scale-projections.json`). The
  20k full-pipeline reference is 9.62–9.95 h at the observed batch-256 rates;
  batch-1,024 and GPU throughput/memory remain unmeasured. Not run on a hosted GPU.
- Stage 6: not started.

## Deviations and corrections

The owner corrected the requested assignment from brief 02 to brief 03 during setup.
Work continued from `work/roadmap` on the required `work/p2-scale-measure` branch;
existing brief 02 reports/artifacts remain unchanged. The setup and first real
runner test were already underway and are retained as evidence. No question was
put to the owner. Commands and original outputs are in `03-artifacts/`; setup
evidence names retain a `continuation-` prefix. The import source shard was cached
from the genuine main import before the runner test. The runner still called and
verified the existing importer. R's historical reproduction failed; the required
flag-off repeat and original/current real-20 check ran before the other
comparisons. A−R and S−R are controlled on this machine; comparisons with the
legacy 3k run remain cross-platform references, rather than a clean isolated
data effect. No application code or runtime tuning was introduced for diagnosis.

## Not verified

R's historical curve reproduction (failed; full flag-off and original/current
real-20 equality passed); A/S/B;
hosted GPU execution; play strength. No gate is claimed.

## Checkpoints

R's best checkpoint is `runs/blueprint/g1000e6/checkpoints/bc_best.pt`, 23,548,503
bytes, SHA-256 `2126f07ad1d115d9f82d00a72a32485158cca17cda379eb90ed4d2a0746fd398`.
Paths/sizes/hashes for its final weights and resumable state, plus the engineering
smoke and flag-off repeat weights, are in `03-artifacts/checkpoints.json`. The machine may be
reclaimed; ignored checkpoints are not retained by pushing this branch.

## Suggestions

Finish the pinned-corpus comparisons before drawing a data or value-learning conclusion.
