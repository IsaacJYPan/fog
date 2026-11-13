# Brief 03: replay measurements

## Answer

Training profiles at 188–194 rows/s and completed runs train at 158–187 rows/s;
model computation accounts for 96% of the profiled step. At 1,000 games, 18
epochs overfit: final policy NLL is 1.949 versus 1.810 after six, although the
selected epoch-8 checkpoint reaches 1.736. Value generalization remains weak;
B is still running and legacy 3k comparisons are cross-platform references.
The scale pipeline and real recovery tests pass on CPU; hosted GPU throughput
and memory remain untested.

## Stage 2 table and reading

Unless stated otherwise, sources are under `docs/briefs/03-artifacts/`. Last-epoch
fields use `<run>-metrics.csv`; best fields use the standalone holdout report.
Trainer wall excludes the two reports and includes A's interrupted attempt.
Exact commands, commits, attempts and formulas are in `<run>-execution.json`
and `<run>-summary.json`. S uses seed 20261202; the other runs use 20261201.

| Run (games × epochs) | Steps | Trainer wall / rows/s | Last NLL / top-1 | Best report NLL / top-1 (epoch) | Final train NLL | Value MSE min / last |
| --- | ---: | --- | --- | --- | ---: | --- |
| R (1,000 × 6) | 1,286 | 2,799.8 s / 187.06 | 1.810490 / 0.340484 | 1.810496 / 0.340476 (6) | 1.737690 | 0.993827 / 1.247020 |
| A (1,000 × 18) | 3,870 | 9,355.4 s / 157.72 | 1.948926 / 0.368527 | 1.736081 / 0.375367 (8) | 1.241824 | 1.015312 / 1.595830 |
| S (1,000 × 6) | 1,286 | 3,224.9 s / 158.59 | 1.794974 / 0.345247 | 1.794978 / 0.345246 (6) | 1.718056 | 0.999632 / 1.237886 |
| B (3,000 × 18) | — | running; budget 21,405 s | — | — | — | — |

The two-seed difference |S−R| is 0.015516 NLL; effects below about twice that,
0.031031, are not claimed. A's final deterioration of 0.138436 exceeds this
threshold: training NLL falls to 1.241824 while holdout NLL rises to 1.948926,
a 0.707102 gap. Its selected epoch-8 policy improves over R's selected policy,
but continuing to the end worsens NLL despite higher top-1. Thus longer training
at this size eventually overfits. A cannot supply a clean data-effect comparison
with the legacy 3k six-epoch NLL of 1.612 because historical reproduction failed.
B's schedule is unfinished; its intermediate checkpoints do not replace that
completed-schedule reference. R/S value minima barely beat constant zero's MSE
of 1.0, and A never does; all finish worse. Full policy/value curves and exact
controlled deltas are retained in the run summaries and metrics.

## Per stage

- **Setup: done.** Four CPU cores of quota, 16 GiB memory, PyTorch 2.13.0+cu130,
  five runtime threads, no CUDA device. The pinned 22,000-game import took 194 s;
  conversion produced 1,127,562 rows in 2,225 s, including a 758 s pause. Every
  requested split matches brief 01: holdout 2,000 games/104,037 rows with exact
  fixed IDs; train 1k/3k/20k contain 52,983/156,597/1,023,525 rows. Fixed splitting
  drops zero games. See `verify-split-output.txt`, `split-fixed-output.txt`,
  conversion outputs and `continuation-commands.txt`. The reviewed application
  passed 82 focused tests in 20.26 s and Ruff; no full suite ran.
- **Stage 1: done; no qualifying source change.** Before/repeat profiles took
  330/321 s, measuring 200 updates after 20 warm-ups: 1.314/1.274 s per step,
  188.123/194.040 rows/s. Batch building is 3% of a step and 7,100–7,182 rows/s
  alone; belief heads take about 46% of forward time. Loader optimization and
  prefetch do not qualify. Thread screens at 1/2/4 used 95.067/62.085/49.680 s;
  four threads saved only 6.05%, below 10%, and changed losses. Five threads
  remain unchanged. Before/repeat first 20 losses match exactly; top-15 profiles
  and full phase timings are in `profile-*` and `thread-benchmarks.json`.
- **Stage 2: partly done.** R/A/S and both reports are complete. Combined
  trainer/report walls are 3,018.406/9,611.736/3,487.043 s. B is running; its
  saved projection is 18,060.446 s from the stage-1 rate, or a conservative
  21,405.112 s from recent measured rates, below eight hours (`B-projection.json`).
  Settings remain frozen. Partial B metrics are explicitly snapshots.
  The environment stopped with atomic epoch 15 saved. B resumed identically
  at 21:28:09 UTC, preserving rows 0–15. Its original exit/total wall are unknown;
  completed epoch timers provide an 18,121.330 s lower bound, excluding downtime
  (`B-environment-resume.json`).
- **Stage 3: done on real data.** A was SIGTERM-killed after atomic state epoch 2
  (exit −15, 1,197.492 s) and restarted with the identical command. It continued
  through epoch 18; the original CSV prefix is exactly unchanged, including
  timing fields (`A-resume-evidence.json`, final `A-execution.json`).
- **Stage 4: done on real data.** The runner imported/converted 600 games/29,508
  rows, then trained/scored. After first completion, only training/results were
  removed; the same command skipped import/conversion/split, was killed after
  epoch one, and resumed epoch two. Attempts took 147.248/151.991 s; previous CSV
  rows and all numeric metrics match the first completion. Evidence is
  in `runner-smoke/` and `runner-first-completion/`. Initial runner work overlapped
  the main conversion; the main converter's 758 s pause and those non-idle timings
  are recorded.
- **Stage 5: done; hosted GPU untested.** The page has public cloning, committed
  holdout IDs, numbered cells, restart/download steps and CPU time/disk arithmetic
  (`scale-projections.json`). The 20k pipeline reference is 9.62–11.69 h at
  measured/profile batch-256 rates; batch-1,024/GPU performance is unmeasured.
- **Stage 6: not run yet.** Awaiting the best completed comparison checkpoint.

## Deviations and corrections

The owner corrected brief 02 to brief 03 during setup. The initial brief-02
commit `9867251` remains on `work/p2-scale-prep`; this branch starts at `4f9d75c`.
On this branch, the existing brief-02 report/artifacts remain unchanged. No owner
question was needed. Streaming buffer boundaries add partial batches, so actual
optimizer steps exceed the brief's approximate counts.

Historical three-decimal reproduction **failed**, and remains failed. Baseline
normalized metrics agree within 2.49e-8, but the current AMD CPU differs from the
recorded Intel CPU and the old thread count is unknown. The required flag-off
six-epoch repeat (2,970.903 s including reports) matches all non-timing metrics
and all 376 final tensors exactly.
Original/current trainers also match 20 real updates and all tensors in both
shared and fresh processes; fresh children start without NumPy/Torch preloaded.
These controls took 426/451.193 s and rule out measured differences in the tested
update/startup paths, without isolating specific historical hardware/thread/kernel
contributions. Sources: `R-default-summary.json`, `real20-*`, `fresh-real20/`,
`reproduction-platform.json`. No model, loss, learning rate, batch or runtime
setting changed for diagnosis.

## Not verified

The exact historical trajectory, completed B, play strength, hosted CUDA
execution, GPU memory/throughput, and scale projections as actual runs remain
unverified. No roadmap task or gate is claimed.

## Checkpoints

The best completed run is currently A:
`runs/blueprint/g1000e18/checkpoints/bc_best.pt`, 23,548,503 bytes, SHA-256
`02dfddcf7f52bcf45b225bf87778c16b33fbb8056fd55ba0b5c3588d47405a1b`.
`checkpoints.json` lists every retained best/final/state path, size and hash.
The machine may be reclaimed; ignored weights are not retained by pushing.

## Suggestions

Measure CUDA throughput/memory before budgeting the first scale session.
A same-runtime 3k six-epoch run would isolate the legacy data comparison cleanly.
