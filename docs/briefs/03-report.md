# Brief 03: replay measurements

## Answer

Training runs at 157–187 rows/s in completed runs (188–194 in short profiles),
with 96% of the profiled step spent in model computation. The best policy reaches
NLL 1.579/top-1 0.441; longer training overfits at both sizes, supporting a
data/generalization limit rather than extra updates alone explaining the earlier
gain. The exact historical data effect remains unisolated across CPUs, and value
learns only weak early signal before finishing worse than constant zero. The CPU
scale pipeline and real recovery tests pass; hosted GPU throughput and memory
remain untested.

## Stage 2 table and reading

Sources below are in `docs/briefs/03-artifacts/`. Last-epoch fields use
`<run>-metrics.csv`; best fields use standalone holdout reports. Rows/s measures
pure training; trainer wall includes validation. B's exact wall is unavailable.
Commands, commits, attempts and formulas are in `<run>-execution.json` and
`<run>-summary.json`. S uses seed 20261202; the others use 20261201.

| Run (games × epochs) | Steps | Trainer wall / rows/s | Last NLL / top-1 | Best report NLL / top-1 (epoch) | Final train NLL | Value MSE min / last |
| --- | ---: | --- | --- | --- | ---: | --- |
| R (1,000 × 6) | 1,286 | 2,799.8 s / 187.06 | 1.810490 / 0.340484 | 1.810496 / 0.340476 (6) | 1.737690 | 0.993827 / 1.247020 |
| A (1,000 × 18) | 3,870 | 9,355.4 s / 157.72 | 1.948926 / 0.368527 | 1.736081 / 0.375367 (8) | 1.241824 | 1.015312 / 1.595830 |
| S (1,000 × 6) | 1,286 | 3,224.9 s / 158.59 | 1.794974 / 0.345247 | 1.794978 / 0.345246 (6) | 1.718056 | 0.999632 / 1.237886 |
| B (3,000 × 18) | 11,431 | unknown, ≥21,149.9 s / 157.29 | 1.842549 / 0.426488 | 1.579018 / 0.440802 (8) | 1.062751 | 0.956552 / 1.601174 |

The two-seed NLL difference |S−R| is 0.015516; effects below about twice that,
0.031031, are not claimed. At completed schedules, controlled A−R is 0.138436
worse. B ends 0.230221 above legacy 3k6, a cross-platform reference. A/B
training–holdout gaps reach 0.707102/0.779798; both holdout minima occur at epoch
8, followed by clear upturns. This supports a data/generalization limit: extra
updates alone do not explain the earlier gain. A's 0.336598 difference from
legacy 3k6 suggests a data effect, but historical reproduction failed, so its
exact size is not isolated; a same-runtime 3k six-epoch run would resolve it.
B's selected-checkpoint improvement over legacy 3k6 is only 0.033310, barely
above the rough noise threshold and across CPUs, so it is not a clean step gain.
Value briefly beats constant-zero MSE 1.0 (B minimum 0.956552), but every final
head is worse; selected B has MSE 1.378004. Its switch mass averages
0.454911 on human-switch rows versus 0.192942 on move rows; switch-target
accuracy is 0.449789. Full current/legacy policy and value curves, constant
reference and exact comparisons are in `comparison-curves.txt` and
`comparison-reading.json`.

## Per stage

- **Setup: done.** Four CPU cores of quota, 16 GiB memory, PyTorch 2.13.0+cu130,
  five runtime threads, no CUDA. Import: 22,000 games in 194 s. Conversion:
  1,127,562 rows in 2,225 s, including a 758 s pause. Every split matches brief
  01, including all 2,000 holdout IDs/104,037 rows; fixed splitting drops zero
  games (`verify-split-output.txt`, `split-fixed-output.txt`, conversion outputs,
  `continuation-commands.txt`). The reviewed application passed 82 focused tests
  in 20.26 s and Ruff; no full suite ran.
- **Stage 1: done; no qualifying source change.** Before/repeat profiles took
  330/321 s, measuring 200 updates after 20 warm-ups: 1.314/1.274 s per step,
  188.123/194.040 rows/s. Batch building is 3% of a step and 7,100–7,182 rows/s
  alone; belief heads take about 46% of forward time. Loader optimization and
  prefetch do not qualify. Four threads save only 6.05%, below 10%, and change
  losses; five remain unchanged. First 20 losses match exactly. Full phase
  timings, thread screens and top-15 profiles are in `profile-*` and
  `thread-benchmarks.json`.
- **Stage 2: done, with B timing incomplete.** R/A/S trainer-plus-report walls:
  3,018.406/9,611.736/3,487.043 s. B's saved conservative projection was
  21,405.112 s, below eight hours. Its original exit/wall were lost when the
  environment ended at an unobserved time. Completed epoch-1–15 timers give
  18,121.330 s; the measured resumed attempt adds 3,028.582 s, so trainer wall
  is at least 21,149.912 s. Reports add 213.020 s; total is at least 21,362.931 s.
  Downtime is excluded, not invented (`B-summary.json`, `B-environment-resume.json`).
- **Stage 3: done on real data.** A was SIGTERM-killed after atomic epoch 2
  (exit −15, 1,197.492 s), resumed identically and completed epoch 18; its
  original CSV prefix remains exact. B also resumed from atomic epoch 15 after
  the environment restart, preserving every earlier row. Runtime settings stayed
  unchanged (`A-resume-evidence.json`, `B-environment-resume.json`).
- **Stage 4: done on real data.** The 600-game runner produced 29,508 rows and
  completed training/scoring. After first completion, only training/results were
  removed; the identical command skipped import/conversion/split, was killed
  after epoch one and resumed epoch two. Attempts took 147.248/151.991 s; prior
  rows and all numeric metrics match the first completion (`runner-smoke/`,
  `runner-first-completion/`). Initial work overlapped the main conversion;
  the main converter's pause and non-idle timings are recorded.
- **Stage 5: done; hosted GPU untested.** The page contains pinned public cloning,
  fixed IDs, numbered cells, restart/download steps and measured CPU time/disk
  arithmetic. CPU20k pipeline reference: 9.62–11.74 h at batch-256 measured/profile
  rates. Actual runner batch-1,024/GPU performance is unmeasured
  (`scale-projections.json`).
- **Stage 6: pending smoke results.** B epoch 8 is selected; no gate is claimed.

## Deviations and corrections

The owner corrected brief 02 to brief 03 during setup. Initial commit `9867251`
remains on `work/p2-scale-prep`; this branch starts at `4f9d75c`. On this branch,
the existing brief-02 report/artifacts remain unchanged. No owner question was
needed. Streaming buffer boundaries add partial batches, so measured steps exceed
the brief's approximate counts.

Historical three-decimal reproduction **failed**. The current AMD CPU differs
from the recorded Intel CPU; old thread count is unknown. The required flag-off
six-epoch repeat matches all non-timing metrics and 376 final tensors exactly
(2,970.903 s including reports). Original/current trainers also match 20 real
updates/tensors in shared and clean fresh processes (426/451.193 s). These show
no measured difference in tested update/startup paths without isolating historical
hardware/thread/kernel contributions (`R-default-summary.json`, `real20-*`,
`fresh-real20/`, `reproduction-platform.json`). No model, loss, learning rate,
batch or runtime setting changed for diagnosis or recovery.

## Not verified

The exact historical trajectory, B's exact total wall/old exit, play strength,
hosted CUDA execution/memory/speed, and scale projections as actual runs remain
unverified. No roadmap task or gate is claimed.

## Checkpoints

Best policy: `runs/blueprint/g3000e18/checkpoints/bc_best.pt`, 23,548,503 bytes,
SHA-256 `bed0804dc21d83b27acba45d3359c5a8ec88cb3233210c21a65d719df5a02c8a`.
`checkpoints.json` lists all retained best/final/state paths, sizes and hashes.
The machine may be reclaimed; ignored weights are not retained by pushing.

## Suggestions

Measure CUDA throughput/memory before budgeting the first scale session.
A same-runtime 3k six-epoch run would isolate the historical data comparison.
