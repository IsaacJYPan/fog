# Brief 03: replay measurements

## Answer

Training measures 194 rows/s (1.27 s/step) on this CPU, and model compute accounts for 97% of the step. The pinned dataset is accessible and the 22,000-game
import and split reproduce brief 01 exactly, including all fixed holdout IDs. The reviewed
trainer/runner code passed all 82 focused tests, and the real 600-game runner
passed its post-completion kill/restart test. The data-versus-steps reading is pending R/A/S/B; the scale pipeline works on real data, but hosted GPU throughput and memory remain untested.

## Stage 2 table

| Run | Games | Epochs | Status |
| --- | --- | --- | --- |
| R | 1,000 | 6 | waiting for verified split and profile |
| A | 1,000 | 18 | not started |
| S | 1,000 | 6 | not started |
| B | 3,000 | 18 | projection pending |

No data-versus-steps conclusion or seed-noise estimate exists yet.

## Per stage

- Setup: locked install and pinned Showdown build complete; four cores of CPU quota,
  16 GiB memory, PyTorch 2.13.0+cu130, no CUDA device. Import: 22,000 games in 194 s,
  with brief 01's rejection counts. Conversion: 22,000 games / 1,127,562 rows in 2,225 s wall, including the 758 s pause. The holdout has 2,000 games / 104,037 rows, uploaded 2026-02-20 04:17:33 through 2026-05-19 04:52:01 UTC. Training sets 1k/3k/20k have 52,983/156,597/1,023,525 rows. Fixed-ID splitting drops zero games and matches chronological holdout/train membership (`verify-split-output.txt`, `split-fixed-output.txt`).
- Stage 1: done, with no qualifying source changes. Before/repeat: 1.314/1.274 s per step, 188.123/194.040 rows/s; batch construction 0.039762/0.038087 s (3.03%/2.99%); forward 0.608237/0.589191 s; backward+optimizer 0.654632/0.634710 s; remainder 0.011513/0.012080 s. Belief heads: 46.11%/45.76% of forward. Batch-only: 7,182.28/7,100.46 rows/s. Both profiles used 20 warm-ups and 200 measured steps; separate cProfile top-15 files are retained. The first 20 losses match exactly (`profile-comparison.json`). No loader/prefetch criterion qualifies. Thread1/2/4 screens used 5 warm-ups and 30 measured steps: 2.417/1.561/1.235 s, slower or only 6.05% faster than the five-thread baseline, below 10%; alternatives change losses, so preserve five threads. Profile walls 330/321 s; screens 95.067/62.085/49.680 s (`profile-*-timing.json`, `profile-*-cprofile.txt`, `thread-benchmarks.json`).
- Stages 2–3: not started.
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
- Stage 5: notebook page updated for public cloning and included fixed IDs;
  projections await measurement. Not run on a hosted GPU.
- Stage 6: not started.

## Deviations and corrections

The owner corrected the requested assignment from brief 02 to brief 03 during setup.
Work continued from `work/roadmap` on the required `work/p2-scale-measure` branch;
existing brief 02 reports/artifacts remain unchanged. The setup and first real
runner test were already underway and are retained as evidence. No question was
put to the owner. Commands and original outputs are in `03-artifacts/`; setup
evidence names retain a `continuation-` prefix. The import source shard was cached
from the genuine main import before the runner test. The runner still called and
verified the existing importer.

## Not verified

R reproduction; A/S/B;
hosted GPU execution; play strength. No gate is claimed.

## Checkpoints

Only engineering smoke checkpoints currently exist, under
`runs/blueprint_runner_smoke/training/checkpoints/`. Sizes/hashes are retained in
`03-artifacts/checkpoints.json`. The machine may be reclaimed; ignored checkpoints are
not retained by pushing this branch.

## Suggestions

Finish the pinned-corpus comparisons before drawing a data or value-learning conclusion.
