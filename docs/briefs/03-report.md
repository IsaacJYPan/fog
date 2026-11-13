# Brief 03: replay measurements

## Answer

Measurements are in progress. The pinned dataset is accessible and the 22,000-game
import reproduced brief 01's filter counts; conversion is running. The reviewed
trainer/runner code passed all 82 focused tests, and the first real 600-game runner
test has resumed from epoch one. Training throughput, the data-versus-steps reading
and hosted GPU readiness are not yet established.

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
  with brief 01's rejection counts. Conversion in progress.
- Stage 1: waiting for the verified split; no loader/model changes made.
- Stages 2–3: not started.
- Stage 4: real 600-game pipeline in progress; first epoch saved, SIGTERM delivered,
  identical command resumed from epoch one. A further post-completion kill/restart
  check will follow brief 03's procedure. Its initial evaluation competed with
  conversion and was unusually slow; conversion was paused to serialize CPU work.
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

Full split counts/dates; idle-machine throughput; R reproduction; A/S/B; the full
brief 03 runner procedure; hosted GPU execution; play strength. No gate is claimed.

## Checkpoints

Only the engineering smoke checkpoint currently exists, under
`runs/blueprint_runner_smoke/training/checkpoints/`. Sizes/hashes will be retained
after the recovery check. The machine may be reclaimed; ignored checkpoints are
not retained by pushing this branch.

## Suggestions

Finish the pinned-corpus comparisons before drawing a data or value-learning conclusion.
