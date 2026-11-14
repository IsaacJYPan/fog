# Collection and play-training speed on a GPU (smoke)

**Answer.** Collection on this GPU host, PPO update speed, end-to-end speed,
and decisions/games per 30-hour week are not measured yet. After the existing
GPU job finished, the initial private collection job at `c4df5e2` failed
before cloning: `/kaggle/temp` did not exist. The startup fix is tested;
collection is being retried, and training follows its measurements.

| Stage | Status | Evidence |
| --- | --- | --- |
| 1: CPU/GPU collection, twice per setting | Initial startup failed; retry prepared | `06-artifacts/collection-attempt1-*.json` |
| 2: fixed-team PPO and greedy evaluation | Depends on stage 1's chosen batch and CPU count | Not run |
| 3: bottleneck and 30-hour projection | Depends on measured collection/update rates | Not run |

The private collection entry point is `06-artifacts/gpu_job.py`. It clones a
pushed commit supplied through `FOG_COMMIT`, builds pinned Showdown, preserves
the installed CUDA PyTorch, records CPU hardware/quota and GPU memory, and
runs the exact collection matrix. It stops on the first failed probe and
preserves completed rows. Use T4: the installed client documents P100 as
retired. Retry collection/training platform limits are 1,800/3,000 seconds, with
300 seconds reserved inside each for export. Even a two-T4 allocation is
bounded to three GPU-hours including a 600-second reserve for the first
failure; its submission-to-observed-terminal upper bound is 141 seconds.
Both allocated GPUs count toward the budget and weekly projection.

`06-artifacts/training_timing.py` supplies optional process-local timing hooks
through a temporary `sitecustomize.py`: collection/update wall times go into
the trainer's existing metrics CSV and a separate JSONL file. CUDA updates
are synchronized at their boundaries; update rows/s counts unique batch rows.
The CPU engineering check (`check_timing.py`, `timing-equivalence.json`)
preserves every update metric and final parameter exactly with CUDA timing
calls mocked. This is not GPU validation. Local pinned Showdown builds;
18 focused rollout/process-pool tests pass and Ruff is clean (artifact outputs).

**Fixes.** The job creates its temporary parent before cloning. The regression
fails without the fix and passes with it (`job-bootstrap-{before,test}.txt`).
No model, loss, trainer or search code changed.

**Not verified.** GPU startup in spawned actors, CUDA tensor placement,
collection rates, PPO GPU memory/speed, and learning on the batched GPU path.
P0.5 remains unchecked. This report records preparation, not a completed run.
