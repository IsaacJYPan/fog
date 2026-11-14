# Collection and play-training speed on a GPU (smoke)

**Answer.** Collection on this GPU host, PPO update speed, end-to-end speed,
and decisions/games per 30-hour week are not measured yet. After the existing
GPU job finished, the initial private collection job at `c4df5e2` failed
before cloning: `/kaggle/temp` did not exist. The next allocation completed
all self-play rows, then hit its time limit before MaxDamage. A longer
same-job matrix is being prepared within the remaining GPU budget.

| Stage | Status | Evidence |
| --- | --- | --- |
| 1: CPU/GPU collection, twice per setting | Self-play done; MaxDamage not run | `06-artifacts/collection-attempt2/` |
| 2: fixed-team PPO and greedy evaluation | Depends on stage 1's chosen batch and CPU count | Not run |
| 3: bottleneck and 30-hour projection | Depends on measured collection/update rates | Not run |

The private collection entry point is `06-artifacts/gpu_job.py`. It clones a
pushed commit supplied through `FOG_COMMIT`, builds pinned Showdown, preserves
the installed CUDA PyTorch, records CPU hardware/quota and GPU memory, and
runs the exact collection matrix. It stops on the first failed probe and
preserves completed rows. Use T4: the installed client documents P100 as
retired. The two completed allocations cost at most 0.937223 GPU-hours,
including startup/export, on two T4s. A complete matrix will use a
2,520-second platform cap; training gets only the measured remaining budget.
Both allocated GPUs count toward the budget and weekly projection.

Partial self-play smoke, four Intel Xeon @ 2.00GHz cores: CPU B32
155.76 / 155.37 learner decisions/s; CUDA B32 205.81 / 208.95;
B64 209.10 / 213.15; B128 216.34 / 212.95. These are not combined with
future hosts to form the final comparison table.

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
