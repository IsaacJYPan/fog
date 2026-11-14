# Collection and play-training speed on a GPU (smoke, in progress)

**Answer.** Self-play collection measured **155–156 learner decisions/s on CPU** and
**206–216 on CUDA**, four Intel Xeon @ 2.00GHz cores and two Tesla T4s. MaxDamage
collection, PPO update speed, end-to-end decisions/games per second and the 30-hour
projection remain unmeasured. The owner chose to finish G20 first, then run the missing
MaxDamage matrix and full 100k-decision PPO loop. No new GPU job has been submitted.

| Self-play, 512 games/repeat (smoke) | Repeat 1 learner decisions/s | Repeat 2 learner decisions/s |
| --- | ---: | ---: |
| CPU, batch 32 | 155.76 | 155.37 |
| CUDA, batch 32 | 205.81 | 208.95 |
| CUDA, batch 64 | 209.10 | 213.15 |
| CUDA, batch 128 | 216.34 | 212.95 |

These rows are from `06-artifacts/collection-attempt2/`; they are a separate-job result,
not a completed same-job self/MaxDamage matrix. Attempt 1 failed before clone because
`/kaggle/temp` was absent; attempt 2 reached its internal deadline after these eight rows.
Both failed/partial allocations remain in the ledger and artifacts.

The planned private `FOG_STAGE=combined` job uses `06-artifacts/gpu_job.py`, the reviewed
current roadmap trainer, and `configs/collect_gpu_check.yaml`. It measures both repeats of
CPU batch 32 and CUDA batches 32/64/128 against MaxDamage, selects the aggregate fastest
GPU batch, then trains the exact 100k-decision recipe and evaluates 200 greedy games on
the first pool-A team. The config's four-worker/batch-128 values are a template, replaced
by measured hardware/selection before training. Minibatch stays 1,024; a CUDA OOM may
only enable the brief's activation-checkpointing option, with the failed attempt retained.

Budget: at most **0.937222 allocated GPU-hours** already used, counting both T4s and
submission-to-observed-terminal upper bounds. A 3,600-second platform cap plus prior
allocations totals **2.937222 allocated GPU-hours**, leaving 113 session seconds for
additional startup/export; the internal deadline is 3,300 seconds including setup. The
remaining budget is shared by probes, training and evaluation. The initial G20/collection
overlap is recorded; every new launch requires an empty account-wide active/queued-job check.

The timing wrapper preserves the trainer's native phase columns and adds `timing_` fields,
CUDA-synchronized optimizer/forward timers and the full update phase. End-to-end wall
includes all training attempts; saved-state continuations retain only their completed timing
prefix. A regression fails on the prior wrapper and passes after the fix. **15 focused tests
pass**, timing checks preserve update metrics/model tensors with CUDA calls mocked, Ruff
and byte compilation pass (`endtoend-*`, `timing-native-before.txt`). No new model, loss,
trainer or search edits were made; the reviewed brief-04 trainer was merged from the roadmap.

**Not verified:** MaxDamage collection, PPO GPU memory/speed, end-to-end games/second,
30-hour projections and learning on the batched GPU path. The completed CUDA self-play rows
exercised spawned actors. The single same-job matrix requirement is not met. P0.5 stays open.
