# Collection and play-training speed on a GPU (smoke, in progress)

**Answer.** On the fresh single-T4 runtime, the first CPU self-play repeat collected
**72.92 learner decisions/s** over 512 games. The second repeat and the rest of the
same-runtime matrix are running. CUDA collection speed, PPO update rate, end-to-end
throughput, learning and the 30-hour projection are not yet measured in this allocation.

| Opponent / device / batch | Repeat 1 learner decisions/s | Repeat 2 |
| --- | ---: | ---: |
| self / CPU / 32 | 72.92 | pending |

Machine: one Tesla T4, 15,360 MiB; two logical Intel Xeon 2.00 GHz CPUs;
PyTorch 2.14.1+cu130; pinned Showdown `e440c4a`. Evidence is under
`06-artifacts/runtime-t4/`, including raw probe output, commands, commit IDs,
process wall time and two-second GPU-memory samples. Every number is **smoke**.

The runtime CLI login and SSH connection work. Setup reused the supplied CUDA PyTorch;
the missing Node.js was installed. An unpinned Ruff invocation reported findings;
the lock's Ruff 0.15.22 is clean. **17 focused tests passed in 117.72 seconds**.
No application, model, loss or trainer change was needed.

Completed probes are skipped on restart. Evidence is copied to the controlling computer
every 20 seconds. Training will use the brief's config with measured workers/batch;
runtime-only overrides place its outputs in `runs/collect_gpu` and save full training
state every 8,000 decisions, at update boundaries. Weights, optimizer/RNG sidecars and
metrics are copied off the runtime. Evaluation uses its existing four-game-block resume.
An interrupted active probe or unfinished update must be repeated.

The earlier attempts remain recorded and count up to **0.937222 GPU-hours**. This
single-T4 allocation is capped at two hours, including setup/export: cumulative upper
bound **2.937222 GPU-hours** (`runtime-t4/session.json`). A local deadline guard stops it.

**Not verified:** the remaining probe rows, GPU PPO memory/speed, learning, weekly
projections and recovery after an actual runtime loss. P0.5 remains open.
