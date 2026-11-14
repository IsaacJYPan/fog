# Collection and play-training speed on a GPU (smoke, in progress)

**Answer.** On two logical CPU threads and one T4, the corrected worker-only probe
collects **106.19/107.68 learner decisions/s in self-play** and **150.79/150.75 against
MaxDamage**, batch 32, 512 games per repeat. The remaining sweep, full PPO update rate,
end-to-end throughput, learning and the 30-hour projection are still pending.

| CUDA batch 32, smoke | Repeat 1 learner decisions/s | Repeat 2 |
| --- | ---: | ---: |
| self | 106.19 | 107.68 |
| MaxDamage | 150.79 | 150.75 |

The initial standard T4's four CPU rows and failed CUDA row are retained under
`06-artifacts/runtime-t4/`. Its first CUDA probe was killed by **host RAM exhaustion**
while returning observation buffers; sampled device use was 415 MiB. High-memory T4
requests were unavailable. The replacement standard T4 uses the same actor code, with
only probe counters transferred to the parent. Those timings exclude full-buffer IPC;
PPO still transfers real buffers. Fresh CPU baselines are being measured with the same
counter-only boundary, so the earlier CPU rows are not the corrected GPU comparison.

The memory regression fails before the fix and passes after it; eight focused checks
pass and pinned Ruff is clean (`runtime-t4-counts/`). No model, loss, trainer or actor
implementation changed. Environment: PyTorch 2.14.1+cu130, Showdown `e440c4a`.

Completed probes are skipped on restart and copied off the runtime every 20 seconds.
PPO saves model/optimizer/RNG state at update boundaries with an 8,000-decision snapshot
interval; weights, state sidecars and metrics are copied locally. Evaluation resumes at
four-game blocks. An active probe or unfinished update must be repeated after a drop.

Run order now protects the full-loop result: finish MaxDamage's matrix, train/evaluate,
then complete the remaining self-play rows. One newly started CPU self-play row was
preempted after 24.188 seconds and will restart. It is not a throughput result.

All earlier allocations count toward the **3 GPU-hour** budget. The local 10:21 UTC
stop guard caps the conservative cumulative allocation at **2.959444 GPU-hours**
(`runtime-t4-counts/session.json`). Only one T4 is active. P0.5 stays open.
