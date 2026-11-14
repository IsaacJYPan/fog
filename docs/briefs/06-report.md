# Collection and play-training speed on a GPU — short smoke check

**Answer.** The worker probe collected **122.66/124.00 learner decisions/s on CPU** and
**158.79/163.22 on CUDA**, the selected batch 128, against MaxDamage. Two completed PPO
updates measured **144.98 collection decisions/s**, **159.89 processed update rows/s**
(6.08 s per optimizer step on average), and **37.50 decisions/s end to end**. At this initial
fixed-team rate, 30 GPU-hours project to **4.05 million decisions / 157,901 games**. The
update takes **71.9%** of completed-prefix wall time. These are smoke timings, not a learning result.

| Worker-only probe, 512 games/repeat | CPU 32 | CUDA 32 | CUDA 64 | CUDA 128 |
| --- | ---: | ---: | ---: | ---: |
| MaxDamage, repeat 1 / 2, learner decisions/s | 122.66 / 124.00 | 150.79 / 150.75 | 155.66 / 158.19 | 158.79 / 163.22 |
| self, repeat 1 / 2 | not completed | 106.19 / 107.68 | not completed | not completed |

Machine: one T4, two logical Xeon 2.00 GHz CPUs, PyTorch 2.14.1+cu130, pinned Showdown
`e440c4a`. Probe timings exclude full-buffer IPC; trainer timings include it. All current
numbers and commands are in `06-artifacts/runtime-t4-counts/`.

The completed prefix is **26,268 decisions, 1,024 games, 81 optimizer steps, three effective
epochs**, in **700.389 s** through its saved checkpoint. Collection/update/other shares are
25.9% / 71.9% / 2.2%. Timed forward calls take 86.9% of optimization time; one live stack
caught `_permute_hidden_moves` reading GPU scalars, but its exact share is unmeasured.
Sampled whole-attempt GPU peak: **11,561 MiB**; 1,024-row minibatches fit without activation
checkpointing. Including the unfinished third update discarded at the owner's stop request,
the entire attempt cost 995.824 s, or 26.38 retained decisions/s. Prefix and spent-time rates
are kept separate in `short-run-summary.json`.

**Scope and recovery.** The owner chose a minimal timing check instead of extending the
GPU budget. Training stopped with two complete updates; the 100k run, 200-game greedy
evaluation and six remaining probe repeats were not completed. Weights and optimizer/RNG
sidecars were copied locally and their ZIP integrity checked. Latest:
`runs/collect_gpu/checkpoints/ppo_26268.state.pt`. The T4 was released; no runtime remains.
Cumulative allocation upper bound, including prior attempts: **2.441111 GPU-hours**, below three.

**Fix/checks.** The first runtime exhausted host RAM while serializing unused probe buffers.
The probe now returns counters around the unchanged actor worker. Its regression fails
before the fix and passes after; eight focused checks and final pinned Ruff pass. Earlier
CPU/full-buffer and failed-run evidence is retained separately. No model, loss, trainer or
actor implementation changed.

**Not verified:** learning, strength, complete same-runtime matrix, steady-state weekly
throughput, or recovery after an actual runtime drop. P0.5 remains open.
