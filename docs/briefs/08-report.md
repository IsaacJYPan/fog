# GPU update speed after the belief-loop fix

**Answer.** In this smoke, mean PPO optimizer-step time fell from **6.08 to 1.31 s** and end-to-end throughput rose from **37.50 to 122.80 decisions/s**. Replay training reached **1,850.26 rows/s**, against **583.24** before (**3.17×**), at **0.553 s/step**. Backward is the largest sampled PPO component: **0.735 s, 54.4%** of a 1,024-row step. The update still takes more wall time than collection: **52.3% versus 41.8%**. Brief 14's numerical condition is met in this short smoke; its review is still required.

Machine: four logical Xeon 2.00 GHz CPUs; two T4s allocated, all model work on `cuda:0`; PyTorch 2.11.0+cu130. Before: two CPUs, one T4, PyTorch 2.14.1+cu130. Production bytes are identical to `work/roadmap` at `f36bc11`; measurement commit `e23682f`.

| PPO, smoke | Brief 06 | After fix |
| --- | ---: | ---: |
| Mean optimizer step, s | 6.085 | 1.315 |
| Processed update rows/s | 159.89 | 770.24 |
| Collection decisions/s | 144.98 | 293.83 |
| End-to-end decisions/s | 37.50 | 122.80 |
| Collection / update / other wall share | 25.9 / 71.9 / 2.2% | 41.8 / 52.3 / 5.9% |
| Sampled peak device memory, MiB | 11,561 | 11,333 |

| One synchronized 1,024-row step, smoke seconds | PPO | Replay |
| --- | ---: | ---: |
| Shared trunk, both perspectives when present | 0.1393 | 0.0299 |
| Opponent belief | 0.2171 | 0.1640 |
| Self-belief | 0.2049 | absent |
| Policy/value heads | 0.0039 | 0.0019 |
| Other forward work | 0.0166 | 0.0049 |
| Loss assembly | 0.0237 | 0.0129 |
| Backward | 0.7345 | 0.1965 |
| Gradient clipping / metrics | 0.0048 | 0.0036 |
| Optimizer | 0.0053 | 0.0036 |
| Total | 1.3501 | 0.4174 |

Scope: two PPO updates, 26,330 decisions, 78 optimizer steps and 78,990 processed rows in 214.410 s through the checkpoint. The brief 06 command/config was reused, with four rollout workers and a stop after update two. Replay: random weights, current converted rows from 1,000 games (49,478 rows), 200 complete batches / 204,800 row-visits, 110.687 s including batch building, synchronization and periodic checkpoints; model/data loading excluded. It reproduces `bc.py`'s CUDA autocast/GradScaler branch; the recipe's `mixed_precision: false` does not disable that branch. Partial epoch tails are skipped. Sample timings exclude batch building and need not equal aggregate step averages. Before/after device-memory scopes are the old whole attempt and the new prefix through PPO, respectively.

**Not verified:** learning, strength, steady-state rates, exact runtime-drop recovery, or an isolated causal ablation of the loop fix: CPU counts, PyTorch versions and replay rows differ.

Three successful update-two stacks are in `08-artifacts/profiling-followup/` (`ee058d8`); the original permission-denied attempts are retained. All-attempt GPU use, counting both T4s and overhead, is **≤0.469 GPU-hours**. Recovery checkpoints remain in private notebook outputs; all committed artifacts are text/JSON.
