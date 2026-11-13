# Limited learning from play — diagnostic, smoke

**Answer (in progress).** The small problem was learned at the first checkpoint: greedy
MaxDamage wins rose from 16.0% to 100.0% at 106,889 decisions, +84 points with non-overlapping
95% intervals. That is 12,725 decisions per ten points at the first measured gain.
By 204,288 decisions MaxDamage remains 100.0%, and SimpleHeuristics is 90.5% on the same team. The prescribed scaled stage 1
budget is still running; improvement of the human-data blueprint is not yet measured.
The isolated throughput check averaged 22.27 decisions/s; updates consumed 86.5% of wall time.
Every result here is smoke. No checkpoint is a candidate and no gate is claimed.

## Stage 0 — completed

Four CPU cores, 16 GiB, no GPU; pinned Showdown `e440c4a18385274f10c405d0b158b6a962ce6d94`.

| Decisions | Collection | Update | Epochs | End to end |
| ---: | --- | --- | ---: | --- |
| 20,336 | 171.15 decisions/s, 118.82 s | 77.88 processed rows/s, 783.40 s optimization; 790.12 s total | 3 | 22.27 decisions/s, 913.16 s wall |

Collection profile: Node round trip/JSON 14.0%, observation building 24.0%, acting forward
43.7%, other 18.3%. Zero calls to either belief head while acting. These are single-worker
cProfile shares; Node CPU (2.55 s) overlaps and is not another wall-time category.
One learner thread was slower on the matched 1,180-decision probe (110.25 vs 60.47 s), so
four were retained. Budget factor 22.2699198388 / 40 = 0.556748: stage 1 334,049,
fallback grid 167,024 each if needed, stage 2 556,748 if eligible. Evaluation game counts and
100k/200k cadence stay unchanged. Raw measurements: [04-artifacts](04-artifacts/).

## Stage 1 — in progress, first point meets the learning rule

Random weights, first `pool_a` paste on both sides, MaxDamage only. Each cell is 200 greedy
pinned-Showdown games, win rate [95% Wilson]. Whole-update boundaries explain the counters.

| Decisions | MaxDamage | SimpleHeuristics |
| ---: | --- | --- |
| 0 | 0.160 [0.116, 0.217] | 0.135 [0.094, 0.189] |
| 106,889 | 1.000 [0.981, 1.000] | 0.640 [0.571, 0.703] |
| 204,288 | 1.000 [0.981, 1.000] | 0.905 [0.856, 0.938] |

No failures, truncations or draws in these evaluations. Intermediate evaluations run on one
CPU thread alongside training. The first-point gain is +84 points; the brief requires at least
+30 and non-overlapping intervals. All trainer diagnostics, including behavior-value explained
variance against GAE returns and unshaped mean episode return, are in `scratch-metrics.csv`.
Later updates stop for the existing KL constraint (0.249–0.636 effective epochs in the last five recorded updates), so their higher decisions/s reflects less optimization work. The full budget continues; the final reading and any later points remain to be recorded.

## Stage 2 — not started

The replay-host precheck returned HTTP 200. Rebuilding run A and the blueprint comparison wait
for stage 1 to finish. No substitute checkpoint has been used.

## Config changes and fixes

Both configs extend `gen9ou_typed_token_selfplay.yaml`. Common changed keys:
`preflight.required=false` (missing preflight reports), `replay_policy.coef=0` (no replay
regularizer), `ppo.init_ckpt=null` (no unavailable starting weights), `device=cpu`,
`experiment`, `log.dir`, and `env.team_pool` (the loader accepts directories and single pastes).
`ppo.batch_decisions=8000`, `minibatch_size=1024`, `epochs=3`, `clip=0.2`, `target_kl=0.03`,
`entropy_coef=0.01`, `entropy_coef_final=0.003`; `total_decisions` and `lr` follow each stage.
`league.latest_frac`, `past_frac`, `scripted_frac`, and `scripted` implement the stated mixtures;
scratch additionally changes `league.snapshot_every_decisions=100000`. Blueprint retains the
inherited 24k latest-self refresh. Exact inherited/new values: `config-changes.json`.

`ppo.activation_checkpointing=true` is the memory fix: ordinary 1,024-row updates exceeded
16 GiB, including a reduced-total diagnostic. Recomputing the four large encoder/head blocks
preserves model, losses, minibatches and optimizer steps. On recorded Showdown observations,
two epochs gave bit-identical parameters, metrics and RNG versus ordinary updates. An explicit
resume now preserves earlier dashboard CSV rows; fresh-run behavior is unchanged. Warm-started runs can also resume with their original init key retained for the config check: the full state supplies weights and optimizer without reloading the initializer. Focused
verification: 32 tests passed, lint clean. OOM attempts and the successful memory probe are
retained; no failed attempt supplied starting weights to stage 1.

**Deviations / not verified.** Budgets scaled as authorized; snapshots occur after whole
updates. A 1.73 s unit test overlapped the first throughput update; remaining timing was idle.
The profile is not a four-process trace. These are one-seed, fixed-team smoke results; no
held-out-team learning, blueprint gain, full-budget endpoint, GPU speed or gate is verified yet.
