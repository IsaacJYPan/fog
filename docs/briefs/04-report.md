# Limited learning from play — diagnostic, smoke

**Answer.** The small problem was learned at the first checkpoint: greedy
MaxDamage wins rose from 16.0% to 100.0% at 106,889 decisions, +84 points with non-overlapping
95% intervals. That is 12,725 decisions per ten points at the first measured gain.
At the final 334,236 decisions MaxDamage remains 100.0%, and SimpleHeuristics is 92.0% on the same team. Stage 1 finished in 2 h 10 min; the blueprint has improved on its eight PPO training teams (+58.5 points against SimpleHeuristics at 216,451 decisions), with no clear held-out gain at that point.
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

## Stage 1 — completed, learned by the stated rule

Random weights, first `pool_a` paste on both sides, MaxDamage only. Each cell is 200 greedy
pinned-Showdown games, win rate [95% Wilson]. Whole-update boundaries explain the counters.

| Decisions | MaxDamage | SimpleHeuristics |
| ---: | --- | --- |
| 0 | 0.160 [0.116, 0.217] | 0.135 [0.094, 0.189] |
| 106,889 | 1.000 [0.981, 1.000] | 0.640 [0.571, 0.703] |
| 204,288 | 1.000 [0.981, 1.000] | 0.905 [0.856, 0.938] |
| 302,039 | 1.000 [0.981, 1.000] | 0.930 [0.886, 0.958] |
| 334,236 | 1.000 [0.981, 1.000] | 0.920 [0.874, 0.950] |

No failures, truncations or draws in these evaluations. Intermediate evaluations run on one
CPU thread alongside training. The first-point gain is +84 points; the brief requires at least
+30 and non-overlapping intervals. All trainer diagnostics, including behavior-value explained
variance against GAE returns and unshaped mean episode return, are in `scratch-metrics.csv`.
Later updates stop for the existing KL constraint: 1.387 effective epochs overall, so higher decisions/s reflects less optimization work. Full-run throughput 42.81 decisions/s; full-budget cost 39,790 decisions per ten points. No fallback grid was needed.

## Stage 2 — completed, improved on pool A

The replay-host precheck returned HTTP 200. The 1,000-game/52,983-row blueprint was rebuilt through epoch eight of run A’s 18-epoch schedule; its best epoch is eight (holdout action CE 1.742139). The 2,000-game/104,037-row holdout matches the fixed IDs exactly. PPO completed 556,907 decisions with the prescribed 50%/10%/40% mixture and scaled 556,748-decision budget; no substitute checkpoint was used. Each cell below is 400 greedy games, win rate [95% Wilson].

| Decisions | Pool A SimpleHeuristics | Pool A MaxDamage | Pool B SimpleHeuristics | Pool B MaxDamage |
| ---: | --- | --- | --- | --- |
| 0 | 0.255 [0.215, 0.300] | 0.720 [0.674, 0.762] | 0.220 [0.182, 0.263] | 0.565 [0.516, 0.613] |
| 216,451 | 0.840 [0.801, 0.873] | 0.980 [0.961, 0.990] | 0.2075 [0.171, 0.250] | 0.5475 [0.499, 0.596] |
| 414,669 | 0.9675 [0.945, 0.981] | 0.9875 [0.971, 0.995] | 0.200 [0.164, 0.242] | 0.545 [0.496, 0.593] |
| 556,907 | 0.9125 [0.881, 0.936] | 0.9725 [0.951, 0.985] | 0.215 [0.178, 0.258] | 0.560 [0.511, 0.608] |

The first point meets the training-pool rule (+58.5 points, non-overlapping intervals): 37,000 PPO decisions per ten points at the first measured gain. Both held-out rates remain within their baseline intervals. At 414,669, pool A has gained 71.25 points against SimpleHeuristics; both held-out intervals still overlap baseline. The final point retains a +65.75-point training-pool gain, with held-out rates near baseline; the 414k peak was higher. PPO completed in 6 h 55 min, without a restart. The 24k league cadence makes the first scored point 216,451 rather than exactly 200k.

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
The profile is not a four-process trace. These are one-seed, fixed-team smoke results; training-seed variance, GPU speed and gates remain unverified.
