# Limited learning from play — diagnostic, smoke

**Answer.** The small problem was learned: MaxDamage wins rose from 16% to 100% by
106,889 decisions, +84 points with non-overlapping 95% intervals. Play improved the blueprint
on its eight training teams: SimpleHeuristics rose from 25.5% to 91.25% at 556,907 decisions,
+65.75 points with non-overlapping intervals. Held-out pool B stayed near baseline throughout.
This is the brief’s eight-team learning case. First observed gains cost **12,725 / 37,000 PPO
decisions per ten points**, respectively. Updates dominate time: 86.5% in stage 0, 82.6% in
stage 2. **Every result is smoke; no gate or checkpoint candidate is claimed.**

## Stage 0: throughput

Four CPU cores, 16 GiB, no GPU; Showdown `e440c4a18385274f10c405d0b158b6a962ce6d94`.

| Decisions | Collection | Update, three epochs | End to end |
| ---: | --- | --- | --- |
| 20,336 | 118.82 s; 171.15 decisions/s | 790.12 s; 25.74 unique rows/s | 913.16 s; 22.27 decisions/s |

Optimization alone: 783.40 s, 61,008 processed rows, **77.88 rows/s**. The 64-game profile:
Node round trips/JSON 14.0%, observations 24.0%, acting forward 43.7%, other 18.3%; **zero
calls to either belief head while acting**. Node CPU time (2.55 s) overlaps these categories.
A matched 1,180-decision probe favored four learner threads over one: 60.47 versus 110.25 s.
The authorized scale factor is 22.2699198388/40 = 0.556748: budgets became 334,049 and
556,748; evaluation counts and 100k/200k cadence stayed fixed.

## Stage 1: learned

Random weights; first `pool_a` paste on both sides; MaxDamage only. Each cell is **200 greedy
pinned-Showdown games**, win rate [95% Wilson]; counters reflect completed games/updates.

| Decisions | MaxDamage | SimpleHeuristics |
| ---: | --- | --- |
| 0 | .160 [.116, .217] | .135 [.094, .189] |
| 106,889 | 1.000 [.981, 1.000] | .640 [.571, .703] |
| 204,288 | 1.000 [.981, 1.000] | .905 [.856, .938] |
| 302,039 | 1.000 [.981, 1.000] | .930 [.886, .958] |
| 334,236 | 1.000 [.981, 1.000] | .920 [.874, .950] |

The +84-point gain passes the specified ≥30-point rule. No fallback grid was needed.
Training took 2 h 10 min, **42.81 decisions/s**, 1.387 effective epochs overall because later
updates hit the existing KL stop. Faster throughput therefore reflects less optimization work.

## Stage 2: improved on pool A; no clear held-out gain

Replay access returned HTTP 200. Run A’s 1,000-game/52,983-row blueprint was rebuilt through
epoch eight of its **18-epoch schedule**; `bc_best.pt` is epoch eight, holdout action CE
1.742139. The 2,000-game/104,037-row holdout matched the fixed IDs. PPO used the prescribed
50% SimpleHeuristics / 10% MaxDamage / 40% latest-self mixture. Each cell is **400 greedy
games**, win rate [95% Wilson]; B was never used for PPO training.

| Decisions | A: SimpleHeuristics | A: MaxDamage | B: SimpleHeuristics | B: MaxDamage |
| ---: | --- | --- | --- | --- |
| 0 | .255 [.215, .300] | .720 [.674, .762] | .220 [.182, .263] | .565 [.516, .613] |
| 216,451 | .840 [.801, .873] | .980 [.961, .990] | .2075 [.171, .250] | .5475 [.499, .596] |
| 414,669 | .9675 [.945, .981] | .9875 [.971, .995] | .200 [.164, .242] | .545 [.496, .593] |
| 556,907 | .9125 [.881, .936] | .9725 [.951, .985] | .215 [.178, .258] | .560 [.511, .608] |

Pool A passes the specified ≥10-point rule at every PPO point. Its SimpleHeuristics peak
exceeds the endpoint; gains were not monotonic. All B intervals overlap their baselines.
PPO completed in **6 h 55 min, 22.37 decisions/s**, 2.758 effective epochs, within ten hours.
Neither PPO arm restarted. Every evaluation completed without failures, truncations or draws.

## Stage 3: costs, calculated without new experiments

First observed cost = decisions / (gain in percentage points / 10): 106,889/8.4 and
216,451/5.85. Full-budget costs are **39,790 / 84,701** per ten points. These exclude
blueprint reconstruction (about 66 minutes) and evaluation; earlier gain onset was not measured.

| Part of stage-0 training hour | Minutes/hour | Decisions/s if this part alone were free |
| --- | ---: | ---: |
| Node round trips/JSON | 1.10 | 22.68 |
| Observations | 1.87 | 22.99 |
| Acting forward | 3.41 | 23.61 |
| Other collection | 1.42 | 22.81 |
| Update | 51.92 | 165.28 |
| Outside loop | .28 | 22.37 |

Collection subparts project single-worker profile shares onto four-worker collection time;
free-part rates are Amdahl arithmetic, **not measured speedups**. Stage 1 spent 14.12/45.85
minutes per hour collecting/updating; stage 2 spent 10.39/49.56. Making collection or update
free projects 55.99/181.51 decisions/s for stage 1 and 27.05/128.59 for stage 2. The update
is the first bottleneck. Exact arithmetic: [cost-breakdown.json](04-artifacts/cost-breakdown.json).

## Config changes, fixes and limits

Both configs extend `gen9ou_typed_token_selfplay.yaml`. Changed keys:
`preflight.required=false` (missing reports), `replay_policy.coef=0` (no replay term),
`ppo.init_ckpt=null` (remove unavailable weights); `device=cpu`, `experiment`, `log.dir`,
`env.team_pool` (loader accepts a directory or paste). Under `ppo`: `lr` 3e-4/1e-4,
`total_decisions` as above, `batch_decisions=8000`, `minibatch_size=1024`, `epochs=3`,
`clip=.2`, `target_kl=.03`, `entropy_coef=.01`, `entropy_coef_final=.003`.
`league.scripted`, `scripted_frac`, `latest_frac`, `past_frac` implement each mixture;
scratch sets `league.snapshot_every_decisions=100000`. Blueprint retains 24k refresh, making
its first saved points after 200k/400k be 216,451/414,669. Runtime `ppo.init_ckpt` and
`league.past_ckpts` use the rebuilt blueprint, seeding latest self immediately. Exact inherited
and new values: [config-changes.json](04-artifacts/config-changes.json),
[blueprint-runtime-overrides.json](04-artifacts/blueprint-runtime-overrides.json).

Ordinary 1,024-row updates exceeded memory. New default-off `ppo.activation_checkpointing`,
enabled here, recomputes four blocks; recorded-input checks gave bit-identical updated
parameters, metrics and RNG. Model, losses and update algorithm are unchanged. Tested resume
fixes preserve CSV history and restore warm-started full states without reloading initialization.
**32 focused tests passed; lint clean.** Complete diagnostic CSVs, trainer tails, failed-attempt
logs, profile and evaluation JSONs are in [04-artifacts](04-artifacts/); commands/commits are
in the append-only ledger.

**Deviations / not verified:** scaled budgets; whole-update counters; inherited snapshot cadence;
intermediate evaluations shared CPU with training; a 1.73 s test overlapped stage 0’s first
update. The collection split is approximate. One training seed, GPU throughput, seed variance
and all gates remain unverified.
