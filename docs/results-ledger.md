# Results ledger

Append-only. One row or short entry per measured result, with the command and commit that
produced it. Negative and interrupted results are recorded too. Gates in
[ROADMAP.md](../ROADMAP.md) are judged only from this file.

Row format for matches (printed by the evaluation command once ROADMAP P0.6 exists):

| Date | Candidate | Opponent | Backend | Games | Win rate [95%] | s/decision | Fallbacks | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

No match rows yet. No gate has passed.

## 2026-11-11: baseline measurements

Machine: 4 CPU cores, no GPU, Python 3.13, locked environment (`uv sync --locked --extra dev`).
Showdown could not be built in this session, so every number below is on the Rust engine
backend. Throughput numbers vary by about ±20% between runs on this machine.

Model: `configs/gen9ou_typed_token_selfplay.yaml` builds 5,853,084 parameters. Belief and
self-belief heads are 2.58 million of them; the typed-token trunk 1.20 million; the earlier
backbone 1.41 million; policy and value heads 0.31 million.

### Self-play throughput (random weights vs MaxDamage, engine backend)

Command:

```sh
python -m fog.env.engine_runner --probe --pool data/teams/pool_b \
  --config configs/gen9ou_typed_token_selfplay.yaml --device cpu --games 32 --batches 1,32
```

| Code | Batch 1 decisions/s | Batch 32 decisions/s |
| --- | ---: | ---: |
| Before (`fog` commit `cd7c5e8`) | 29 | 46–54 |
| After memoizing nature/EV inversion (ROADMAP P0.1) | 50 | 117–167 |

For reference, the recorded hosted PPO baseline in `docs/kaggle-candidate-training.md` processed
24,079 decisions in about 23 minutes on two rollout workers: about 17 decisions/s end to end.

### Where the time went (cProfile, batch 16, 16 games, before the fix)

Same command under `python -m cProfile -o probe.prof`, sorted by cumulative time. Total 16.2 s.

| Share | Where |
| ---: | --- |
| 59% | `fog/encoding/entities.py:_own_nature_evs`, called for every own Pokémon at every decision from `encode_entities` and `mirror_recording._own_truth_tokens` |
| 13% | model forward, over half of it in the belief heads |
| 4% | history encoding |
| < 3% | engine step (`EngineBattle.step`) |

After the fix (batch 32, 32 games, total 14.9 s under the profiler):

| Share | Where |
| ---: | --- |
| 40% | `Observation.from_battle`: history encoding 10% (re-parses every past turn each decision), entity encoding 9%, action features 7% |
| 29% | model forward: belief heads 16%, earlier backbone 6%, typed-token trunk 4% |
| 8% | `EngineBattle.view` construction |
| 7% | opponent-view mirror recording |
| 7% | engine step and applying the chosen branch |

Conclusion: rollout speed is bounded by Python observation encoding and by heads the acting
policy does not read, not by the simulator. ROADMAP P0.2–P0.5 follow from this table.

### Rust engine search speed

One opening position built from two teams in `data/teams/pool_b` via `EngineBattle(...).state`,
single thread.

| Call | Result |
| --- | --- |
| `poke_engine.monte_carlo_tree_search(state, duration_ms=50)` | 23,000 iterations (about 415,000/s) |
| `duration_ms=200` | 81,000 iterations (about 380,000/s) |
| `duration_ms=1000` | 347,000 iterations (about 324,000/s) |
| `poke_engine.generate_instructions(state, a1, a2)` called from Python | about 25,000 calls/s |

The result object has `side_one`, `side_two` (lists with `move_choice`, `visits`,
`total_score`) and `total_visits`. This is the basis for the EngineMCTS anchor (ROADMAP P1).

### Checks run for the P0.1 change

- Old and new `_own_nature_evs` return identical results on 3,000 randomized
  (species, level, nature, spread) cases, including perturbed stats, the attribute path and both
  failure paths.
- `pytest tests/test_native_evs.py tests/test_encoding.py tests/test_events.py
  tests/test_engine_runner.py tests/test_player_observation_boundary.py
  tests/test_engine_battle.py tests/test_player_state.py`: 62 passed.
- The full test suite was not run.

## 2026-11-11: throughput after P0.2 and P0.3, and first EngineMCTS smoke run

Same machine and command as the baseline entry, idle machine, two runs each. `fog` commit
`bdcd63a`.

| Code | Batch 1 decisions/s | Batch 32 decisions/s |
| --- | ---: | ---: |
| Baseline (`cd7c5e8`) | 29 | 46–54 |
| + nature/EV memo (`b3efdbe`) | 50 | 117–167 |
| + belief heads skipped when acting (`3712b45`) and history cache (`1719a31`) | 94–95 | 203–212 |

The two later changes were measured together, not separately. Equivalence checks: action
log-probabilities and values bit-identical with belief on and off on real engine batches;
cached and uncached history arrays identical at all 214 decisions of three engine games.
`pytest` on the nine affected test files: 66 passed. The full suite was not run.

Sequential test (`BlockSPRT`, `fog` commit `d3d0bef`), 2,000 simulated matches per row,
four-game blocks, defaults `p0 = 0.50`, `p1 = 0.55`, `alpha = beta = 0.05`:

| True win rate | Wrong decision | Mean blocks used |
| ---: | ---: | ---: |
| 0.50 | 3.75% accept H1 | 162 |
| 0.55 | 3.8% accept H0 | 160 |
| 0.60 | none | 101 |

So a typical comparison needs about 650 games rather than a fixed 2,000–4,000.

EngineMCTS smoke run. **Not a gate result:** approximate engine backend, 40 games, a reduced
budget, and the opponent is the weakest anchor. Command: `python -m fog.eval.engine_mcts_smoke`
with 4 worlds and 2,000 iterations per world, teams from `data/teams/pool_b`, sides and teams
alternated.

| Opponent | Games | Result | Fallbacks | Worlds dropped | s/decision |
| --- | ---: | --- | ---: | ---: | ---: |
| MaxDamage (engine policy) | 40 | 39 wins, 1 draw, 0 losses | 0 | 0 | 0.038 |

Findings from building it:

- The engine build's tree search never proposes a Terastallization move, so EngineMCTS does not
  terastallize. Tracked as ROADMAP P1.2b.
- `data/usage` is empty in the checkout and `priors.json` has no species-usage marginal; the
  sampler uses each species' total teammate co-occurrence mass instead.
- The Showdown adapter has not been run (ROADMAP P1.2a). Gate G1 is open.

## 2026-11-11: engine rebuilt with Terastallization

`fog` commit `cc9e2de`. The engine was built with `--features poke-engine/gen9`; its separate
`terastallization` feature was off, so the built-in search never proposed a Tera move. With the
feature on, the search lists Tera moves until a side has terastallized, type checks and
effectiveness use the Tera type for a terastallized Pokémon, and Tera-type moves get the
60 base-power floor. `uv.lock` did not change.

Checks: eight engine and encoder test files, 59 passed before the rebuild and 60 after (one new
test). Equivalence with pinned Showdown for terastallized Pokémon has **not** been checked;
see ROADMAP P1.2c for a known typing error in sampled worlds.

Tera ablation. **Not a gate result:** engine backend, reduced budget (4 worlds, 2,000
iterations per world), single thread, sides and teams alternated. Command:
`python -m fog.eval.engine_mcts_smoke --tera-ablation`. The Tera-off player drops Tera moves
only at the root.

| Candidate | Opponent | Games | Wins | Draws | Losses |
| --- | --- | ---: | ---: | ---: | ---: |
| EngineMCTS, Tera allowed | EngineMCTS, Tera filtered | 100 | 47 | 11 | 42 |

The difference is within noise at 100 games. The Tera-allowed player terastallized in 62 games.

## 2026-11-11: Showdown rollout path profiled (ROADMAP P0.4)

Pinned Showdown now builds in the work sandbox (`npm ci --ignore-scripts`, then `npm run build`).
Probe: `python -m fog.eval.showdown_rollout_probe`, one process, CPU, random weights, self-play on
`showdown_direct`, teams from `data/teams/pool_b`. Another job was using about two of the four
cores, so the rates are approximate.

Profile before (cProfile, 12 games, 1,666 decisions):

| Share | Where |
| ---: | --- |
| 62% | model forward; belief heads alone 26% |
| 23.5% | observation encoding (action features 9.1%, entities 5.4%, history about 6%) |
| 5.9% | Node round trip and JSON |
| 2.5% | species-key regex normalisation in `fog/env/actions.py` |
| 2.0% | poke-env message parsing |
| 1.4% | mirror recording |

The Showdown simulator step is a small share here too. The Node worker already skips state
serialisation.

Fix: the rollout worker always attaches a private-truth provider, which kept the belief heads on
for every acting forward although nothing reads them at act time. Removing that provider and the
leak probe from the trigger list gives:

| Code | Decisions/s | Games/s |
| --- | ---: | ---: |
| Before | about 42 | 0.24 |
| After | about 77 | 0.45 |

Every recorded fragment, action, log-probability, value and outcome hashes identically before
and after over 8, 12 and 16 games (2,758 decisions in the largest). Not verified by hash: runs
with the leak probe active (`beta > 0`), which this recipe does not use.

After the fix: model forward 48% (trunk and policy), observation encoding 33%, Node round trip
8%. Batching the forward across battles (ROADMAP P0.5) is the next lever. Small leftovers of
2–4% each: regex in `fog/env/actions.py`, scalar clipping and a heuristic call in action
features, the legality mask computed twice.

## 2026-11-11: batched Showdown rollout runner (ROADMAP P0.5, part)

`fog/env/showdown_batch_runner.py` plays N in-memory Showdown battles in one process with one
forward per wave. `python -m fog.eval.showdown_rollout_probe --batch N`, CPU, one torch thread,
another job on about two of four cores, so approximate.

| Batch | Decisions/s |
| --- | ---: |
| existing single-battle path | 87 |
| 1 | 87 |
| 8 | 137 |
| 32 (32 games) | 158 |
| 32 (96 games) | 175 |
| 32 (96 games, 2 torch threads) | 194 |

Batch 1 records exactly what the existing path records (hash-identical over 2, 4 and 32
games). At batches 4 and 8 actions and observations are identical and log-probabilities and
values differ by at most 5e-7. At batch 32: observation building 41%, forward 29%, Node step
and parsing 19%, sampling and recording 10%.

Not yet wired into PPO. That needs: the runner in place of the per-game loop in
`_ShowdownRolloutWorker.run`; per-battle providers for private state and opponent; dropping
per-game global seeding; per-battle heuristic-teacher state; a batch-size setting with more
games per worker; and a GPU-resident actor for a GPU forward. Not checked with the leak probe on.
