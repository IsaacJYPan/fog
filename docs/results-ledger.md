# Results ledger

Append-only. One row or short entry per measured result, with the command and commit that
produced it. Negative and interrupted results are recorded too. Gates in
[ROADMAP.md](../ROADMAP.md) are judged only from this file.

Row format for matches (printed by the evaluation command once ROADMAP P0.6 exists):

| Date | Candidate | Opponent | Backend | Games | Win rate [95%] | s/decision | Fallbacks | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Match rows are in the dated sections below. No gate has passed.

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

## 2026-11-11: EngineMCTS on pinned Showdown, first smoke rows

**Smoke, not gate G1:** 100 games each, quarter budget (4 worlds, 400 ms per decision, one
thread), and `data/teams/pool_b` is the only tracked pool (16 teams, not a held-out split).
The SPRT minimum was lowered to 25 blocks for these runs.

| Date | Candidate | Opponent | Backend | Games | Win rate [95%] | s/decision | Fallbacks | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-11 | EngineMCTS | SimpleHeuristics | showdown_direct | 100 | 0.800 [0.711, 0.867] | 0.434 | 3/2645 | `ef773aa` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k": 4, "budget_ms": 400, "threads": 1}' --opponent simpleheuristics --games 100 --sequential --out runs/smoke/a.json` |
| 2026-11-11 | EngineMCTS | MaxDamage | showdown_direct | 100 | 0.990 [0.946, 0.998] | 0.444 | 0/2399 | `ef773aa` | same with `--opponent maxdamage` |

No illegal action was sent and Showdown rejected none. Twelve worlds were dropped against
SimpleHeuristics. All three fallbacks came from Rest sleep being reconstructed as ordinary
sleep, which makes the engine panic. Other reconstruction faults found: 0 PP rebuilt as full
PP, and Tera still proposed after use (64 results, dropped correctly). See ROADMAP P1.2d.

## 2026-11-12: reconstruction checked against Showdown's own state

New sandbox, same shape as before: 4 CPU cores, no GPU, locked environment, pinned Showdown
built with `npm ci --ignore-scripts` and `npm run build`. Nothing else was running during the
match rows unless a row says so.

Tests at `fog` commit `679f79b`: the fifteen engine, search and rollout test files listed in
`HANDOFF.md` plus `tests/test_state_recon_fixes.py`, 115 passed. At `8f9afc7`: the same files plus
the two new ones, 124 passed, and fifteen further files that import the changed modules
(`test_search_gauntlet`, `test_pb_*`, `test_joint_search`, `test_step_branches`,
`test_train_engine`, `test_distill_v2`, `test_belief_refresh`, `test_value_finetune`,
`test_source_agreement`, `test_pool_id`), 110 passed. The full suite was not run.

### Smoke re-run after the first reconstruction fixes (ROADMAP P1.2d)

**Smoke, not gate G1:** 100 games, quarter budget (4 worlds, 400 ms, one thread),
`data/teams/pool_b` (not a held-out split), SPRT minimum lowered to 25 blocks. The Tera damage
check below ran on another core for a few minutes of this run.

| Date | Candidate | Opponent | Backend | Games | Win rate [95%] | s/decision | Fallbacks | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-12 | EngineMCTS | SimpleHeuristics | showdown_direct | 100, SPRT accept_h1 at 25 blocks | 0.830 [0.745, 0.891] (83-17-0) | 0.447 | 0/2640 | `679f79b` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k": 4, "budget_ms": 400, "threads": 1}' --opponent simpleheuristics --games 100 --sequential --sprt-min-blocks 25 --out runs/smoke/v4_sh.json` |

Against the first smoke row (`ef773aa`): fallbacks 3 to 0, dropped worlds 12 to 0 of 10,312
sampled, Tera entries proposed after the side had used Tera 64 to 0, rejected choices 0. The
three faults in P1.2d are closed. The win rate is inside the earlier interval; 100 games cannot
show a difference of this size.

### Terastallization damage against pinned Showdown (ROADMAP P1.2c)

`python -m fog.eval.tera_damage_check` (`fog` commit `616711c`): 22 one-move cases in a
one-versus-one custom game. Showdown's exact non-critical damage rolls come from
`ShowdownDirectClient.enumerate`; the engine side is `poke_engine.calculate_damage` (maximum
roll) on a state with `terastallized=True` and the original `types` kept, which is what
reconstruction builds, and the `Damage` instruction of a `<move>-tera` turn.

| Case (Great Tusk into Snorlax unless noted) | Showdown rolls | Engine max | Engine Tera turn |
| --- | ---: | ---: | ---: |
| Earthquake, no Tera | 193–228 | 228 | 210 |
| Liquidation, Tera Water (new type) | 165–195 | 195 | 180 |
| Earthquake, Tera Ground (matches an original type) | 258–304 | 304 | 281 |
| Water Gun, Tera Water (60 base-power floor) | 36–43 | 43 | 39 |
| Crawdaunt Adaptability Liquidation, Tera Water | 229–270 | 270 | 249 |
| Earthquake into Snorlax that terastallizes to Steel | 386–456 | 456 | 421 |
| Earthquake, Tera Stellar | 258–304 | 228 | 210 |
| Liquidation, Tera Stellar | 132–156 | 130 | 120 |

20 of 22 cases agree within 1 HP, including the bonus for a new Tera type (1.5x), a kept
original type (1.5x), a matching Tera type (2.0x), Adaptability, the floor (and its priority
exception) and defensive typing in both directions. **The engine does not model Tera Stellar**:
it applies neither the 2.0x on an original type nor the 1.2x on other types. Stellar is legal in
this format; no fix is planned here because the engine is an external dependency. The Pursuit
note in P1.2c (the engine records a Tera during a Pursuit switch for side one regardless of
side) was not exercised.

### Rebuilt engine states against Showdown's state (ROADMAP P1.3)

`python -m fog.scripts.fuzz_recon_showdown --pool data/teams/pool_b --games 30` plays scripted
games (random, SimpleHeuristics and MaxDamage in rotation) on pinned Showdown and, at every
decision, rebuilds the engine state the way `ShowdownEngineMCTS` does, with the opponent's true
sheet as the world, then compares it with Showdown's own battle state. 1,779 positions.

| Field | Positions wrong at `679f79b` | At `8f9afc7` |
| --- | ---: | ---: |
| Any field | 892 (50.1%) | 236 (13.3%) |
| Stealth Rock on either side | 264 | 0 |
| Rain, sun or snow missing | 192 | 0 |
| Weather turns remaining | 33 | 0 |
| Terrain turns remaining | 17 | 0 |
| Quark Drive or Protosynthesis boost missing | about 320 | 0 |
| Binding (partial trap) missing | 18 | 0 |
| Opponent HP of a benched Regenerator (cannot be observed) | 142 | 142 |
| Semi-invulnerable turn of a two-turn move (Phantom Force) | 37 | 37 |
| Locked move (Outrage-style) | 25 | 25 |
| Reflect or Light Screen turns | 24 | 24 |
| Flash Fire boost missing | 13 | 11 |

Causes: poke-env stores the start turn as the value of Stealth Rock and Sticky Web, and the
engine applies them only when the value is exactly 1, so every search since `ef773aa` ignored
existing Stealth Rock and Sticky Web unless they were set on turn 1; poke-env's names for rain,
sun and snow were missing from the weather map; field timers were rebuilt as a fresh five turns;
the boosted-stat volatiles were filtered out. Timers now count `|upkeep|` lines in the public
log, and weather or terrain lasts eight turns when its setter holds the matching item in the
state being built (the searcher's real item, or the sampled world's). A second seed (60 games,
3,873 positions) gives the same 13.3%, of which 329 positions are benched Regenerator HP.

Correction: the message of commit `8f9afc7` says the mismatch rate fell "from 66.6%". That
figure came from an earlier version of the checker that also counted two things that are not
faults (Showdown clears the Tera flag of a fainted Pokémon; Rest sleep was detected wrongly).
The 50.1% above is the same checker on both commits. The same message lists Flash Fire as fixed;
it still differs in 11 positions.

Not fixed, by choice (rare in play or not public): the three volatiles above, screen timers
with Light Clay, the side's last used move (Fake Out and Encore legality, choice locks on the
opponent's side), Wish and Future Sight, the toxic counter.

Engine round-trip (`python -m fog.scripts.fuzz_recon --pool data/teams/pool_b --games 2000
--max-turns 40 --extended`, 3,454 positions): the core fields still match everywhere. The new
`--extended` fields differ in 42.3% of positions, led by the last used move (455) and disabled
flags of the searcher's own moves (266; on Showdown the request overlay supplies these), then
weather turns (196; the engine log has no upkeep marker, so the start turn is used and is off by
one for weather set by a replacement). Running it also showed that on the engine backend a
lead's weather ability set weather with no end; it is now five turns (`8f9afc7`).

## 2026-11-12: smoke after the Showdown-state fixes, and first replay conversion at 3,000 games

Both entries below are **transcribed from the session notes**. The sandbox that produced them was
reclaimed before the report files (`runs/smoke/v5_sh.json`, `runs/smoke/v5_sh.log`, and the data
directories) could be kept, so they cannot be re-read; re-running the commands is the only check.

**Smoke, not gate G1:** 100 games, quarter budget (4 worlds, 400 ms, one thread),
`data/teams/pool_b` (not a held-out split), SPRT minimum lowered to 25 blocks.

| Date | Candidate | Opponent | Backend | Games | Win rate [95%] | s/decision | Fallbacks | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-12 | EngineMCTS | SimpleHeuristics | showdown_direct | 100 | 0.880 [0.802, 0.930] | 0.450 | 0/2404 | `8f9afc7` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k": 4, "budget_ms": 400, "threads": 1}' --opponent simpleheuristics --games 100 --sequential --sprt-min-blocks 25 --out runs/smoke/v5_sh.json` |

No dropped worlds and no rejected choices. The three smoke rows (0.800 at `ef773aa`, 0.830 at
`679f79b`, 0.880 here) are inside each other's intervals; 100 games cannot rank them. Two
default-budget shards (8 worlds, 1600 ms) were started and killed after 4 and 8 games to free the
CPU; they are not results.

Replay import and conversion (ROADMAP P2.1, P2.2; a timing check, no model was trained):
`import_metamon_replays` with rating ≥ 1500 and upload date ≥ 2024-02-01 took 3,000 games from the
first shard of the pinned revision `792a86b5cdbcab9696032e947483b218453a0d4f` in 16 s
(14,141 source rows scanned, 10,406 below the rating floor). `convert_replay_decisions
--imputations 1 --workers 4` turned them into 149,559 decision rows in 3 min 32 s on four cores
(about 14 games/s; 84 MB of shards, about 560 bytes per decision), with 374 rows excluded
(306 called or unaligned moves, 68 forced or unaligned switches).

## 2026-11-12: replay blueprint learning curve at 1k–20k games (ROADMAP P2)

Brief `docs/briefs/01-p2-replay-learning-curve.md`; report `docs/briefs/01-report.md`; every
number below is in a file under `docs/briefs/01-artifacts/`. Machine: 4 vCPU, 15.7 GB, no GPU.
Recipe `configs/replay_blueprint_cpu.yaml` unchanged (from scratch, lr 3e-4 cosine to 3e-5,
batch 256, 6 epochs, early stopping patience 2). Deviations from the roadmap: batch 256 instead of
≥ 1,024; the largest point is 3,000 games instead of 100,000; 10,000 and 20,000 were not run
(projected 6.3 h and over 15 h on this CPU, above the brief's 6-hour limit).

Data (`fog` commit `41cc7b9` for import and conversion). `import_metamon_replays` at revision
`792a86b5cdbcab9696032e947483b218453a0d4f`, rating ≥ 1500, uploaded ≥ 2024-02-01: 22,000
games from the first 2 shards (104,736 source rows scanned) in 34 s. `convert_replay_decisions
--imputations 1 --workers 4`: 22,000 games, 0 quarantined, 1,127,562 decision rows (2,366 called or
unaligned moves and 554 forced or unaligned switches excluded), 624 MB, 28 min 29 s
(12.9 games/s). Chronological split (`split_replay_manifest --chronological-holdout-games 2000
--train-sizes 40 1000 3000 10000 20000`, commit `c895fac`): holdout = the 2,000 newest games,
104,037 rows, uploaded 2026-02-20 to 2026-05-19; training 20,000 games, 1,023,525 rows,
2024-02-01 to 2026-02-20; nested training sets of 1,000 / 3,000 games = 52,983 / 156,597 rows.

Checks. Policy rows (train_1000): 100% `policy_valid`, 99.97% with exact chosen-action features,
100% exact chosen legality (the recipe's `require_exact_*` flags kept); mean 10.45 legal actions
per row (not about 3.5 as the 2026-11-12 entry says), 28.2% switches. Overfit 40 games at
batch 256 × 40 epochs (320 steps): top-1 0.614, below the brief's 0.90 pass mark; action loss
2.286 → 1.055. Diagnostic with `bc.batch_size=32`: top-1 0.988, loss 0.036. The owner chose to
run the curve. Training speed about 112–141 rows/s on CPU, 2.4 GB peak, 5,853,084 parameters.

Holdout scores of `bc_best.pt` with `python -m fog.eval.replay_holdout_report --config
configs/replay_blueprint_cpu.yaml --checkpoint runs/blueprint/g<S>/checkpoints/bc_best.pt
--manifest runs/blueprint/split/holdout_manifest.parquet --out <file>` (`--untrained` for the
zero row; commits `87de3a1`, `abd3c96`, `5442ef5` for the untrained, 1,000 and 3,000 rows as each
JSON records). Training command for each size: `python -m fog.training.bc
--config configs/replay_blueprint_cpu.yaml --set bc.manifest=runs/blueprint/split/train_<S>.parquet
bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet bc.streaming=true
log.dir=runs/blueprint/g<S>`, started at commits `5577c52` (1,000) and `5442ef5` (3,000). No code changed between them; the
commits in between touch only documents.
Diagnostics, not gates.

| Games | Rows | Epochs (best) | Wall | Holdout top-1 | Lift | NLL | Value Brier | ECE | Train top-1 | Train action loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (untrained) | – | – | – | 0.100 | 1.05 | 2.307 | 0.253 | 0.058 | – | – |
| 1,000 | 52,983 | 6 (4) | 55.8 min | 0.328 | 3.44 | 1.841 | 0.264 | 0.131 | 0.366 | 1.731 |
| 3,000 | 156,597 | 6 (4) | 2 h 6 min | 0.413 | 4.32 | 1.635 | 0.229 | 0.032 | 0.449 | 1.523 |
| 10,000 | 514,103 | not run | | | | | | | | |
| 20,000 | 1,023,525 | not run | | | | | | | | |

Reliability of the 3,000-game value head (ten equal bins of predicted win probability; count,
mean predicted, observed): 244, 0.095, 0.102 · 6,725, 0.155, 0.211 · 10,052, 0.252, 0.322 ·
14,239, 0.353, 0.403 · 22,395, 0.454, 0.480 · 23,540, 0.545, 0.542 · 12,170, 0.646, 0.616 ·
8,152, 0.748, 0.711 · 5,833, 0.845, 0.807 · 687, 0.912, 0.897 (bin 0.0–0.1 to 0.9–1.0).

Reading: holdout NLL falls by 0.433 per tenfold data over the two trained points, far above the
P2.4 threshold of 0.02, so more data is the next step. The 1,000 → 10,000 change is not
available. Lift 4.32 at 3,000 games is clearly above chance. At 3,000 games move rows have lift
5.13 and switch rows 2.12; value Brier is 0.248 / 0.235 / 0.204 in the early / middle / late third
of the game.

**Smoke, not gate G2:** 400 games per row, `data/teams/pool_b` (not a held-out team split), the
3,000-game `bc_best.pt` with no search, loaded natively with `--checkpoint-config` (the default
loader widens the checkpoint with posterior features it was not trained with).

| Date | Candidate | Opponent | Backend | Games | Win rate [95%] | s/decision | Fallbacks | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-12 | blueprint | simpleheuristics | showdown_direct | 400 | 0.258 [0.217, 0.303] (103-297-0) | n/a | n/a | `62f16ae` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint blueprint=runs/blueprint/g3000/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --out runs/blueprint/play_t0.json` |
| 2026-11-12 | blueprint | maxdamage | showdown_direct | 400 | 0.623 [0.574, 0.669] (249-151-0) | n/a | n/a | `62f16ae` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint blueprint=runs/blueprint/g3000/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --out runs/blueprint/play_t0.json` |
| 2026-11-12 | blueprint | simpleheuristics | showdown_direct | 400 | 0.223 [0.184, 0.266] (89-311-0) | n/a | n/a | `62f16ae` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint blueprint=runs/blueprint/g3000/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0.5 --out runs/blueprint/play_t0.5.json` |
| 2026-11-12 | blueprint | maxdamage | showdown_direct | 400 | 0.517 [0.469, 0.566] (207-193-0) | n/a | n/a | `62f16ae` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint blueprint=runs/blueprint/g3000/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0.5 --out runs/blueprint/play_t0.5.json` |

These rows do not decide G2. Checkpoints are git-ignored and were not kept; their SHA-256 values
are in the report.

## 2026-11-13: review of the brief 01 learning curve

The branch `work/p2-replay-curve` (head `03cf182`) was reviewed and merged into `work/roadmap`.
Checked: every number in `docs/briefs/01-report.md` against the files in
`docs/briefs/01-artifacts/`; the new tests (22 passed with `tests/test_bc_metrics.py`) and
`ruff check fog tests` at `03cf182`; all curve runs used one recipe; no frozen area was touched.
No new measurement was made in the review. Four readings of the same numbers differ from, or add
to, the section above:

1. **The curve mixes data size with training length.** Every run had six epochs, so 1,000 games
   got 1,242 optimizer steps and 3,000 games got 3,672. At 1,000 games the policy was still
   improving when training ended and was not overfitting (final training action loss 1.731,
   holdout 1.804; top-1 0.366 and 0.344). The overfit check shows the same thing from the other
   side: 320 steps at batch 256 could not memorise 2,020 rows, 2,560 steps at batch 32 could.
   So "−0.43 NLL per tenfold data" is an upper bound on the effect of data, and the P2.4 rule
   cannot be applied to these two points. A run at 1,000 games with the step count of the
   3,000-game run separates the two effects.
2. **The value head has learned little.** Holdout value MSE at 3,000 games is 0.916 at its best
   epoch, against 1.0 for a constant zero (Brier 0.229 against 0.25), and it rises again from
   epoch 4 (1.055 at epoch 6, worse than the constant). The low calibration error (0.032) says
   the predictions are honest, not that they are informative: most sit near 0.5. A game supplies
   one outcome label however many decisions it has, so 3,000 games is a small value dataset.
3. **`bc_best.pt` is not the best policy.** It is chosen on the total loss, which the rising
   value loss turns upward at epoch 4 while the policy loss is still falling.
4. **The stage 3b pass rule in the brief was badly set** (the brief's fault, not the run's): it
   fixed 40 epochs at batch 256, too few steps to memorise anything. The batch-32 diagnostic is
   the valid check, and it passed.

Play strength (0.258 against SimpleHeuristics, smoke) comes from an under-trained policy, so it
does not yet say whether the roadmap's covariate-shift case applies. Gate G2 stays open.
Training speed was 112 to 141 rows/s on four cores; 100,000 games (about 5.2 million rows) for
six epochs is about 70 hours at that rate, so the scale run needs a GPU or a faster trainer.

## 2026-11-12: replay trainer profile, resume, and the data-or-steps runs (ROADMAP P2)

Brief `docs/briefs/02-p2-scale-prep.md`; report `docs/briefs/02-report.md`; branch
`work/p2-scale-prep`, base `f0dc3d1`. Evidence: `docs/briefs/02-artifacts/`.
Executor timestamps are 2026-11-13 UTC; section date uses the owner's local date.

Stage 1 **blocked on the real corpus**: pinned import command in `02-report.md`, output
`stage0-import-output.txt`, failed with proxy tunnel HTTP 403 for `huggingface.co`.
No local shards/splits were present. Profiling invocation
`uv run --no-sync python -m fog.scripts.profile_replay_training --out runs/blueprint/profile-before`
failed on the absent `train_1000.parquet` (`stage1-profile-output.txt`). Before/after step
components, belief share, top-15 profile and batch-only rows/s are **unmeasured**.
No batch optimisation or thread-default change was made. PyTorch threads: 5 intra-op,
5 inter-op; 4-core quota; no GPU (`environment.txt`). The profiler's independent CPU
update-loop check preserves 20 losses exactly on made-up inputs (4 tests, 3.66 s,
`stage1-tests.txt`); this does not substitute for the required replay comparison.

| Run | Games / epochs requested | Steps / wall / rows/s | Last / best NLL and top-1 | Final train NLL | Value MSE best / last | Command / commit / status |
| --- | --- | --- | --- | --- | --- | --- |
| R | 1000 / 6 | unmeasured | unmeasured | unmeasured | unmeasured | brief stage 3 R command, not run: no corpus |
| A | 1000 / 18 | unmeasured | unmeasured | unmeasured | unmeasured | brief stage 3 A command, not run: no corpus |
| S | 1000 / 6 | unmeasured | unmeasured | unmeasured | unmeasured | brief stage 3 S command, not run: no corpus |
| B | 3000 / 18 | unmeasured | unmeasured | unmeasured | unmeasured | brief stage 3 B command, not run: no corpus |

Data versus training length and seed noise **cannot be determined**. No new value or switch
claim, runtime projection, or play-strength result exists. Stages 2 and 4 are in progress;
missing exact fixed holdout IDs and the real runner smoke remain explicit blockers. G2 stays open.

Stage 2 implemented: atomic `bc.resume` epoch state; default-loss or opt-in `action_ce`
selection and stopping; epoch/throughput columns; optional unchanged-default PyTorch thread
setting; additive switch-mass and switch-target diagnostics. Fixed-holdout splitter code is
also implemented, but the genuine IDs are unavailable. Focused commands:
`uv run --no-sync python -m pytest -q -p no:cacheprovider tests/test_bc_resume.py tests/test_bc_metrics.py tests/test_bc_scope.py`
(44 passed, 3.70 s; `stage2-trainer-tests.txt`) and
`uv run --no-sync python -m pytest -q -p no:cacheprovider tests/test_replay_holdout_report.py tests/test_split_replay_manifest.py`
(`stage2-report-split-tests.txt`). Ruff: `uv run --no-sync python -m ruff check fog tests`
(`stage2-ruff.txt`). Four-epoch CPU interruption after epoch two matches numeric metrics,
model, optimizer, scheduler, scaler, RNG and best/early-stop bookkeeping exactly; elapsed-time
columns differ and are excluded. Made-up data checks features, not the replay learning curve.

Final implementation/runner validation at code commit
`ddd83638f8068a85c740eb90029e8201cd98279c` (full commands in
`02-artifacts/test-commands.txt`): 81 focused tests passed in 15.40 s, plus the last
added hard-interruption timing test passed in 1.63 s; `ruff check fog tests` clean
(`focused-tests.txt`, `hard-interruption-test.txt`, `ruff.txt`). The existing
`cache_dir` warning is caused by the requested disabled pytest cache provider.
Original `f0dc3d1` versus the changed default trainer on the generated full-model
fixture: all 24 training losses and all final state tensors exactly identical
(`stage2-default-equivalence.json`, script `check_default_updates.py`, trained at
`00a2d79`). This supplements, and does not replace, the blocked run R comparison.

Stage 4 **partial**: fixed-ID splitting and scale runner implemented, hosted-notebook
instructions in `docs/replay-blueprint-scale-run.md` explicitly not GPU-verified.
No exact fixed holdout ID file could be recovered, so default scale launch refuses
before importing. After data recovery, recover those IDs from the verified brief 01
split and complete stage 1/R/A/S/B before making data/training-length claims.

The genuine 600-game smoke was blocked. A **generated-data engineering smoke**
actually ran all scripts and killed/restarted the runner on the same command at
`ddd8363`: `uv run --no-sync python -m fog.scripts.run_replay_blueprint --games 600
--work-dir runs/blueprint_runner_fixture_final --device cpu --batch-size 256
--epochs 2 --chronological-holdout-games 100 --parquet
runs/blueprint_runner_fixture_source/synthetic-public-replays.parquet`.
600 generated two-turn protocols, 2,400 rows, train 500 games/2,000 rows, holdout
100 games/400 rows. First attempt 16.890 s, SIGTERM exit143 after epoch one; same
command 15.349 s, exit0, resumed epoch two with CSV epochs `[0,1,2]`. Per-stage walls:
import 0.210 s; convert 3.232 s; split 1.067 s; training attempts combined 24.157 s;
best/untrained CPU reports 1.620/1.656 s. Sources:
`02-artifacts/runner-smoke/interruption-test.json`, `runner-smoke/results/run-summary.txt`,
output/metrics/report files beside them, and reproducible fixture scripts in the
artifacts directory. Fixture timings/metrics are not real-data performance or learning
results. `--revision` on this offline import labels the exercised importer interface,
not genuine corpus provenance. Checkpoint paths/sizes/hashes: `02-artifacts/checkpoints.json`.

Stage 5 **not run**: no replay-trained checkpoint available; no new smoke play rows,
win-rate intervals or switch-play claim. P2.3–P2.5 and G2 remain open. The final report
records all blocked work without substituting synthetic measurements for it.

## 2026-11-13: review of brief 02, and the fixed holdout ids

The branch `work/p2-scale-prep` (head `773e2ef`) was reviewed and merged into `work/roadmap`.
It holds code only: the worker's machine was refused by the dataset host, so the profile, the
four step-matched runs, the real-data runner test and the play smoke of brief 02 **were not run**
and are carried into brief 03. Checked in review at `773e2ef`: the seven test files covering the
change (82 passed in 98 s), `ruff check fog tests` clean, and the trainer diff read line by line
(new options default to off; `metrics.csv` gains two timing columns and the start-up line prints
the thread count).

The holdout id list the worker could not produce was made in review on a machine with dataset
access and is now tracked as `data/replay_splits/p2_holdout_2000_ids.txt`:

- `python -m fog.scripts.import_metamon_replays --out data/metamon_gen9ou_r1500 --revision 792a86b5cdbcab9696032e947483b218453a0d4f --min-rating 1500 --after 2024-02-01 --max-games 22000 --max-shards 4 --max-download-mb 600`
  gave 22,000 games with the same rejection counts as brief 01 (77,225 below rating, 4,176 invalid
  metadata, 495 outside dates, 840 other format). The first attempt failed on a reset connection
  during a shard download; the retry completed.
- The 2,000 games with the largest `uploadtime` (ties by id; there was no tie at the boundary)
  were uploaded 2026-02-20 04:17 to 2026-05-19 04:52 UTC, and the newest remaining game
  2026-02-20 02:05, as in `docs/briefs/01-artifacts/stage2-split-summary.txt`.
- Converting only those 2,000 games (`convert_replay_decisions --imputations 1 --workers 4`)
  gave 104,037 accepted rows, the brief 01 holdout row count.

## 2026-11-13: limited learning-from-play check (diagnostic, smoke)

Brief `docs/briefs/04-limited-rl-check.md`; branch `work/rl-check`. All numbers in this
section are **smoke**, no gate is claimed, and no checkpoint is a candidate. Four CPU cores,
16 GiB limit, Python 3.12.14, no GPU; pinned Showdown
`e440c4a18385274f10c405d0b158b6a962ce6d94`. Artifacts: `docs/briefs/04-artifacts/`.

Stage 0, code `c7476c95b14701cb2d57d1df68579ef3ef8b7f6a`:
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run --no-sync python -m fog.train --config
configs/rl_check_scratch.yaml --trainer ppo --set ppo.total_decisions=20000
log.dir=runs/rl_check/stage0`. Random weights, first `pool_a` team on both sides,
MaxDamage only, four process workers with two actor threads each. Whole games/batches gave
20,336 learner decisions. Collection 118.818 s (171.153 decisions/s); update 790.123 s
(25.738 unique rows/s); PPO optimization 783.404 s for 61,008 processed rows
(77.876 rows/s, exactly three epochs). Wall 913.160 s, 22.269 decisions/s end to end.
`stage0-metrics.csv`, `stage0-summary.json`, `stage0-tail.txt`. One 1.73 s dashboard unit test
ran during the first update; otherwise idle, timings approximate.

Collection profile, `PYTHONPATH=. uv run --no-sync python
 docs/briefs/04-artifacts/profile_collection.py` at `b33fe70`: 64 games, 1,563 learner
 decisions, 16.508 s under cProfile. Node round trip including JSON 14.0%, observation
 construction 24.0%, model forward 43.7%, other 18.3%. Node CPU separately 2.550 s;
 it is not an additive wall-time category. Both belief-head forward counters were zero while
 acting. Single-process profile of the same worker path, not a four-process wall trace.
 `stage0-profile.json` and `stage0-profile.txt` (top 30 cumulative entries).

Failed setup attempts, before any completed PPO update: the original 20k command at `9bea817`
was OOM-killed twice (55.602 / 53.663 s; `stage0-oom*-tail.txt`). A 1,024-decision diagnostic
also exited 137, as did an uncommitted whole-forward checkpointing experiment
(`checkpointing-whole-forward-oom.txt`). Separate checkpointing of the backbone, shared latent,
belief and self-belief blocks resolved the peak: the memory-fix smoke completed 1,180 decisions
in 60.470 s (`memory-fix-smoke-*`). No initial checkpoint existed to resume these failed attempts;
all restarted from random weights, separately from stage 1. This changes activation storage,
not the model, losses, batches or optimizer steps. Eight recorded Showdown observations, two
epochs: parameters, metrics and RNG bit-identical to ordinary updates at four learner threads
(`check_checkpointing.py`, `checkpointing-equivalence.json`). Focused verification: 31 tests
passed, lint clean (`focused-tests.txt`, `ruff.txt`). The dashboard fix preserves existing rows
only for an explicit resume; fresh-run behavior stays unchanged.

Stage 0 thread check (smoke), `a6c73bc`: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run
--no-sync python -m fog.train --config configs/rl_check_scratch.yaml --trainer ppo --set
ppo.total_decisions=1024 log.dir=runs/rl_check/thread1_smoke`, 1,180 decisions, wall
110.250 s versus 60.470 s for the corresponding four-thread memory-fix smoke. Four learner
threads retained (`thread-comparison.json`, `thread1-smoke-*`). No further throughput tuning.
The full-run wall rate above rounds to **22.270** decisions/s; the exact value is
22.2699198388. Proportional budget factor = 22.2699198388 / 40 = 0.556747996;
stage 1 budget 334,049, fallback-grid budget 167,024 each, stage 2 budget 556,748 if eligible.
The normal 100k / 200k evaluation cadence and game counts stay unchanged; report the actual
whole-update checkpoint counters, and also score the final checkpoint. Stage 1 starts from
fresh random weights; neither throughput checkpoint is used as its start.

Stage 1 untrained baselines (smoke), greedy, first `pool_a` team on both sides, matched
four-game blocks. Both rows completed all 200 games with zero failures, truncations or draws.

| Decisions | Opponent | Games | Win rate [95% Wilson] | Commit | JSON | Command |
| ---: | --- | ---: | --- | --- | --- | --- |
| 0 | maxdamage (smoke) | 200 | 0.160 [0.116, 0.217] | `2793b3087b54bccd915831358f813b3f3e070198` | `scratch-0-maxdamage.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/untrained.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent maxdamage --out docs/briefs/04-artifacts/scratch-0-maxdamage.json --resume` |
| 0 | simpleheuristics (smoke) | 200 | 0.135 [0.094, 0.189] | `2793b3087b54bccd915831358f813b3f3e070198` | `scratch-0-simpleheuristics.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/untrained.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent simpleheuristics --out docs/briefs/04-artifacts/scratch-0-simpleheuristics.json --resume` |

Stage 1 training uses the logged CLI in `commands.jsonl`. Intermediate snapshot evaluations
run concurrently on one CPU thread; only stage 0 is used for the isolated cost breakdown.
The final controller skips snapshots already scored by the watcher. No second training job runs.

Stage 1 first checkpoint, **smoke**, still training toward the 334,049-decision budget.
The first whole-update boundary after 100k is 106,889. Both evaluations completed 200 games,
zero failures, draws or truncations.

| Decisions | Opponent | Games | Win rate [95% Wilson] | Commit | JSON | Command |
| ---: | --- | ---: | --- | --- | --- | --- |
| 106889 | maxdamage (smoke) | 200 | 1.000 [0.981, 1.000] | `9e29e1d0d3f502ceaa5bfa0cc2de06136e96b74c` | `scratch-106889-maxdamage.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/checkpoints/ppo_106889.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent maxdamage --out docs/briefs/04-artifacts/scratch-106889-maxdamage.json --resume` |
| 106889 | simpleheuristics (smoke) | 200 | 0.640 [0.571, 0.703] | `9e29e1d0d3f502ceaa5bfa0cc2de06136e96b74c` | `scratch-106889-simpleheuristics.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/checkpoints/ppo_106889.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent simpleheuristics --out docs/briefs/04-artifacts/scratch-106889-simpleheuristics.json --resume` |

First-point reading: MaxDamage 0.160 [0.116, 0.217] → 1.000 [0.981, 1.000],
+84 percentage points with non-overlapping 95% intervals, satisfying the brief’s limited
learning criterion at this checkpoint. First observed cost = 106889 / (84 / 10) =
12,724.881 decisions per ten points. SimpleHeuristics 0.135 → 0.640. No held-out-team or
gate claim follows. `scratch-metrics.csv` and `scratch-tail.txt` are an in-progress copy;
the final run will extend them. Stage 2 has not started.

Stage 1 second checkpoint, **smoke**, 204,288 decisions. Still running toward 334,049.
Each row has zero failures, draws and truncations.

| Decisions | Opponent | Games | Win rate [95% Wilson] | Commit | JSON | Command |
| ---: | --- | ---: | --- | --- | --- | --- |
| 204288 | maxdamage (smoke) | 200 | 1.000 [0.981, 1.000] | `f7c3318301140da3ec21146b3e223c0ddc691f5a` | `scratch-204288-maxdamage.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/checkpoints/ppo_204288.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent maxdamage --out docs/briefs/04-artifacts/scratch-204288-maxdamage.json --resume` |
| 204288 | simpleheuristics (smoke) | 200 | 0.905 [0.856, 0.938] | `f7c3318301140da3ec21146b3e223c0ddc691f5a` | `scratch-204288-simpleheuristics.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/checkpoints/ppo_204288.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent simpleheuristics --out docs/briefs/04-artifacts/scratch-204288-simpleheuristics.json --resume` |

MaxDamage stays at 200/200; SimpleHeuristics rises to 181/200. Later updates stop for
the existing target-KL constraint (`stopped_for_kl=1`); the five updates ending at 204,288
process 0.249–0.636 effective epochs, rather than three. Optimization throughput remains
70–80 processed rows/s; the increased decisions/s reflects less optimization work, not a code
speed-up. These actual counts are retained in `scratch-metrics.csv`. The first passing point
took 4,612.136 s of training-loop collection plus updates (sum through 106,889 in that CSV).

Stage 2 preparation found a second reproducible resume blocker: a warm-started run could not
use `ppo.resume_ckpt` while retaining `ppo.init_ckpt`; removing the init key instead fails the
saved configuration digest. The regression test failed before the fix with the explicit
mutual-exclusion error (`warm-resume-regression.txt`). Full resume now skips the initializer
load while retaining its config key, then restores the saved weights and optimizer. Test:
a warm-started model is updated, saved, its initializer deleted, and its parameters and optimizer
moments restored exactly. Ordinary initialization and PPO update math are unchanged. Updated
focused verification: **32 passed**, lint clean (`focused-tests.txt`, `ruff.txt`). The running
scratch job was not restarted or altered by this fix.

Stage 1 completed, **smoke**, uninterrupted: 334,236 decisions versus scaled budget 334,049
(whole-game overshoot), 41 updates, no restarts. Training command and its starting commit
`2793b3087b54bccd915831358f813b3f3e070198` are in `scratch-summary.json` / `commands.jsonl`.

| Decisions | Opponent | Games | Win rate [95% Wilson] | Commit | JSON | Command |
| ---: | --- | ---: | --- | --- | --- | --- |
| 302039 | maxdamage (smoke) | 200 | 1.000 [0.981, 1.000] | `ad94f0d54e54e01c5a66b743e6a8c1646bad44cc` | `scratch-302039-maxdamage.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/checkpoints/ppo_302039.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent maxdamage --out docs/briefs/04-artifacts/scratch-302039-maxdamage.json --resume` |
| 302039 | simpleheuristics (smoke) | 200 | 0.930 [0.886, 0.958] | `ad94f0d54e54e01c5a66b743e6a8c1646bad44cc` | `scratch-302039-simpleheuristics.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/checkpoints/ppo_302039.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent simpleheuristics --out docs/briefs/04-artifacts/scratch-302039-simpleheuristics.json --resume` |
| 334236 | maxdamage (smoke) | 200 | 1.000 [0.981, 1.000] | `ad94f0d54e54e01c5a66b743e6a8c1646bad44cc` | `scratch-334236-maxdamage.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/checkpoints/ppo_final.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent maxdamage --out docs/briefs/04-artifacts/scratch-334236-maxdamage.json --resume` |
| 334236 | simpleheuristics (smoke) | 200 | 0.920 [0.874, 0.950] | `ad94f0d54e54e01c5a66b743e6a8c1646bad44cc` | `scratch-334236-simpleheuristics.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a/core_v1_offensive.txt --config configs/rl_check_scratch.yaml --checkpoint scratch=runs/rl_check/scratch/checkpoints/ppo_final.pt --checkpoint-config scratch=configs/rl_check_scratch.yaml --device cpu --temperature 0 --games 200 --seed 0 --opponent simpleheuristics --out docs/briefs/04-artifacts/scratch-334236-simpleheuristics.json --resume` |

Final reading: **learned** by the stated fixed-team rule, MaxDamage 16% → 100%,
+84 points with non-overlapping intervals. No fallback grid is run. SimpleHeuristics
13.5% → 92.0% on the same team. First observed cost 12,724.881 decisions per ten points;
full-budget cost 39790.000. Wall 7807.432 s
(42.810 decisions/s), collection 1837.584 s,
update 5966.054 s; 463589 optimization rows,
1.387 effective epochs overall. Later updates stop for target KL.
Endpoint entropy / measured policy KL / clip fraction / value loss / value explained variance /
mean episode return: 0.45304 / 0.01359 / 0.04069 / 0.00677 / -0.05549 / 0.99457.
All per-update diagnostics and the last 60 log lines are retained. Stage 2 is now eligible;
no gate or held-out-team claim is made.

Stage 2 initializer reconstructed, **smoke / diagnostics**, at code
`120bd43f90949c64b3efeda7b420615ae8a28d12`. Replay-host precheck returned 200.
The pinned 22,000-game import, single-imputation conversion, chronological split and fixed-ID
check succeeded: training 1,000 games/52,983 rows; holdout 2,000 games/104,037 rows, exact
fixed IDs. The 3,000- and 20,000-game manifests also match the brief-03 counts.
`blueprint-input-check.json` and import/convert/split tail files retain the measurements.

| Step (smoke) | Seconds | Command |
| --- | ---: | --- |
| import | 206.437 | `uv run --no-sync python -m fog.scripts.import_metamon_replays --out data/metamon_gen9ou_r1500 --revision 792a86b5cdbcab9696032e947483b218453a0d4f --min-rating 1500 --after 2024-02-01 --max-games 22000 --max-shards 4 --max-download-mb 600` |
| convert | 887.392 | `uv run --no-sync python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou_r1500 --out data/metamon_gen9ou_r1500_decisions --imputations 1 --workers 4` |
| split | 2.283 | `uv run --no-sync python -m fog.scripts.split_replay_manifest --manifest data/metamon_gen9ou_r1500_decisions/manifest.parquet --out runs/blueprint/split --chronological-holdout-games 2000 --train-sizes 40 1000 3000 10000 20000` |
| bc | 2835.644 | `uv run --no-sync python -m fog.train --config configs/replay_blueprint_cpu.yaml --trainer bc --set bc.manifest=runs/blueprint/split/train_1000.parquet bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet bc.epochs=18 bc.streaming=true bc.early_stopping_patience=0 bc.selection_metric=action_ce bc.resume=true log.dir=runs/blueprint/g1000e18` |

Run A retained its 18-epoch schedule and was intentionally stopped after epoch eight
(exit 143 after the atomic epoch-eight state was saved, not a failed training attempt).
`bc_best.pt` is epoch eight, holdout action CE 1.742139; this is a diagnostic, not a strength
claim. `blueprint-build-metrics.csv`, `blueprint-build-tail.txt`, and `blueprint-build-summary.json`
are retained. No training restart occurred. The environment was later resumed with these
completed artifacts and checkpoint intact; stage-2 PPO had not started.

The stage-2 launcher seeds `league.past_ckpts` with that same initializer as well as setting
`ppo.init_ckpt`, so latest-self play is available immediately. Five SimpleHeuristics entries
and one MaxDamage entry within scripted fraction 0.6 implement 50%/10%; latest self is 40%.
The inherited 24k snapshot cadence is retained. The first saved boundary after each 200k is
evaluated, plus the final checkpoint, with 400 games per opponent per pool. A persisted
ten-hour deadline bounds the PPO job; the scaled decision budget remains 556,748.

Stage 2 untrained-from-play blueprint baselines, **smoke**, 400 greedy pinned-Showdown
games per row, zero failures, truncations and draws. Pool B is held out from PPO training.

| Decisions | Pool | Opponent | Games | Win rate [95% Wilson] | Commit | JSON | Command |
| ---: | --- | --- | ---: | --- | --- | --- | --- |
| 0 | pool_a | simpleheuristics (smoke) | 400 | 0.255 [0.215, 0.300] | `66cd49c16008537e92f068b729152b2b26eaf7a7` | `blueprint-0-pool_a.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/blueprint/g1000e18/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-0-pool_a.json` |
| 0 | pool_a | maxdamage (smoke) | 400 | 0.720 [0.674, 0.762] | `66cd49c16008537e92f068b729152b2b26eaf7a7` | `blueprint-0-pool_a.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/blueprint/g1000e18/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-0-pool_a.json` |
| 0 | pool_b | simpleheuristics (smoke) | 400 | 0.220 [0.182, 0.263] | `66cd49c16008537e92f068b729152b2b26eaf7a7` | `blueprint-0-pool_b.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/blueprint/g1000e18/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-0-pool_b.json` |
| 0 | pool_b | maxdamage (smoke) | 400 | 0.565 [0.516, 0.613] | `66cd49c16008537e92f068b729152b2b26eaf7a7` | `blueprint-0-pool_b.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/blueprint/g1000e18/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-0-pool_b.json` |

Stage-2 PPO started from the rebuilt `bc_best.pt`, using the generated overrides in
`blueprint-runtime-overrides.json` and command `uv run --no-sync python -m fog.train
--config runs/rl_check/blueprint/config.yaml --trainer ppo`. Budget 556,748 decisions,
with the ten-hour deadline and full-state resume described above. The starting comparison
for the improvement rule is SimpleHeuristics on pool A: 25.5% [21.5%, 30.0%].

Stage-2 startup measurement, **smoke**, through 24,751 completed learner decisions:
all three PPO epochs ran in each of the first three updates; value explained variance
0.11878 → 0.21822 → 0.37202. First snapshot `ppo_24751.pt` is saved with its full-state
sidecar. The rollout schedule contains both the original blueprint and that PPO snapshot
as self-play opponents, verifying the initial league seed and inherited 24k refresh.
`blueprint-startup-check.json` contains scheduled-game counts (including the current batch),
not a win-rate claim. `blueprint-metrics.csv` and `blueprint-tail.txt` are in-progress copies.
The first scored PPO point remains the saved boundary after 200k.

Stage 2 first scored PPO point, **smoke**, 216,451 decisions. The 192k league snapshot
landed at 199,807, below 200k; the next inherited 24k boundary supplies this point.
This preserves latest-self refresh timing. Each row completed 400 games, with zero failures,
truncations and draws. Training continues toward 556,748.

| Decisions | Pool | Opponent | Games | Win rate [95% Wilson] | Commit | JSON | Command |
| ---: | --- | --- | ---: | --- | --- | --- | --- |
| 216451 | pool_a | simpleheuristics (smoke) | 400 | 0.8400 [0.801, 0.873] | `6103abf2e4ff2100ec72ebf857d24ef6f0701695` | `blueprint-216451-pool_a.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_216451.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-216451-pool_a.json` |
| 216451 | pool_a | maxdamage (smoke) | 400 | 0.9800 [0.961, 0.990] | `6103abf2e4ff2100ec72ebf857d24ef6f0701695` | `blueprint-216451-pool_a.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_216451.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-216451-pool_a.json` |
| 216451 | pool_b | simpleheuristics (smoke) | 400 | 0.2075 [0.171, 0.250] | `6103abf2e4ff2100ec72ebf857d24ef6f0701695` | `blueprint-216451-pool_b.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_216451.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-216451-pool_b.json` |
| 216451 | pool_b | maxdamage (smoke) | 400 | 0.5475 [0.499, 0.596] | `6103abf2e4ff2100ec72ebf857d24ef6f0701695` | `blueprint-216451-pool_b.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_216451.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-216451-pool_b.json` |

First-point reading: **improved on pool A** by the brief’s rule. SimpleHeuristics
25.5% → 84.0%, +58.5 points, non-overlapping intervals; first observed cost
37000.171 PPO decisions per ten points. MaxDamage 72.0% → 98.0%.
On held-out pool B, SimpleHeuristics 22.0% → 20.75%, MaxDamage 56.5% → 54.75%;
intervals overlap their baselines, so no clear transfer is observed at this point. This is
the brief’s eight-team learning case, pending the remaining curve. No gate is claimed.

Stage 2 second scored point, **smoke**, 414,669 decisions, the first saved inherited
24k snapshot after 400k. All four comparisons completed with zero failures, truncations or
draws. The PPO run continues unchanged toward 556,748; no restart has occurred.

| Decisions | Pool | Opponent | Games | Win rate [95% Wilson] | Commit | JSON | Command |
| ---: | --- | --- | ---: | --- | --- | --- | --- |
| 414669 | pool_a | simpleheuristics (smoke) | 400 | 0.9675 [0.945, 0.981] | `2ca541e85678907ae8de8fffede6cbd127d62ba9` | `blueprint-414669-pool_a.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_414669.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-414669-pool_a.json` |
| 414669 | pool_a | maxdamage (smoke) | 400 | 0.9875 [0.971, 0.995] | `2ca541e85678907ae8de8fffede6cbd127d62ba9` | `blueprint-414669-pool_a.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_414669.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-414669-pool_a.json` |
| 414669 | pool_b | simpleheuristics (smoke) | 400 | 0.2000 [0.164, 0.242] | `2ca541e85678907ae8de8fffede6cbd127d62ba9` | `blueprint-414669-pool_b.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_414669.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-414669-pool_b.json` |
| 414669 | pool_b | maxdamage (smoke) | 400 | 0.5450 [0.496, 0.593] | `2ca541e85678907ae8de8fffede6cbd127d62ba9` | `blueprint-414669-pool_b.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_414669.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-414669-pool_b.json` |

Pool A SimpleHeuristics 25.5% → 96.75%, +71.25 points with non-overlapping intervals;
MaxDamage 72.0% → 98.75%. Pool B SimpleHeuristics 22.0% → 20.0% and MaxDamage
56.5% → 54.5%; both intervals overlap their baselines. Thus the training-pool improvement
persists without a clear held-out gain at this second point. This is still the brief's
eight-team learning case, with the final evaluation pending; no gate is claimed.

Stage 2 complete, **smoke**, 556,907 decisions versus scaled budget 556,748
(whole-game overshoot), 68 updates, zero PPO restarts. The training process exited 0
within the ten-hour cap. Its command `uv run --no-sync python -m fog.train --config runs/rl_check/blueprint/config.yaml --trainer ppo`, starting code `66cd49c16008537e92f068b729152b2b26eaf7a7`,
ran for 24,898.840 s (6 h 54 min 59 s including controller polling), 22.367 decisions/s.
The environment connection reconnected several times; the same learner/controller PIDs and
uninterrupted counters persisted. No job was relaunched. `blueprint-summary.json`, the complete
`blueprint-metrics.csv`, last 60 trainer lines and every evaluation JSON are retained.

| Decisions | Pool | Opponent | Games | Win rate [95% Wilson] | Commit | JSON | Command |
| ---: | --- | --- | ---: | --- | --- | --- | --- |
| 556907 | pool_a | simpleheuristics (smoke) | 400 | 0.9125 [0.881, 0.936] | `74da7385d50fd36880f8ea0eb262c68ac6cf33d5` | `blueprint-556907-pool_a.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_final.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-556907-pool_a.json` |
| 556907 | pool_a | maxdamage (smoke) | 400 | 0.9725 [0.951, 0.985] | `74da7385d50fd36880f8ea0eb262c68ac6cf33d5` | `blueprint-556907-pool_a.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_final.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-556907-pool_a.json` |
| 556907 | pool_b | simpleheuristics (smoke) | 400 | 0.2150 [0.178, 0.258] | `74da7385d50fd36880f8ea0eb262c68ac6cf33d5` | `blueprint-556907-pool_b.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_final.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-556907-pool_b.json` |
| 556907 | pool_b | maxdamage (smoke) | 400 | 0.5600 [0.511, 0.608] | `74da7385d50fd36880f8ea0eb262c68ac6cf33d5` | `blueprint-556907-pool_b.json` | `uv run --no-sync python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --config configs/rl_check_blueprint.yaml --checkpoint blueprint=runs/rl_check/blueprint/checkpoints/ppo_final.pt --checkpoint-config blueprint=configs/rl_check_blueprint.yaml --device cpu --temperature 0 --games 400 --seed 0 --opponent simpleheuristics --opponent maxdamage --resume --out docs/briefs/04-artifacts/blueprint-556907-pool_b.json` |

Final reading: **improved on pool A** by the stated rule, SimpleHeuristics 25.5% →
91.25%, +65.75 points with non-overlapping 95% intervals; MaxDamage 72.0% → 97.25%.
The SimpleHeuristics endpoint is below the 96.75% intermediate point, so improvement was
not monotonic. On held-out pool B, SimpleHeuristics 22.0% → 21.5% and MaxDamage
56.5% → 56.0%, both intervals overlapping baseline. There is **no clear held-out gain**
at any measured point. This is the brief's eight-team learning case, not evidence of a
broader game-playing improvement. All comparisons are smoke; no gate or candidate is claimed.

Totals: collection 4,310.608 s, update 20,567.948 s, optimization 20,383.826 s;
1,535,686 processed optimizer rows, 2.7575 effective epochs overall. Intermediate evaluations
shared the four-CPU allocation with PPO; their contention is included in measured wall time.
Final entropy / policy KL / clip fraction / value loss / value explained variance /
mean episode return: 0.71365 / 0.01315 / 0.07823 / 0.24381 / 0.52057 / 0.42500.
The no-new-runs cost analysis follows next.

Stage 3 completed, **smoke / arithmetic only, no new experiments**. Command
`UV_CACHE_DIR=/tmp/fog-uv-cache uv run --no-sync python docs/briefs/04-artifacts/summarize_costs.py`,
code `4790766638d0baa37e62d46102a0bfcc4aed01e0`; output `cost-breakdown.json`.

| Learned comparison (smoke) | First observed gain | Decisions per ten points at first measured gain | Full-budget cost per ten points |
| --- | --- | ---: | ---: |
| Stage 1, same-team MaxDamage | +84 points at 106,889 | 12,724.881 | 39,790.000 |
| Stage 2, pool A SimpleHeuristics | +58.5 points at 216,451 | 37,000.171 | 84,700.684 |

Cost is learner PPO decisions / (gain in percentage points / 10). First measured points
bound when improvement was observed, not when it first occurred. Stage-2 full-budget gain
is +65.75 points at 556,907; the intermediate +71.25-point gain cost 58,199.158 per ten
points. These costs exclude blueprint reconstruction (~66 minutes from the recorded build
steps) and evaluation. No held-out cost per gain is claimed because no clear gain was measured.

Stage-0 hour split projects the measured single-worker collection profile shares onto its
118.81777 s of four-worker collection wall time. It is approximate, not a four-process trace.
Node CPU (2.55 s) overlaps the Node round-trip/JSON and other wall categories, so is not added.
The hypothetical rates use N/(wall - component); they are Amdahl arithmetic, not benchmarks.

| Component (smoke projection) | Minutes per training hour | End-to-end decisions/s if this part alone were free |
| --- | ---: | ---: |
| node_roundtrip_and_json | 1.096067 | 22.684312 |
| observation_building | 1.874337 | 22.988042 |
| acting_forward | 3.412785 | 23.613023 |
| other_collection | 1.423840 | 22.811246 |
| update | 51.915719 | 165.283130 |
| outside_training_loop | 0.277252 | 22.373304 |

Measured end-to-end rates: stage 0 22.269920, stage 1 42.809979, stage 2 22.366785
learner decisions/s. Stage-1 collection/update minutes per hour: 14.121809/45.849035;
stage 2: 10.387491/49.563631. If all collection alone were free, stage 1 projects
55.987359 decisions/s and stage 2 27.049773; if the update alone were free,
181.514038 and 128.589466. The update dominates in all three runs.
Optimization throughput is 77.875567 / 79.051466 / 75.338457 processed rows/s, with
3 / 1.387011 / 2.757527 effective epochs. The stage-1 rate increase is principally fewer
KL-limited optimizer rows, not a new implementation speedup. Intermediate evaluations shared
CPU with training; use stage 0 for the approximately isolated collection breakdown.

Final report `docs/briefs/04-report.md` contains all evaluation points, the stated rule readings,
config changes, fixes and limitations. Focused tests remain 32 passed; final full lint over
`fog tests docs/briefs/04-artifacts` is clean. No model, loss, search or PPO algorithm was
changed. No gate, candidate checkpoint, broader held-out improvement or GPU result is claimed.

## 2026-11-12: replay trainer profile and the data-or-steps runs (ROADMAP P2)

Brief `docs/briefs/03-p2-scale-measure.md`; branch `work/p2-scale-measure`, base
`4f9d75c`; report `docs/briefs/03-report.md`; evidence
`docs/briefs/03-artifacts/`. Section date uses the owner's local start date;
execution timestamps below are UTC. Application/model/dataset/loss/recipe code
is unchanged from the reviewed base; this work adds measurements, supervision
helpers and instructions. P2.3–P2.5 and G2 remain open.

### Setup, exact data and checks

The dataset metadata endpoint returned HTTP 200. The pinned revision
`792a86b5cdbcab9696032e947483b218453a0d4f`, rating floor 1500 and upload date
from 2024-02-01 produced 22,000 games in 194 s, with brief 01's filter counts:
77,225 below rating, 4,176 invalid metadata, 495 outside dates, 840 other format.
Conversion produced 22,000 accepted games/1,127,562 rows in 2,225 s wall,
including the documented 758 s main-converter pause (04:02:31–04:15:09 UTC).
Nested 40/1k/3k/10k/20k splits have
2,020/52,983/156,597/514,103/1,023,525 rows. Holdout: 2,000 games/104,037 rows,
exactly the committed IDs, uploaded 2026-02-20 04:17:33 through
2026-05-19 04:52:01 UTC. Fixed splitting drops zero games and matches both
chronological holdout and train20k membership.

Commands/outputs: `continuation-commands.txt`, `continuation-import.txt`,
`continuation-import-wall.txt`, `convert-output.txt`, `convert-quality.json`,
`convert-wall.txt`, `verify_split.py`, `verify-split-output.txt`,
`split-fixed-output.txt`. Component bytes are in `disk-bytes.json`.
The owner corrected the assignment from brief 02 to brief 03 during setup;
initial brief-02 commit `9867251` remains on its separate `work/p2-scale-prep`
branch. Existing brief-01/02 files on this branch are unchanged.

Reported machine: AMD EPYC 9V74, affinity five, four cores of quota, 16 GiB,
PyTorch 2.13.0+cu130, no CUDA device, five intra/inter-op threads. Locked setup and
pinned Showdown build pass. All 82 focused tests passed in 20.26 s; Ruff clean;
no full suite ran (`continuation-tests.txt`, `continuation-ruff.txt`, commands
above). No model, loss, learning rate, batch or thread tuning was introduced.

### Stage 1: measured; no source optimization qualifies

Commands: `uv run --no-sync python -m fog.scripts.profile_replay_training --out
runs/blueprint/profile-before` and the same with `profile-after`, at `2a608ad`.
Both use the unchanged CPU recipe, five threads, 20 warm-ups and 200 measured
updates. Wall times: 330/321 s. Sources: `profile-before-*`, `profile-after-*`,
`profile-comparison.json`; cumulative top-15 cProfile files are retained.

| Measurement | Before | Unchanged-default repeat |
| --- | ---: | ---: |
| Batch building, s/step | 0.039762 | 0.038087 |
| Forward, s/step | 0.608237 | 0.589191 |
| Belief heads within forward, s/step | 0.280480 | 0.269641 |
| Belief share of forward | 46.11% | 45.76% |
| Backward + optimizer, s/step | 0.654632 | 0.634710 |
| Other, s/step | 0.011513 | 0.012080 |
| Total, s/step | 1.314143 | 1.274067 |
| Training rows/s | 188.123 | 194.040 |
| Batch-only rows/s | 7,182.28 | 7,100.46 |

First 20 losses match exactly. Batch building is 3.03%/2.99% of a step and
exceeds 3,000 rows/s alone, so neither loader optimization nor prefetch qualifies;
no batch change requires a 50-batch equivalence check. Model computation accounts
for 96% of the repeated step. Thread screens used five warm-ups, 30 updates and
one separate cProfile update (`thread-benchmarks.json`, raw files beside it):

| Threads | s/step | rows/s | Wall s |
| --- | ---: | ---: | ---: |
| 1 | 2.417102 | 102.133 | 95.067 |
| 2 | 1.561363 | 158.110 | 62.085 |
| 4 | 1.234657 | 199.948 | 49.680 |

Four threads save only 6.05% versus the five-thread baseline, below 10%;
alternatives also change the first 20 losses by up to 1.54e-4. Five threads
remain unchanged. The repeat's 3% timing difference is observed variation,
not a code speedup. Belief computations/losses were retained.

### Completed comparisons and checkpoint selection

All runs retain the recipe's batch 256, learning rate/cosine, loss coefficients,
exact-feature/legality requirements and five runtime threads, with
`bc.streaming=true bc.selection_metric=action_ce bc.resume=true
bc.early_stopping_patience=0`. R/A/B seed 20261201; S seed 20261202.
Let `C` mean `uv run --no-sync python docs/briefs/03-artifacts/run_comparison.py`.
Every expanded trainer/report command is in `<run>-execution.json`; metrics,
last 60 output lines and both standalone reports are `<run>-metrics.csv`,
`<run>-tail.txt`, `<run>-holdout-{best,final}.json` and matching output files.
`<run>-summary.json` records exact values, formulas, state steps and curves.
Trainer wall includes validation; rows/s measures pure training. Last fields use
epoch metrics; best fields use standalone reports.

| Run | Games / epochs / actual steps | Trainer wall / rows/s | Last NLL / top-1 | Best report NLL / top-1 (epoch) | Final train NLL | Value MSE min / final | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R | 1,000 / 6 / 1,286 | 2,799.785 s / 187.060 | 1.810490 / 0.340484 | 1.810496 / 0.340476 (6) | 1.737690 | 0.993827 / 1.247020 | `247a1e1` | `C R` |
| A | 1,000 / 18 / 3,870 | 9,355.371 s / 157.725 | 1.948926 / 0.368527 | 1.736081 / 0.375367 (8) | 1.241824 | 1.015312 / 1.595830 | `41b6ffd` | `C A` |
| S | 1,000 / 6 / 1,286 | 3,224.913 s / 158.587 | 1.794974 / 0.345247 | 1.794978 / 0.345246 (6) | 1.718056 | 0.999632 / 1.237886 | `aed6f68` | `C S` |
| B | 3,000 / 18 / 11,431 | unknown, ≥21,149.912 s / 157.288 | 1.842549 / 0.426488 | 1.579018 / 0.440802 (8) | 1.062751 | 0.956552 / 1.601174 | `620d6c2`; resume `1041c20` | `C B` |

R/A/S trainer-plus-report walls: 3,018.406/9,611.736/3,487.043 s. Their best/final
report walls are 108.494/110.127, 128.951/127.414 and 131.955/130.175 s.
Each projection was saved before launch (`R/A/S/B-projection.json`). B's
stage-1-rate reference was 18,060.446 s; conservative recent-rate budget
21,405.112 s (5.95 h), below eight hours, so B was run. Actual step counts exceed
the brief's approximations because streaming buffers end in partial batches.

B's original attempt ended with the environment at an unobserved time. On return
no trainer was live; atomic state/CSV held epoch 15, step 9,523. Original wall
and exit remain null in `B-execution.json`. Its completed epoch timers sum to
18,121.330 s; the identical command resumed at 21:28:09 UTC, took 3,028.582 s
and exited 0. Thus trainer wall is at least 21,149.912 s; best/final reports add
108.992/104.027 s, giving at least 21,362.931 s combined. The UTC gap/downtime
is not charged to training. Reported CPU/runtime/settings were unchanged;
rows 0–15 remain exact after epoch 18. `B-environment-resume.json`, its prefix
CSV/tail, final metadata and `B-summary.json` retain the evidence. The six-epoch
snapshot is explicitly intermediate, not a completed six-epoch schedule.

### Historical reproduction guard and actual recovery

R's historical three-decimal guard **failed**, and remains failed. Current
baseline/epochs 1–6: 2.306/2.045/1.992/1.917/1.850/1.826/1.810; legacy:
2.306/2.043/1.990/1.909/1.841/1.819/1.804. Last NLL drift is 0.006311.
Baseline normalized metrics agree within 2.49e-8. Current AMD versus historical
Intel CPU differs, while the old thread count is unrecorded; specific hardware/
thread/kernel contributions are not isolated. Source comparison over model,
constants, encoding, dataset and recipe is empty. See `reproduction-platform.json`.

Before A/S/B, the required six-epoch repeat with resume/selection omitted
(`C R-default`, at `c5938bc`) matched every non-timing metric and all 376 final
parameter/buffer tensors exactly. Trainer wall 2,737.795 s, reports
107.419/125.688 s, total 2,970.903 s. Default loss selection chooses epoch 5
(NLL 1.825620) versus action-CE selection's epoch 6 (1.810496). No resume state
is written with the option omitted (`R-default-summary.json` and raw files).

Original `f0dc3d1` and current trainers then matched all 20 real-input losses
(max difference 0) and all 376 tensors after 20 completed updates, retaining
normal full baseline evaluation and six-epoch schedule. Shared-process control
wall 426 s (`check_real_updates.py`, `real20-*`); clean fresh children confirmed
NumPy/Torch initially absent and matched the shared losses and each other, wall
451.193 s (`check_fresh_updates.py`, `fresh-real20/`). Current losses also match
the profiler exactly. These isolate the tested update/startup paths without
turning the failed historical guard into a pass or introducing runtime variants.

A was actually SIGTERM-killed after atomic epoch 2 (exit −15, 1,197.492 s), then
resumed the identical command at 07:07:12 UTC. It completed epoch 18; the original
CSV prefix, including timing fields, is exact (`A-resume-evidence.json`,
`A-resume-metrics.csv`, `A-resume-tail.txt`, final `A-execution.json`). This
completes the required real-data recovery stage. B's later environment recovery
is additional recorded evidence, not a planned timing experiment.

### Reading: data, steps, value and switching

The two-seed NLL difference |S−R| is 0.015516; twice that is 0.031031. No smaller
difference is called an effect. Controlled A−R ends 0.138436 worse, above that
threshold. B ends 0.230221 above legacy 3k6, a cross-platform reference. A/B
train–holdout gaps open from 0.142664/0.153659 at their epoch-8 minima to
0.707102/0.779798 at the end; holdout NLL rises 0.212838/0.263535 after the
minima. B's upturn already exists at epoch 15 before interruption (NLL 1.779463).
Both longer runs overfit. This supports a data/generalization limit: extra
updates alone do not explain the earlier gain. A versus legacy 3k6 differs by
0.336598 at roughly matched steps, but exact historical data-effect size remains
unisolated across CPUs; a same-runtime 3k six-epoch run would resolve it.
B's selected-checkpoint difference from legacy 3k6 is −0.033310, barely above
the rough noise threshold and cross-platform, so it is not a clean step gain.

Value learns weak early signal (B minimum MSE 0.956552), then overfits: all final
heads are worse than constant zero's MSE 1.0. B's policy-selected value MSE is
1.378004. `comparison-curves.txt` places every current/legacy epoch's training/
holdout policy NLL and value MSE beside the constant reference; exact deltas,
source links and noise formulas are in `comparison-reading.json`.

B epoch 8 is the lowest-NLL completed policy: 1.579018, top-1 0.440802. Its
mean switching probability is 0.454911 on human-switch rows and 0.192942 on
human-move rows; switch-only target accuracy 0.449789, versus full-action top-1
0.211427 on human-switch rows. Sources: `B-holdout-best.json`,
`comparison-reading.json`. These diagnose the remaining action-family/target
limitations; they are not a play-strength gate.

### Real runner, scale instructions and checkpoints

Real 600-game runner command at `fa35867`:
`uv run --no-sync python -m fog.scripts.run_replay_blueprint --games 600 --work-dir runs/blueprint_runner_smoke --device cpu --batch-size 256 --epochs 2 --chronological-holdout-games 100`.
It called the existing importer on the genuine cached source shard, converted
600 games/29,508 rows, split train 500/24,278 and holdout 100/5,230, trained and
scored. After first completion, only training/results were deleted. The same
command skipped import/conversion/split, was terminated after epoch one (exit
143, 147.248 s), then resumed epoch two (exit 0, 151.991 s). CSV `[0,1]` became
`[0,1,2]` with earlier rows exact; all numeric metrics also match the first
completion (timings excluded). Post-completion training attempts total
284.380 s; best/untrained scores 7.216/7.133 s. Evidence:
`runner-smoke/`, `runner-first-completion/`, `check_real_runner.py`,
`post-completion-step-seconds.json`, `repeat-comparison.json`.
The initial 1.684/74.546/2.260 s import/conversion/split walls competed with the
main converter and are not idle measurements; earlier authorized recovery and
the 758 s main-converter pause are retained, not relabeled as uninterrupted.

Stage 5 is complete in `docs/replay-blueprint-scale-run.md`: public clone,
fixed IDs, pinned code, numbered cells, exact resume/download steps and measured
CPU time/disk arithmetic. Idle fixed split: 2.154823 s at 22k, output 700,419
bytes, projected about 11 s at 112k (`split-timing.json` and output).
Six-epoch checkpoints plus result-best duplicate: 287,619,529 bytes
(`checkpoint-storage.json`). Full rates/profile range is 157.29–194.04 rows/s;
the CPU pipeline for 20k training games projects to 9.618–11.737 h; importing
112k and training about 100k projects to 46.674–57.009 h on CPU
(`scale-projections.json`, complete inputs/formulas).
These are batch-256 CPU references. Actual batch-1,024/GPU speed and memory
remain unmeasured; the page says **not run on a hosted GPU**.

`checkpoints.json` lists actual retained best/final/state paths, sizes and
SHA-256 values. Winner: `runs/blueprint/g3000e18/checkpoints/bc_best.pt`,
23,548,503 bytes, SHA-256
`bed0804dc21d83b27acba45d3359c5a8ec88cb3233210c21a65d719df5a02c8a`.
Ignored weights may disappear if the machine is reclaimed; text evidence is
pushed after each stage.

### Stage 6: play smoke, no gate

B epoch 8 played 400 games per opponent on `data/teams/pool_b`, greedy and
without search: 800 completed games, zero failures, draws or truncations.
Total wall: 668.304 s (row walls 322.215/343.412 s). Sources: `play3_t0.json`,
`play-output.txt`, `play-execution.json`, `play-summary.json`. The following
rows are pasted exactly from the evaluator and are **smoke only**:

| Date | Candidate | Opponent | Backend | Games | Win rate [95% Wilson] (W-L-D) | s/decision | Fallbacks | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-13 | blueprint | simpleheuristics | showdown_direct | 400 | 0.360 [0.314, 0.408] (144-256-0) | n/a | n/a | `26107f2+dirty` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint blueprint=runs/blueprint/g3000e18/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --action-stats --out runs/blueprint/play3_t0.json` |
| 2026-11-13 | blueprint | maxdamage | showdown_direct | 400 | 0.772 [0.729, 0.811] (309-91-0) | n/a | n/a | `26107f2+dirty` | `python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint blueprint=runs/blueprint/g3000e18/checkpoints/bc_best.pt --checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --action-stats --out runs/blueprint/play3_t0.json` |

The `+dirty` stamp reflects pending measurement documents/artifacts at launch;
application/config/dependency/submodule code matches the reviewed base. The
evaluator emitted `n/a` for decision timing and fallbacks; total wall is not
substituted for those fields. Exact MaxDamage win rate is 309/400 = 0.7725;
the printed row rounds it to 0.772.

New SimpleHeuristics 0.360 [0.314, 0.408] and MaxDamage 0.7725 [0.729, 0.811]
smoke intervals are wholly above brief 01's 0.258 [0.217, 0.303] and
0.623 [0.574, 0.669]. This supports improved results in these smoke matches;
400 games and `pool_b` do not establish held-out strength or pass G2.

Switching in play: SimpleHeuristics 3,749/13,099 = 28.62%; MaxDamage
3,349/14,163 = 23.65%; combined 7,098/27,262 = 26.04%, beside the human
replay reference 28% (29,107/103,981 = 27.99%). The numerator is action family
`switch`, slots 8–12; slot 13 is forced/default and excluded. Play includes
forced replacements while retained replay decisions omit forced/single-choice
rows, so these raw rates are not a like-for-like voluntary-switch comparison.
Formulas and counts are in `play-summary.json`. P2.3–P2.5 and G2 remain open.

## 2026-11-14: review of brief 03

The branch `work/p2-scale-measure` (head `c5a1cb4`) was reviewed and merged into `work/roadmap`.
It changes no application code: 22 commits of measurements, the report, the ledger section
above and the rewritten `docs/replay-blueprint-scale-run.md`. Checked in review:

- The stage 2 table against `docs/briefs/03-artifacts/{R,A,S,B}-metrics.csv`: step counts, last
  and best holdout NLL, top-1 and final training NLL agree to the printed digits (run A's best
  is 1.736087 in the CSV and 1.736081 in the standalone report; both are quoted correctly).
- The play smoke against `play-summary.json`: 144/400 against SimpleHeuristics, 309/400 against
  MaxDamage, no failures or truncations.
- No effect smaller than twice the two-seed difference (0.031) is claimed.
- `ruff check fog tests` is clean; every commit is authored as the owner with no attribution.
- Not re-run in review: the trainings themselves and the kill-and-continue tests (their before
  and after files are in the artifacts).

What the review takes from it, all smoke:

| Finding | Evidence |
| --- | --- |
| The policy is limited by data, not by updates | 1,000 and 3,000 games both reach their best holdout NLL at epoch 8 of 18 (1.736 and 1.579) and then get worse while training NLL keeps falling (to 1.24 and 1.06) |
| More games still buy play strength | The 3,000-game model at its best epoch wins 0.360 against SimpleHeuristics and 0.7725 against MaxDamage on `pool_b`, against 0.258 and 0.623 for brief 01's 3,000-game model after six epochs |
| Training is bound by model computation on CPU | 157 to 194 rows/s on four cores, 96% of a step in the model, 46% of the forward pass in the belief heads; batch building alone would allow 7,100 rows/s |
| The value head memorises games | At the best policy epoch of run B, holdout value Brier is 0.3445 (a constant 0.5 scores 0.25) and ECE 0.299. It predicts below 0.1 or above 0.9 on 56% of rows; those rows were won 40% and 59% of the time. Late-game rows are no better than early ones (Brier 0.327 against 0.343). The lowest value MSE of any epoch is 0.957, against 1.0 for a constant |

The last row matters for gate G2, which asks for value ECE below 0.05, and for search in P4,
which needs a value that says something. Run B has 3,000 outcome labels against 156,597 policy
labels: every row of a game carries the same outcome, and the inputs identify the game. The
checkpoint is chosen by policy loss, long after the value head has started to memorise. In
brief 01 the six-epoch model had ECE 0.032 with value MSE 0.92 to 1.05: calibrated because it
predicted close to 0.5 everywhere. In play the same head does learn: brief 04's fixed-team run
shows value explained variance of 0.26 to 0.38 between 80,000 and 250,000 decisions (interim, branch
`work/rl-check`). So this reads as too few independent outcomes and no separate stopping point
for the value head, not as a head that cannot learn. Brief 05 measures whether 20,000 and
100,000 games change it.

Not done by brief 03 and carried forward: any run on a GPU, batch 1,024, and the 20,000 and
100,000-game points. Next: `docs/briefs/05-p2-gpu-scale.md`.

## 2026-11-14: faster rollout collection on headless Showdown (ROADMAP P0.5)

Asked for by the owner on 2026-11-14 after the accelerator simulator was measured below its
bar. Branch `work/fast-collect`, code at `21e57fa`. One machine for every row: 4 CPU cores, no
GPU, PyTorch 2.13 CPU, pinned Showdown `e440c4a`, random weights, teams `data/teams/pool_a`.
All numbers are smoke-level throughput measurements; nothing here is a strength result.

**What changed**

1. `_ShowdownRolloutWorker` (the trainer's rollout process) can play `env.rollout_batch`
   battles at once through `fog/env/showdown_batch_runner.py`, one forward per wave. Default 1
   keeps the old one-battle loop. This is the wiring P0.5 listed as remaining.
2. `env.actor_device` (default `cpu`): a worker keeps its own copy of the actor on that device,
   refreshed from the shared CPU actor at the start of every chunk; the two tensors that
   sampling reads come back to the CPU once per forward. **Not run on a GPU**: the copy and
   refresh are tested on the CPU only.
3. Observation encoding does the same work with less Python: name keys cached and built once per
   Pokémon instead of once per pair, legal moves and switches derived once per observation
   instead of three times, completed history turns recognised by object identity instead of
   comparing every event, rule-player matchup scores reused within one feature table, scalar
   `np.clip` and per-element writes replaced.
4. `fog.eval.showdown_rollout_probe` gained `--workers N` (the trainer's worker processes),
   `--device` and `--opponent`.

**Nothing recorded changed.** The probe's digest hashes every recorded learner observation,
action, log-probability, value and outcome.

| Run | Digest before | Digest after |
| --- | --- | --- |
| `--batch 32 --games 64 --threads 2` (self-play, `pool_b`, 8,743 decisions) | `0efd4319547a4b3b` at `fff88ca` | `0efd4319547a4b3b` after every step, last at `21e57fa` |
| `--games 6` (one-battle loop, 898 decisions) | `23405989f4665988` at `fff88ca` | `23405989f4665988` |
| `--batch 1 --games 6` (runner with one battle) | | `23405989f4665988`, equal to the loop |

**Collection rate, four worker processes** (`python -m fog.eval.showdown_rollout_probe
--workers 4 --batch B --games G --opponent O --pool data/teams/pool_a`; old code is `fff88ca`
with the same probe file, back to back in one session; learner decisions per second, network on
the CPU):

| Opponent | Old code, one battle at a time (128 games) | New, `--batch 1` (256 games) | New, `--batch 32` (256 games) |
| --- | ---: | ---: | ---: |
| frozen copy of the learner (both seats run the network) | 179 | 213 | 451 |
| MaxDamage | 324 | 357 | 647 |

`--batch 64` gave 656 against MaxDamage and a 768-game chunk 643, so 32 is enough. The trainer
itself ran end to end with `env.rollout_batch=32`, four workers and `ppo.games_per_chunk=128`
(11,891 decisions, two updates, MaxDamage only).

**Where a learner decision's time goes now** (one worker, `--batch 32 --games 96 --opponent
MaxDamage`, 3,151 decisions in 14.85 s, about 4.7 ms each):

| Part | ms per learner decision | What it is |
| --- | ---: | --- |
| Network forward on the CPU | 1.3 | goes to the GPU with `env.actor_device` |
| Showdown in the Node process | about 1.0 | both players' turn; 0.78 ms per step once warm |
| Applying Showdown's messages (poke-env) | about 0.45 | both views |
| Building the observation | 0.9 | in the self-play probe: 1.9 before this work, 1.1 after |
| Sampling, recording, opponent's scripted choice | 0.7 | |
| Starting battles | 0.2 | 7 ms per game |

The bridge is not where time is lost: driving the same Node worker from Python with random
legal choices and no observations runs 2,168 player decisions per second on one core (200
games, second pass); Showdown called directly ran 2,598 on the reviewer's machine on 2026-11-13.
What remains is ordinary Python spread thinly over poke-env's message parsing and the encoder;
apart from waiting on Node and the forward, no function's own time is above 3% of a profile.

**Reading.** With the forward on a GPU the CPU side costs about 3.4 ms per learner decision
against a scripted opponent, so four cores should collect roughly 900 to 1,200 learner
decisions per second (not measured; a GPU is needed). Collection scales with CPU cores. P0.5's
target of 150 decisions per second on four cores is met on the CPU alone; its GPU forward is
written and untested, so the box stays open.

**Tests.** `python -m pytest -q tests` at `ab75672`: 1,017 passed, 3 skipped, 3 expected
failures, 1 failed. The failure, `test_active_typed_league_uses_portable_checkpoint_artifacts`,
needs checkpoint files that are not in a fresh clone and fails the same way at `fff88ca`.
`ruff check fog tests` is clean. Next: `docs/briefs/06-collection-on-gpu.md`.

**Not verified:** anything on a GPU; `env.actor_device` together with `leakage.beta > 0`
(refused); heuristic guidance with `env.rollout_batch > 1` (refused); the trainer's learning
curves under the batched path (with more than one battle per forward, log-probabilities and
values differ from the one-battle loop only by floating-point reduction order, as recorded on
2026-11-11).

## 2026-11-13: collection and play-training speed on a GPU (smoke)

Brief `docs/briefs/06-collection-on-gpu.md`, branch `work/collect-gpu`, base
`b5983c6`. Preparation only; report `docs/briefs/06-report.md`. Kaggle token
authentication succeeds using an isolated client installation. The first
list/status sweep returned HTTP 429; a smaller list and targeted status checks
succeeded. An existing replay-training GPU job is running, so this brief has
not submitted a job, in accordance with its one-job-at-a-time rule.

| Measurement | Status | Games / interval | GPU hours | Command and commit | Evidence |
| --- | --- | --- | --- | --- | --- |
| Stage 1 collection matrix (smoke) | Pending: shared account's GPU job running | Not measured | 0 | `KaggleApi.authenticate(); KaggleApi.kernels_status(...)`, base `b5983c6`; account identifier omitted | `06-artifacts/account-status.json` |
| Stage 2 PPO/update/evaluation (smoke) | Not run; depends on stage 1 | Not measured | 0 | Brief stage 2 commands, not run | No metrics/checkpoint/evaluation exists |
| Stage 3 bottleneck/week projection (smoke) | Not measured | Not applicable | 0 | No extrapolation without measured rates | `06-report.md` |

`06-artifacts/gpu_job.py` prepares the collection matrix with an internal
3,300-second deadline and a requested 3,600-second platform cap on one GPU.
No model, losses, trainer or search changes; no defect claimed. P0.5 stays open.

Preparation checks at base `b5983c6`: pinned Showdown builds after redirecting
the local npm cache to a writable temporary directory. Locked development
installation succeeds with writable UV/XDG/Rust caches. Command
`UV_CACHE_DIR=/tmp/fog-uv-cache uv run --no-sync python -m pytest -q -p no:cacheprovider tests/test_showdown_batch_runner.py tests/test_showdown_rollout_probe.py tests/test_direct_parallel.py`:
18 passed (`06-artifacts/focused-tests.txt`).
`UV_CACHE_DIR=/tmp/fog-uv-cache uv run --no-sync python -m ruff check fog tests docs/briefs/06-artifacts/*.py`
is clean. Full suite not run. The optional measurement-only timing shim is
syntax/lint checked, not training-verified; it does not change trainer source.

Preparation follow-up: the installed Kaggle client documents P100 as retired
and substituted with T4. The job script therefore supports one or two allocated
GPUs and now uses collection/training platform caps of 1,800/3,600 seconds
(internal deadlines 300 seconds shorter). At two allocated GPUs these caps
reserve at most three GPU-hours; the earlier proposed single-P100 caps were
never submitted. Stage 2's entry point preserves the inherited recipe, records
unique update rows/s and forward versus remaining update time, retains PPO
checkpoints and halves the initial minibatch after a documented CUDA OOM.
No stage 2 config is chosen before the collection measurements exist.

`UV_CACHE_DIR=/tmp/fog-uv-cache uv run --no-sync python docs/briefs/06-artifacts/check_timing.py`:
CPU engineering smoke, CUDA synchronization/memory calls mocked. Every update
metric and final state tensor is identical with/without timing hooks; collection
result identity, row counts and forward time partition also pass
(`06-artifacts/timing-equivalence.json`). No GPU measurement is implied.

Stage 1 submitted after the account-wide recent-job check found no active GPU
job. Kaggle accepted the private T4 script at `c4df5e2` on 2026-11-14
04:09:05 UTC (`06-artifacts/collection-submission.json`). Platform cap 1,800 s;
requested accelerator `NvidiaTeslaT4`, no datasets, internet enabled, code
frozen to the pushed commit. The private entry point injects `FOG_COMMIT`
and `FOG_STAGE=collection` into the committed `gpu_job.py`. Measurements and
actual allocation/runtime are pending; the three-GPU-hour limit is unchanged.

| Measurement | Result (smoke) | Games / interval | Commit / command | Evidence |
| --- | --- | --- | --- | --- |
| Collection attempt 1 | Failed before clone: `/kaggle/temp` absent; two T4s allocated | 0 / not applicable | `c4df5e2`, private script `gpu_job.py`, `FOG_STAGE=collection` | `collection-attempt1-tail.txt`, `collection-attempt1-budget.json` |

Attempt 1's script timer was 0.048261 s; this excludes container startup/export
and is not the budget charge. Submission-to-observed-terminal is at most
141 seconds, or 0.078333 allocated GPU-hours on two GPUs
(`collection-attempt1-runtime-upper.json`). Reserve 600 session seconds for it
and lower the remaining training platform cap to 3,000 seconds. Together with
the collection retry cap 1,800 seconds, this bounds all three allocations to
three GPU-hours even with two allocated T4s each. No probes or PPO ran.
The initial submission evidence was renamed `collection-attempt1-submission.json`.

Fix: explicitly create `/kaggle/temp` before cloning. Regression command
`uv run --no-sync python -m pytest -q -p no:cacheprovider tests/test_collection_gpu_job.py`
fails without that line and passes with it (one test;
`job-bootstrap-before.txt`, `job-bootstrap-test.txt`). No actor/trainer/model/loss
change. Ruff remains clean. The test simulates job bootstrap, not a GPU.

The collection retry was accepted at `5603782` on 2026-11-14 04:12:45 UTC
(`collection-submission.json`). Live progress confirms both CPU self-play
repeats and both CUDA batch-32 self-play repeats completed; rates await export.
Before training, timing capture now distinguishes unique batch rows from
optimizer row-visits (including partial epochs/KL stopping), retaining the same
update metrics and parameters in the repeated CPU engineering check.

Job runtime handling now allows an explicit collection platform cap up to
3,600 seconds via `FOG_PLATFORM_TIMEOUT`, with the same value sent to Kaggle.
Defaults stay unchanged. Any further allocation must fit the remaining budget
after prior submission-to-observed-terminal bounds, counting both T4s; a
longer collection cap reduces the available training cap. This changes only
the measurement job, not the probe or trainer. Bootstrap regression and Ruff
pass again.

Collection attempt 2 stopped at its internal deadline, not a CUDA error. All
eight self-play rows completed, then `probe-MaxDamage-cpu-32-1` exited 124
after 9.191 s without a result. Evidence is in `06-artifacts/collection-attempt2/`.
Hardware: four quota/affinity cores, Intel Xeon @ 2.00GHz, two Tesla T4 15,360 MiB;
Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, Node v22.23.4. No GPU code fix.
The following completed measurements are partial-stage smoke results, not the
final same-job CPU/GPU matrix:

| Opponent | Device | Batch | Repeat | Games | Learner decisions | Timed s | Learner decisions/s | Commit | Command |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| self (smoke) | cpu | 32 | 1 | 512 | 29982 | 192.49 | 155.76 | `5603782` | `python -m fog.eval.showdown_rollout_probe --workers 4 --batch 32 --games 512 --opponent self --pool data/teams/pool_a --device cpu` |
| self (smoke) | cpu | 32 | 2 | 512 | 29982 | 192.97 | 155.37 | `5603782` | `python -m fog.eval.showdown_rollout_probe --workers 4 --batch 32 --games 512 --opponent self --pool data/teams/pool_a --device cpu` |
| self (smoke) | cuda | 32 | 1 | 512 | 29648 | 144.05 | 205.81 | `5603782` | `python -m fog.eval.showdown_rollout_probe --workers 4 --batch 32 --games 512 --opponent self --pool data/teams/pool_a --device cuda` |
| self (smoke) | cuda | 32 | 2 | 512 | 29648 | 141.89 | 208.95 | `5603782` | `python -m fog.eval.showdown_rollout_probe --workers 4 --batch 32 --games 512 --opponent self --pool data/teams/pool_a --device cuda` |
| self (smoke) | cuda | 64 | 1 | 512 | 29648 | 141.79 | 209.10 | `5603782` | `python -m fog.eval.showdown_rollout_probe --workers 4 --batch 64 --games 512 --opponent self --pool data/teams/pool_a --device cuda` |
| self (smoke) | cuda | 64 | 2 | 512 | 29648 | 139.09 | 213.15 | `5603782` | `python -m fog.eval.showdown_rollout_probe --workers 4 --batch 64 --games 512 --opponent self --pool data/teams/pool_a --device cuda` |
| self (smoke) | cuda | 128 | 1 | 512 | 29648 | 137.04 | 216.34 | `5603782` | `python -m fog.eval.showdown_rollout_probe --workers 4 --batch 128 --games 512 --opponent self --pool data/teams/pool_a --device cuda` |
| self (smoke) | cuda | 128 | 2 | 512 | 29648 | 139.23 | 212.95 | `5603782` | `python -m fog.eval.showdown_rollout_probe --workers 4 --batch 128 --games 512 --opponent self --pool data/teams/pool_a --device cuda` |

Attempt 2's script duration is 1,500.788 s, or 0.833771 allocated GPU-hours.
The stronger budget bound includes startup/export: submission to observed
terminal is at most 1,546 s. Together with attempt 1's 141 s, the two
allocations cost at most 2 × (1,546 + 141) / 3,600 = 0.937222 GPU-hours.
A complete-matrix retry will request a 2,520-second platform cap, with an
internal 2,220-second deadline. Its worst-case allocation is 1.4 GPU-hours;
training will use only the budget remaining after its measured completion.
This preserves all comparisons on one job instead of combining different hosts.

The complete-matrix resubmission was held by the account-wide guard: a new
replay-training GPU job began at 2026-11-14 04:27:10 UTC while attempt 2 was
already running. Its status is now RUNNING, so no further brief 06 job was
submitted. The other job was not active when attempt 2 began. This account
overlap is recorded explicitly; its time is not charged to brief 06. Evidence:
`account-status.json`. Budget accounting is `gpu-budget.json`, using both
allocated T4s and conservative submission-to-terminal bounds.

Training timing will report the complete `_maybe_chunk_update` phase
(batch assembly, optimizer work, diagnostics, logging and checkpoint work),
with `ppo_update` and its forward/remainder as explicit subcomponents.
This avoids assigning update overhead to collection. JSONL phase events
complement the trainer's unchanged CSV logging path. The CPU engineering
check still preserves all update metrics/tensors and wrapper return values
(`timing-equivalence.json`); CUDA timing remains unverified until stage 2.

## 2026-11-14: review of brief 04

The branch `work/rl-check` (head `830a7ea`) was reviewed and merged into `work/roadmap`.
Checked in review:

- Every win rate in the report's two tables against the evaluation files in
  `docs/briefs/04-artifacts/` for the first, the peak and the last point of each run (for
  example 32/200 and 200/200 against MaxDamage; 102/400, 387/400 and 365/400 against
  SimpleHeuristics on `pool_a`; 86/400 on `pool_b` at the end). All 400- and 200-game files
  show no failures or truncations.
- The trainer changes, read line by line: `ppo.activation_checkpointing` (default off), timing
  and diagnostic columns in the metrics, CSV history kept on resume, and a warm-started run
  resuming from its saved state. No change to the model, the losses or the update rule.
- After the merge with the collection work of the same day: the three new test files and the
  trainer, probe and batch-runner tests pass (38), `ruff check fog tests` is clean, and the
  rollout digest is unchanged (`0efd4319547a4b3b`). Commits are authored as the owner.
- Not re-run in review: the two training runs and their evaluations.

What the review takes from it, all smoke, one training seed, budgets scaled to 0.557 of the
brief's because the CPU trainer ran at 22 decisions per second:

| Question | Answer | Evidence |
| --- | --- | --- |
| Can this model and loop learn from play? | Yes | From random weights on one team: 0.160 to 1.000 against MaxDamage by 106,889 decisions, and 0.920 against SimpleHeuristics, which it never trained against |
| Does play improve the human-data policy? | On the teams it trains on | 1,000-game blueprint on the eight `pool_a` teams: 0.255 to 0.9125 against SimpleHeuristics at 556,907 decisions (peak 0.9675) |
| Does that carry to other teams? | No | `pool_b`, never trained on: 0.220 before, 0.215 after; every interval overlaps the start |
| Does the value head learn from play? | Yes | Explained variance of the behaviour value reaches 0.5 to 0.6 in the second run (`blueprint-metrics.csv`), against a head that memorised games on replays (review of brief 03) |
| What limits the speed? | The update on a CPU | 82.6 to 86.5% of wall time; 78 processed rows per second |
| What does a gain cost? | 12,725 and 37,000 decisions per ten points at the first measured point | |

Reading. Nothing here points at the architecture: policy and value both learn from play, and
fast. What the second run shows is that eight teams of the same six species teach those eight
teams. A self-play run of any size on a narrow pool will do the same, so the pool of teams,
not the number of games, is what decides whether self-play is worth scaling. The wide pool the
roadmap plans is the public team bank (P0.7), which is still an owner action: it has to be
collected on a machine where the forum pages open.

`pool_b` is not a held-out split of one distribution: it has other species. The result says
play on `pool_a` neither helped nor hurt there.

### 2026-11-14: owner prioritizes end-to-end timing after G20 (smoke preparation)

The owner asked for end-to-end games/second and chose to finish G20 first, then measure
brief 06 before G100. No new brief-06 job has been submitted. Merge `92ebd08` brings in
`work/roadmap` at `76eb4ee`, including the reviewed brief-04 timing, checkpointing and
resume code; both ledger histories were retained. The planned private combined job will
measure the eight missing MaxDamage rows, choose the aggregate fastest GPU batch, then
run the specified 100k-decision PPO recipe and 200-game evaluation. The earlier self-play
matrix remains a separate-job smoke result. This is a documented deviation from a single
same-job matrix, preserving the remaining budget for the newly requested full-loop speed.
The proposed cap is 3,600 session seconds, internal deadline 3,300, two T4s conservatively
counted: prior upper bound 0.937222 + cap 2.0 = 2.937222 allocated GPU-hours, leaving 113
session seconds for additional startup/export before three hours. No further allocation
may exceed the measured remaining allowance. An account-wide active/queued-job guard is
required immediately before submission. The initial G20/collection overlap remains recorded.

`configs/collect_gpu_check.yaml` changes only the brief's named settings; its 4-worker,
batch-128 values are a template, replaced by observed hardware and the new MaxDamage
selection before training (`endtoend-config-diff.json`). Minibatch remains 1,024; only the
brief-authorized activation-checkpointing option can be enabled after CUDA OOM, with the
failed attempt retained. No model, loss or trainer edits were made here.

The measurement wrapper now namespaces extra CSV fields with `timing_`, preserving the
new trainer's native `collection_seconds`, `update_seconds` and optimization columns.
A regression fails on the prior wrapper and passes with the fix (`timing-native-before.txt`).
A continuation keeps only the timing/CSV prefix represented by its saved model state, and
sums all trainer-attempt walls so retries cannot inflate end-to-end speed. CPU checks:
15 focused tests pass; timing hooks preserve update metrics and model tensors with CUDA
calls mocked; Ruff and byte compilation pass (`endtoend-*` artifacts). GPU PPO speed,
end-to-end decisions/games per second and batched-path learning remain unmeasured.

Timing equivalence was repeated with `PYTHONPATH=.` and an explicit print of the PPO
module path, ensuring the shared virtual environment used this worktree's newly merged
trainer. Command: `PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/workspace/fog/.venv/bin/python -c 'import runpy; import fog.training.ppo as ppo;
print(ppo.__file__); runpy.run_path("docs/briefs/06-artifacts/check_timing.py",
run_name="__main__")'`. All update metrics and tensors remain identical;
`endtoend-timing-equivalence.txt` records the current source path and CPU-only result.

The combined job now protects the end-to-end benchmark and evaluation before attempting
any additional self-play comparison. After those finish, it compares the remaining internal
deadline with the prior eight self-play execution times, scaled conservatively for fewer
cores, plus 20% and 60 seconds. Only if the whole repeat fits does it run those eight rows
on the same host, completing the requested one-job matrix. Otherwise the prior self-play
cohort stays separate and the report retains that limitation. The cap and three-hour
budget are unchanged. `same-job-self-plan.json` and `collection-completeness.json` will
record the actual branch, rather than assuming completion. The original child environment
is restored after timing instrumentation so subsequent probes use the same settings.

Five measurement-job cases verify bootstrap, benchmark-before-extra-probes order, skipping
extras when they cannot fit, native metric preservation and saved-state timing prefixes.
Together with the trainer measurement/resume cases, **17 focused tests pass**; Ruff and
byte compilation pass (`combined-order-tests.txt`, `endtoend-focused-tests.txt`). No GPU
allocation or application/trainer change was made while preparing this ordering.

## 2026-11-14: collection and play-training speed on a GPU (smoke)

The owner requested a fresh T4 runtime through the runtime CLI and SSH, replacing the
previous private-job plan. Earlier attempts and their 0.937222 GPU-hour allocation upper
bound above are retained. The existing measurement branch was merged before this run;
application, model, losses and trainer code are unchanged from `work/roadmap`.

Runtime: one Tesla T4 (15,360 MiB), two logical Intel Xeon 2.00 GHz CPUs, Python 3.13.16,
PyTorch 2.14.1+cu130, CUDA 13.0, pinned Showdown `e440c4a`. The established runtime installer
reuses the supplied CUDA PyTorch and pins the direct-Showdown dependencies. A missing Node.js
binary was installed; the first unpinned lint invocation used Ruff 0.17.5 and reported 77
findings. Ruff pinned to the lock's 0.15.22 is clean; no source was changed to silence them.
Setup logs, environment versions and checks are in `06-artifacts/runtime-t4/`.

At `571a9c0`, `python -m pytest -q tests/test_showdown_rollout_probe.py
tests/test_showdown_batch_runner.py tests/test_training_state.py tests/test_dashboard_resume.py
tests/test_collection_gpu_job.py`: **17 passed in 117.72 s**. The full suite was not run.
The same-runtime 16-row collection matrix started at this commit, through
`python docs/briefs/06-artifacts/gpu_job.txt probes`; each command and its raw output are
saved separately. No incomplete probe is a throughput result.

Evidence is downloaded to the controlling computer every 20 seconds. Completed checkpoint
weights and their optimizer/RNG state sidecars are downloaded as pairs. Training uses the
brief's recipe, with runtime overrides `log.dir=runs/collect_gpu` and
`league.snapshot_every_decisions=8000` for the owner's requested frequent recovery points.
These affect persistence, not the model, losses or update rule. Evaluation uses its existing
four-game-block resume option. A local guard releases the single T4 by 10:16 UTC, from a
conservative 08:16 UTC allocation start: prior plus this two-hour cap is 2.937222 GPU-hours.

Same-runtime collection, all **smoke**. Command prefix for every row:
`python -m fog.eval.showdown_rollout_probe --workers 2 --games 512
--pool data/teams/pool_a`; the remaining flags are in the last column. Full commands,
wall times and sampled GPU memory are in each named probe's JSON, text and CSV under
`06-artifacts/runtime-t4/`. Probe seconds exclude process startup; command wall includes it.

| Opponent | Device | Batch | Repeat | Games | Learner decisions | Probe seconds | Learner decisions/s | Commit | Remaining command flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| self | cpu | 32 | 1 | 512 | 30009 | 411.52 | 72.92 | `571a9c0` | `--opponent self --device cpu --batch 32` |
| MaxDamage | cpu | 32 | 1 | 512 | 16838 | 150.89 | 111.59 | `571a9c0` | `--opponent MaxDamage --device cpu --batch 32` |
| MaxDamage | cpu | 32 | 2 | 512 | 16838 | 151.75 | 110.96 | `571a9c0` | `--opponent MaxDamage --device cpu --batch 32` |
| self | cpu | 32 | 2 | 512 | 30009 | 410.77 | 73.06 | `571a9c0` | `--opponent self --device cpu --batch 32` |

At 08:50 UTC, integrating the concurrently updated branch revealed the owner-selected
separate single-T4 allocation already active since 08:16. The prepared combined private
Kaggle job was never submitted and is now superseded. Its tested script remains available,
but its planned two-GPU-hour reservation must not be added to the active runtime's
reservation. Prior allocations plus the current two-hour single-T4 cap already reserve
2.937222 of the three GPU-hours. No duplicate brief-06 allocation will be launched.
The active runtime's report/evidence are retained as published by that worker; G20 continues
unchanged on its separate account allocation. The earlier 14:50 estimate assumed waiting
for G20's Kaggle slot; this independent runtime can produce an earlier end-to-end result,
subject to its 10:16 budget stop and measured PPO speed.

The first CUDA self-play probe (`--opponent self --device cuda --batch 32`) at `571a9c0`
failed after 339.145 command seconds with a pipe EOF, not a measured throughput result.
The kernel recorded a host OOM kill of a rollout worker; cgroup `oom_kill=1` and the
sampled GPU peak was 415 MiB (`runtime-t4/{kernel-memory.txt,memory-events.txt,
probe-self-cuda-b32-r1*}`). A temporary swap attempt returned `Invalid argument`.
The low-memory allocation was released before 08:53 UTC and a high-memory single-T4
replacement requested. The same 10:16 UTC deadline applies across both allocations;
retained low-memory CPU rows will not be presented as same-runtime high-memory baselines.

The replacement's reservation is now reflected in `gpu-budget.json`: prior private jobs
0.937222 GPU-hours, first single-T4 allocation at most 0.616667, replacement at most
1.405556, cumulative upper bound 2.959444. Its new stop deadline is 10:21 UTC, accounting
for the gap without an allocation. The duplicate private-job plan stays unsubmitted.
The probe-only memory fix returns counters instead of unused buffers; full PPO still pays
for real rollout-buffer transfer. Those two timing scopes must remain separate in the report.

High-memory T4 requests returned `Service Unavailable` twice; no high-memory runtime was
allocated. A second standard T4 was started, after the first was released. The local guard
now stops it by 10:21 UTC; conservative per-allocation accounting gives a cumulative cap of
2.959444 GPU-hours (`runtime-t4-counts/session.json`), including the earlier private jobs.

The narrow fix at `9e977cc` makes only the measurement worker count decisions/wins before
returning its result. The probe no longer serializes unused observation buffers, which caused
the end-of-run host-memory spike. It calls the unchanged trainer worker for every game. No
model, loss, trainer or actor implementation changed. Probe timings now exclude full-buffer
IPC; the complete training run still includes it. The old CPU rows are retained separately,
and the complete CPU/GPU matrix is rerun using the same counter-only boundary for both.

The new regression explicitly forbids serializing the buffer. At the old `571a9c0` source
it fails with that assertion (3.39 s); with the fix, eight focused checks pass in 49.35 s and
Ruff is clean (`runtime-t4-counts/{regression-before.txt,focused-tests.txt,ruff.txt}`). This
checks identical counters and delegated worker lifetime without transporting fragments.
The replacement matrix runs at `922886b`, with CUDA first to catch any remaining defect.

Corrected same-runtime worker-collection matrix, all **smoke**. Timings include worker play
and counter transfer, excluding full rollout-buffer transfer. Command prefix remains
`python -m fog.eval.showdown_rollout_probe --workers 2 --games 512 --pool data/teams/pool_a`.
Evidence is in `06-artifacts/runtime-t4-counts/`; earlier low-memory rows above are not mixed in.

| Opponent | Device | Batch | Repeat | Games | Learner decisions | Probe seconds | Learner decisions/s | Commit | Remaining command flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| self | cuda | 32 | 1 | 512 | 30381 | 286.10 | 106.19 | `922886b` | `--opponent self --device cuda --batch 32` |

Run-order correction: after both CUDA batch-32 self-play rows and both batch-32 MaxDamage
rows completed, the newly started CPU self-play row was deliberately preempted (24.188
command seconds, exit -15; `planned-interruption.txt`). It is not a throughput result and
will restart later. The remaining MaxDamage sweep runs first, then the full 100k-decision
PPO run and greedy evaluation, then the remaining self-play matrix if time remains. This
protects the learning and end-to-end measurements within the hard cumulative GPU budget;
no completed probe is repeated. The launcher supports this with `FOG_PROBE_OPPONENT=MaxDamage`.

| Opponent | Device | Batch | Repeat | Games | Learner decisions | Probe seconds | Learner decisions/s | Commit | Remaining command flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MaxDamage | cuda | 32 | 1 | 512 | 16225 | 107.60 | 150.79 | `922886b` | `--opponent MaxDamage --device cuda --batch 32` |
| MaxDamage | cuda | 32 | 2 | 512 | 16225 | 107.63 | 150.75 | `922886b` | `--opponent MaxDamage --device cuda --batch 32` |
| self | cuda | 32 | 2 | 512 | 30381 | 282.14 | 107.68 | `922886b` | `--opponent self --device cuda --batch 32` |
| MaxDamage | cpu | 32 | 1 | 512 | 16838 | 137.27 | 122.66 | `922886b` | `--opponent MaxDamage --device cpu --batch 32` |
| MaxDamage | cpu | 32 | 2 | 512 | 16838 | 135.79 | 124.00 | `922886b` | `--opponent MaxDamage --device cpu --batch 32` |
| MaxDamage | cuda | 128 | 1 | 512 | 16225 | 102.18 | 158.79 | `922886b` | `--opponent MaxDamage --device cuda --batch 128` |
| MaxDamage | cuda | 128 | 2 | 512 | 16225 | 99.40 | 163.22 | `922886b` | `--opponent MaxDamage --device cuda --batch 128` |
| MaxDamage | cuda | 64 | 1 | 512 | 16225 | 104.23 | 155.66 | `922886b` | `--opponent MaxDamage --device cuda --batch 64` |
| MaxDamage | cuda | 64 | 2 | 512 | 16225 | 102.56 | 158.19 | `922886b` | `--opponent MaxDamage --device cuda --batch 64` |
