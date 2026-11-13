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
