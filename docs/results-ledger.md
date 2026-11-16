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

### Owner-requested short throughput check; runtime released

The owner asked why one backpropagation update was not enough and requested minimal work
because of possible runtime loss. Two full PPO updates were already complete, so the long
run was stopped and the latest completed state retained. This supersedes the 100k-decision
learning/evaluation plan for this session; it is a **short smoke timing check**. Six deferred
self-play probe repeats and the 200-game greedy evaluation were not run. No learning or
strength claim is made and P0.5 remains open (`short-run-scope.json`, `evaluation-not-run.json`).

Training command, at `ac75355`, generated by `gpu_job.txt training`:

```sh
env PYTHONPATH=/content/brief06-timing:/content/fog \
FOG_TIMING_OUTPUT=/content/fog/docs/briefs/06-artifacts/runtime-t4-counts/training-timing.jsonl \
/content/fog/.venv/bin/python -m fog.train --config configs/collect_gpu_check.yaml --trainer ppo \
--set log.dir=runs/collect_gpu league.snapshot_every_decisions=8000
```

All rows below are **smoke**, first team of pool A versus MaxDamage; native trainer phase
columns are preserved. Optimization rows include epoch revisits. Raw evidence:
`runtime-t4-counts/{metrics.csv,short-run-metrics.csv,short-run-timing.jsonl,training-tail.txt}`.

| Cumulative decisions | New decisions | Games | Collection s | Full update s | Optimization s | Processed rows | Optimizer steps | Processed rows/s | Effective epochs | Commit | Command |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 12938 | 12938 | 512 | 93.89950 | 245.79056 | 240.63271 | 38814 | 39 | 161.29977 | 3 | `ac75355` | training command above |
| 26268 | 13330 | 512 | 87.28461 | 257.76961 | 252.24138 | 39990 | 42 | 158.53862 | 3 | `ac75355` | same uninterrupted command |

Aggregate completed-prefix results (`short-run-summary.json`, reproducible with
`python docs/briefs/06-artifacts/summarize_short_run.txt`): 26,268 decisions / 1,024 games,
78,804 processed rows / 81 optimizer steps, three effective epochs. Native collection
181.18411 s = 144.97960 decisions/s; optimization 492.87409 s = 159.88668 processed rows/s.
Mean optimizer-step wall, including forward/loss/backward, is 6.08487 s. The prefix from
command launch through the last modification of the completed state is 700.389269 s:
37.50486 decisions/s and 1.46204 games/s. This includes startup and checkpoint writes.
Native collection/update/other wall shares are 25.869% / 71.897% / 2.234%.

The update is larger. Forward calls sum to 428.29906 s, 86.898% of optimization and
85.054% of native full-update time. Loss/backward/optimizer remainder is 64.57081 s,
12.823% of full-update time; assembly/diagnostics occupy the remaining 2.123%. One live
stack during the second update was in `BeliefModel._permute_hidden_moves`, whose Python
loop converts individual GPU tensor elements to bool/int. This is a location sample,
not a measured function share (`update-stack-snapshot.txt`). No belief/model change was made.

Sampled peak device use over the whole attempt, including unfinished work: 11,561 MiB;
peak learner allocation over completed updates: 10,432,924,160 bytes. Minibatch 1,024
completed without activation checkpointing. The raw attempt ran 995.824165 s and exited
-15 at the owner's stop request; 295.434896 s after the latest completed state are discarded,
not silently credited as completed work. All-attempt retained throughput is 26.37815
 decisions/s. Both spent-time and completed-prefix scopes are retained.

A 30-hour single-GPU week at the initial completed-prefix rate projects to 4,050,525
learner decisions and 157,901 games. This is arithmetic on the initial fixed-team smoke,
not measured steady-state throughput, learned performance, or general self-play speed.

Checkpoint pairs for 12,938 and 26,268 decisions are on the controlling computer, with
successful ZIP CRC checks (`off-runtime-checkpoints.json`). Weights and full optimizer/RNG
state are ignored binaries; the text evidence is committed. The partial third update is
not a resumable boundary. No actual runtime-drop/resume experiment was performed.

The runtime was explicitly released and the CLI reported no active sessions by 09:49:54
UTC (`runtime-released.json`). Allocation accounting: prior private jobs at most 0.937222
GPU-hours; first standard T4 at most 0.616667; replacement at most 0.887222. Total upper
bound **2.441111 GPU-hours**. The budget remained three; no extension was used. Local
backup/deadline helpers were stopped. The short timing request is complete; the original
full learning check and complete same-runtime matrix remain unverified.

### End-to-end games/second recomputation (smoke)

At `bc5bd85`, a read-only arithmetic check sums the completed-prefix collection events to
26,268 learner decisions and 1,024 games. Dividing by the recorded 700.389269-second wall
gives **37.504858 decisions/s and 1.462044 games/s**, including collection, PPO updates,
startup and checkpoint writes. Dividing the same retained work by the whole
995.824165-second attempt, including the discarded unfinished update, gives **26.378151
decisions/s and 1.028294 games/s**. These are two explicit timing scopes for the same short
smoke; the 100k learning/evaluation run remains unverified. No new GPU work was run.
Evidence: `06-artifacts/runtime-t4-counts/endtoend-recomputed.json`; source training commit
`ac75355`, command unchanged from the preceding rows.

## 2026-11-14: review of brief 06

The branch `work/collect-gpu` (head `bc5bd85`) was reviewed and merged into `work/roadmap`. The
owner shortened the brief to a timing check because GPU time ran out: two complete PPO updates
instead of the 100,000-decision run, no greedy evaluation, 10 of 16 probe rows. Everything here
is smoke. Checked in review:

- Every number in `docs/briefs/06-report.md` against
  `docs/briefs/06-artifacts/runtime-t4-counts/{short-run-summary.json,short-run-metrics.csv,stage1-summary.json}`
  and `gpu-budget.json`: the eight probe rates, 26,268 decisions, 1,024 games, 81 optimizer
  steps, 700.389 s, the three wall-time shares, 11,561 MiB, the 30-hour projection
  (37.50 × 108,000 = 4.05 million) and 2.441 GPU-hours. All agree.
- The only application change, `fog/eval/showdown_rollout_probe.py`: the pool probe now sends
  counts instead of whole rollout buffers across the worker pipe. The trainer is untouched.
- On the merged tree: `tests/test_rollout_probe_memory.py`, `tests/test_collection_gpu_job.py`
  and `tests/test_replay_slot_order_check.py` pass (8), `ruff check fog tests` is clean. Commits
  are authored as the owner; no account name or token was found in the artifacts.
- Not re-run in review: anything on a GPU; the regression test against the code before the fix.

What the review takes from it:

| Question | Answer | Evidence |
| --- | --- | --- |
| Does the GPU acting path run? | Yes | `cuda` probe rows completed at batch 32, 64 and 128; two PPO updates ran with `env.actor_device: cuda` |
| What does the GPU buy for collection? | About 30% on this machine | 2 logical CPUs, MaxDamage: 123 and 124 decisions/s on the CPU, 159 and 163 with the forward on the T4 at batch 128 |
| What does it buy for the update? | About 2× a four-core CPU | 160 processed rows/s on the T4 against 78 in brief 04's CPU run; 6.08 s per 1,024-row optimizer step |
| End to end? | No gain as the code stands | 37.5 decisions/s, inside brief 04's CPU-only range of 22 to 43 |
| Where does the update's time go? | The forward pass | 86.9% of optimization time; loss, backward and optimizer 12.8% of the update |

Reading. Two limits, neither of them the GPU's arithmetic. Collection is bound by the hosted
runtime's two slow CPU cores: per core it collects about 80 decisions/s with the GPU forward,
against 162 per core on the four-core sandbox with no GPU at all (647 on four cores, 2026-11-14).
The update is bound by Python inside the forward: 5.3 s for one 1,024-row forward of a
5.9-million-parameter model is far more than the matrix work costs on a T4. One live stack was
in `BeliefModel._permute_hidden_moves` (`fog/models/belief.py`), which loops over every row and
team slot in Python and reads single tensor elements, each read waiting for the GPU. Its share
is **not measured**; `_training_route` has the same per-row shape. Brief 03 measured 46% of the
CPU forward in the belief heads.

P0.5 stays open: the GPU path runs, learning under it was not shown. The estimate of 900 to
1,200 learner decisions per second for four cores and a GPU (2026-11-14) is neither confirmed
nor refuted, since this machine had two cores. Next for speed: profile the update's forward
on a GPU by function, then remove the per-row Python from the belief training path with a test
that the route and permutation distributions are unchanged.

## 2026-11-14: per-row Python removed from the belief training path (ROADMAP P0.5)

Asked for by the owner after brief 06 showed 37.5 decisions per second end to end on a T4.
Branch `work/fast-update`. Machine for every row: 4 CPU cores, no GPU, PyTorch CPU, pinned
Showdown `e440c4a`. Smoke timings; nothing here is a strength result, and **nothing here ran on
a GPU**.

**Where one PPO update spends its time on the CPU** (`python -m cProfile -m fog.train --config
configs/collect_gpu_check.yaml --trainer ppo --set device=cpu env.actor_device=cpu
env.rollout_workers=3 env.rollout_batch=32 ppo.games_per_chunk=96 ppo.total_decisions=2400
ppo.batch_decisions=2400 ppo.minibatch_size=512 ppo.epochs=1`; one update of 4,837 decisions,
10 optimizer steps; code at `d603920` before and at this section's commit after):

| Part of the update | Before, s | After, s |
| --- | ---: | ---: |
| Whole update | 47.5 | 43.6 |
| Forward passes | 22.3 | 20.3 |
| of which the two belief models (four passes per forward) | 15.4 | 12.1 |
| of which `BeliefModel._permute_hidden_moves` | 2.84 | 0.02 |
| of which the shared trunk | 3.6 | 4.2 |
| Backward | 24.5 | 22.4 |

**What changed.** `_permute_hidden_moves` looped over every row and team slot in Python and
read single tensor elements; on a GPU each read waits for the device. It is now one vectorised
pass. On 1,024 rows on the CPU the function takes 1.4 ms instead of 230 ms.

**How it was checked.** The random stream differs, so results are equal in distribution, not
bit for bit. `tests/test_belief_move_permutation.py` keeps the old loop as the specification:
only hidden, in-range moves move; every slot keeps its multiset; over 6,000 draws all twelve
arrangements of a two-slot case appear with equal frequency under both versions. The belief,
trainer and candidate tests pass (59 and 52), `ruff check fog tests` is clean, and the rollout
digest is unchanged (`--games 6`: `23405989f4665988`; acting does not call this function).

**Reading.** On the CPU this is 8% of an update. On the T4 it should be far more: the loop is
about 6,000 iterations of roughly fourteen small tensor operations per call, two calls per
forward, and brief 06's forward took 5.3 s per 1,024 rows. An estimate of 3 to 5 s of that
5.3 s is **not measured**; `docs/briefs/08-update-speed-gpu.md` measures it. The same function
runs in replay training whenever `loss.belief_chain_coef` is above zero, so brief 05's GPU job
was slowed by it as well (size not measured).

What the profile also shows: the two belief models are 59% of the forward on the CPU after the
fix and the trunk that serves policy and value is 20%. The PPO update does not use mixed
precision (`fog/training/bc.py` does; `fog/training/ppo.py` has no autocast). Both are levers
only if a GPU profile shows the same split.

## 2026-11-14: move-order leak removed and order-blind encoder (smoke)

Brief 07, branch `work/order-blind`; stage 1 preparation at `e195b26` from `d603920`.
All results in this section are smoke. No strength gate or roadmap task is complete.

| Run / check | Status and evidence | Command and producing commit |
| --- | --- | --- |
| Import | 22,000 games; pinned release; `07-artifacts/import-output.txt` | `python -m fog.scripts.import_metamon_replays --out data/metamon_gen9ou_r1500 --revision 792a86b5cdbcab9696032e947483b218453a0d4f --min-rating 1500 --after 2024-02-01 --max-games 22000 --max-shards 4 --max-download-mb 600`, `d603920` |
| Converter tests | 79 pass; Ruff clean; `07-artifacts/tests-stage1.txt`, `ruff-stage1.txt` | `.venv/bin/python -m pytest -q tests/test_replay_conversion_resume.py tests/test_replay_order.py tests/test_replay_slot_order_check.py tests/test_replay_parser.py tests/test_replay_candidates.py tests/test_split_replay_manifest.py tests/test_engine_mcts.py tests/test_replay_holdout_report.py tests/test_run_replay_blueprint.py`; `.venv/bin/ruff check fog tests`, `e195b26` |
| R0 / R1 | Pending: preserved original source and corrected corpus conversion; no training metrics claimed | Fixed recipe in brief 07; completed commands recorded in `07-artifacts/R0-execution.json` |

The correction permutes finalized moves and their provenance in independent per-slot seed
streams. Measured lead-position bias also calls for shuffling public preview order before
reconstructing either side; no hidden opponent information is introduced. Candidate-bank
duplicate detection continues to use the original log digest. Completed corrected conversions
can resume per game. The expanded slot diagnostic reports bench and lead positions, value
Brier/ECE, and intervals that resample games. See `docs/briefs/07-report.md` for limits and
pending comparisons. Stages 2–5 are not started; search, PPO and belief code are unchanged.

Stage 1 continuation, application commits `ef7c5c4` and `e691661`: a real-corpus comparison
exposed stale action-to-snapshot identity around Illusion and form changes. Candidate identity
is now checked before accepting policy labels, and rejected stale slots do not enter own-action
history. The final corpus is being rebuilt; the initial corrected corpus and partial rebuild
are explicitly superseded (`row-count-changes.json`, `guarded-partial-attempt.json`).
81 focused tests pass and Ruff is clean (`tests-identity-history.txt`, `ruff-identity-history.txt`).

The default-wait-policy R0 attempt was interrupted before completing an epoch; command,
elapsed time and exit are in `R0-default-attempt-execution.json`. Completed fixed-recipe runs
use `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0`, five torch threads. On recorded inputs and
perturbed parameters, all 241 output/gradient tensors match bit-for-bit between scheduling
settings (`runtime-equivalence.json`). No model, loss, batch size, seed or epoch count changed.
The three historical rollout digests fail already on the unchanged checkout; every corresponding
before/after digest matches on this runtime. A separate CPU-only build repeat also fails the
historical sequential reference (`digests-summary.json`). No cause is claimed.

Further converter checks before any R1 training: empty move columns now remain trailing for
species unable to fill four slots (`7890d3a`), and private-set completion runs in original
preview slots before sheets/provenance are shuffled (`531f29d`). This keeps imputation draws
paired with R0 instead of changing sampled attributes alongside order. 84 focused tests pass,
including installed Ditto move order, original-slot imputation pairing and full-converter
opponent-information isolation (`tests-final.txt`, `ruff-final.txt`). Earlier corrected-corpus
files marked `pre-guard` or `pre-empty-packing`, and the partial-attempt JSONs, are superseded;
no R1 model was trained on them. The paired corpus is being rebuilt.

### Completed original baseline (smoke)

R0 ran on the isolated original source at `d603920`, with the original rows. Eight epochs completed; the lowest holdout policy NLL selected epoch 7. All policy groups and game-resampled 95% intervals are in `07-artifacts/R0-slot-check-extended.json`; the original diagnostic output is retained separately. The retained-history group means no earlier move by the acting species in the stored history window, not necessarily its first move ever.

| Run | Training / holdout games | All stored NLL / top-1 | All shuffled NLL / top-1 | Retained-history top-1 stored → shuffled | Value Brier / ECE | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 (smoke, best epoch 7) | 1,000 / 2,000 | 1.7717 / 0.3652 | 2.2135 / 0.2269 | 0.7910 → 0.2543 | 0.3905 / 0.3604 | `d603920` | `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 /workspace/fog/.venv/bin/python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set bc.manifest=runs/order_blind/leaky_split/train_1000.parquet bc.holdout_manifest=runs/order_blind/leaky_split/holdout_manifest.parquet bc.epochs=8 bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce log.dir=runs/order_blind/R0` (cwd `/workspace/fog-order-baseline`) |

Value Brier 95% interval: [0.3761, 0.4044]; ECE: [0.3449, 0.3755]. The unchanged value head fails the constant-0.5 Brier reference of 0.25. No value repair has been attempted. The paired converter retains all 22,000 games with zero quarantines and the same fixed 2,000 holdout / 1,000 training IDs; raw rows are 103,944 / 52,955. R1 and its second seed are pending; no cross-run effect is claimed yet.

### Completed corrected-corpus comparison (smoke)

The paired conversion at `531f29d` and all fixed-recipe runs are complete. R0's
original rows and checkpoint are preserved. Both R1 seeds completed eight epochs; policy NLL
selects the checkpoints below. Every result remains smoke. Intervals resample entire games,
with both perspectives together. The H group is defined by the retained history window,
not lifetime first use. Complete group metrics and intervals are in the per-run slot checks.

| Run / seed | Best policy epoch | All shuffled NLL / top-1 | H stored → shuffled top-1 | Value Brier / ECE | Producing commit | Command |
| --- | --- | --- | --- | --- | --- | --- |
| R1 / 20261201 | 8 | 1.9239 / 0.2641 | 0.3371 → 0.3352 | 0.3920 / 0.3660 | `531f29d` | `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 /workspace/fog/.venv/bin/python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set bc.manifest=runs/blueprint/split/train_1000.parquet bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet bc.epochs=8 bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce log.dir=runs/order_blind/R1 seed=20261201` (cwd `/workspace/fog`) |
| R1-seed2 / 20261202 | 7 | 1.9290 / 0.2668 | 0.3297 → 0.3277 | 0.3975 / 0.3712 | `531f29d` | `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 /workspace/fog/.venv/bin/python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set bc.manifest=runs/blueprint/split/train_1000.parquet bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet bc.epochs=8 bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce log.dir=runs/order_blind/R1-seed2 seed=20261202` (cwd `/workspace/fog`) |

R1 value Brier 95% interval [0.3773, 0.4058]; ECE [0.3507, 0.3807]. R1-seed2 value Brier 95% interval [0.3828, 0.4118]; ECE [0.3558, 0.3865].

All-row nll_shuffled: R1−R0 = -0.289550; R1 absolute two-seed gap 0.005092, twice that gap 0.010184; effect exceeds twice the gap: True. All-row top1_shuffled: R1−R0 = 0.037167; R1 absolute two-seed gap 0.002723, twice that gap 0.005446; effect exceeds twice the gap: True. Only shuffled columns support this comparison; the original stored holdout is leaky.
The identity guards change the retained rows as well as order. The unchanged value head still
fails the constant-prediction Brier reference of 0.25; no repair is claimed.

Final conversion: all 22,000 games, no quarantines, 1,126,792 accepted rows; same 1,000 training
and 2,000 holdout game IDs. The H slot-zero label share is 0.246984 (95% game interval
[0.240865, 0.253005]), just below the brief's expected 0.25 lower point bound; the interval
includes one quarter and the seed was not retuned. Source commands, row counts, checkpoints
and every measured group are in `docs/briefs/07-artifacts/stage1-summary.json`,
`stage1-execution.json`, `row-intervals.json` and the per-run CSV/JSON.

Final validation: 84 focused tests plus 15 additional tests pass; the latter command is
`OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 .venv/bin/python -m pytest -q tests/test_metamon_native_comparison.py tests/test_replay_supervision.py`
(`tests-shared-imputer.txt`). `.venv/bin/ruff check fog tests` is clean. Both an
untrained original checkpoint and the original trained epoch-1 checkpoint load strictly;
trained policy/value outputs are finite on recorded rows. Model, encoding, loss, search,
PPO and belief source are unchanged. Final sequential / batched / MaxDamage digests are
`0d026fed7728c0c4` / `ef414aee0fe96bad` / `40644081cca71ebe`, matching before-change runs on
this runtime. The required historical references remain **unreproduced**. Negative repeats
include unchanged reference commit `21e57fa`, CPU-only Torch and alternate CPU kernels;
`digests-summary.json` has exact commands, exits and hashes. The discrepancy is unisolated;
the historical constants were not changed.

This handoff ends after the standalone converter stage. R2, R3, R4 and its second seed,
order-blind permutation acceptance tests, logistic and auxiliary value targets, mechanical
features, rare-move groups, trained leads and play were **not run or implemented**. Own replay
PP remains zero. Full Illusion/Transform reconstruction beyond identity guards and lifetime
first-move statistics remain unverified. No phase task or strength gate is completed.

## 2026-11-15: review of brief 07, stage 1

The branch `work/order-blind` (head `e92a565`) was reviewed and merged into `work/roadmap`. The
worker stopped after stage 1 (the converter); stages 2 to 5 of the brief were not started and
are carried into `docs/briefs/09-order-blind-model.md`. Everything is smoke. Checked in review:

- Every number in the two tables of `docs/briefs/07-report.md`, the value intervals, the seed
  gaps and the row counts against `docs/briefs/07-artifacts/stage1-summary.json`,
  `rows-before.json`, `rows-after.json` and `lead-order-after.json`. All agree.
- The converter, parser and slot-check diffs, read line by line. The move shuffle draws from
  its own stream, so every other imputed field is the same draw as before; empty move columns
  stay last; team order is shuffled by permuting the public preview lines, so sheets, both
  views, switch labels and history follow one order; a move or switch label is rejected when
  the slot it names no longer holds that move or species in the encoded observation.
- On the merged tree, on the reviewer's machine: the six test files covering the change pass
  (43), `ruff check fog tests` is clean, and **all three historical rollout digests are
  reproduced** (`23405989f4665988`, `0efd4319547a4b3b`, `338cfca82f31b181`).
- Not re-run in review: the conversion of 22,000 games and the three trainings.

On the digests. The worker could not reproduce the historical values even with unchanged
source and reported that as open. It is the machine: the worker ran the same PyTorch version
(2.13.0) on an AMD EPYC, the reviewer on an Intel Xeon, and the digest hashes floating-point
outputs. From now on a digest is a before-and-after check on one machine; the historical
values are valid only on the reviewer's kind of CPU. No code defect is indicated.

What the review takes from it (1,000 training games, 2,000 holdout games, 8 epochs):

| Finding | Evidence |
| --- | --- |
| The leak was most of what the old model knew about unmoved Pokémon | Old rows, old model, rows where the acting Pokémon has no earlier move in the kept history: top-1 79.1% as stored, 25.4% with the four slots shuffled |
| Old holdout accuracy was inflated by about ten points overall | Top-1 on all rows 36.5% (old rows, stored) against 26.5% and 26.7% (corrected rows, two seeds) |
| Training on corrected rows gives a better policy where it counts | Scored with shuffled slots, which is what live play resembles: NLL 2.2135 (old) to 1.9239 and 1.9290 (corrected); top-1 22.7% to 26.4% and 26.7%. Twice the seed gap is 0.010 NLL |
| The corrected rows no longer carry the bias | Label share per move slot 25.1 / 24.8 / 24.8 / 25.3%; unmoved rows with the label in slot 0: 98.4% before, 24.7% after; corrected models score the same stored and shuffled (differences of 0.2 points or less) |
| Team order carried a smaller bias too | The lead was team position 0 in 24.5% and 25.0% of games (p1, p2); 16.8% and 17.3% after the preview shuffle |
| The value head is unchanged and still worse than a constant | Brier 0.39 in all three runs against 0.25 for a constant |

Two cautions the report states and the review repeats: the corrected conversion also drops 93
holdout rows through the new identity guard, so old against new is not a pure ordering
comparison; and "has not moved yet" means within the kept history window, not in the game.

Consequence for brief 05. Its G20 and G100 runs trained on the old rows, so their holdout
accuracy (G20: NLL 1.279, top-1 54.4%) is inflated and is not comparable with any number on
corrected rows. Their timing, value and play measurements stand as measured. The next replay
run on a GPU should use the corrected converter and the belief-loop fix of 2026-11-14.

## 2026-11-15: plan changes after the leak

No measurement. This section records why `ROADMAP.md` changed on this date, as `AGENTS.md`
requires, and what the owner asked for: a plan that other people and agents can carry on
without the session that wrote it.

| Change | Why |
| --- | --- |
| The work is run from `docs/briefs/QUEUE.md`, reviewed by `docs/briefs/REVIEW.md` | Reviews were done by one session that will not be available. The procedure it followed is now written down and can be given to any agent |
| P0.7 accepts a replay-derived team pool (brief 10) | The forum bank has been blocked on an owner action since 2026-11-12 and it blocks G0 and G1. The replays already hold thousands of real high-rated compositions with public previews. Limits: unrevealed items, spreads and moves are sampled from usage, so teams can be less coherent than a person's; tier claims should use the forum bank's test split if it is ever collected |
| New tasks P2.2a and P2.2b | The move-order leak (ledger, "review of brief 07, stage 1") and four other differences between replay rows and live observations found by reading the code on 2026-11-14 |
| Gate G2 also requires value Brier below 0.24 | A tightening. ECE below 0.05 is met by predicting 0.5 everywhere (brief 01's six-epoch model: ECE 0.032, value MSE 0.92 to 1.05). No current model benefits: every value head measured on corrected rows scores 0.39 |
| P5.4 points to `docs/equilibrium-training.md` | Plain self-play has no convergence guarantee in a zero-sum game; the anchored form does in the tabular case, and the trainer already has the anchor term. A Kuhn poker check is in that document. Self-play stays frozen until G3 |
| Parking lot: regret-matching selection in tree search; sampled-candidate equilibrium search on small pools; removing parameters with no gradient | Recorded so they are not rediscovered; none is scheduled |

Order of work from here: the replay policy line (briefs 09, 12, 13, ending at G2) and the
search line (briefs 10, 11, ending at G1) in parallel; speed (briefs 08, 14) alongside; then
P3. Reading behind the order: EngineMCTS at a quarter of its budget already beats
SimpleHeuristics 0.88 in a 100-game smoke, while the best replay model (20,000 games, old rows)
scores 0.52 in a 400-game smoke. The search agent is the nearest gated result and the teacher
for P3, and it has been waiting only for a team pool.

## 2026-11-14: replay-derived team pool and anchor round-robin

Brief 10, stage 1 complete at `3506a39`; stage 2 pending. Pool statistics below are smoke
construction diagnostics, not agent-strength evidence. Evidence: `docs/briefs/10-artifacts/`;
exact pool and per-team evidence counts: `data/teams/replay_v1/`.

Recreated the pinned import with `uv run --no-sync python -m fog.scripts.import_metamon_replays
--out data/metamon_gen9ou_r1500 --revision 792a86b5cdbcab9696032e947483b218453a0d4f
--min-rating 1500 --after 2024-02-01 --max-games 22000 --max-shards 4 --max-download-mb 600`.
It reproduced 22,000 games and the earlier rejection counts: 77,225 below rating, 4,176 invalid
metadata, 495 outside dates, 840 other format (`import-output.txt`). No decision-row conversion
was needed: the fast public-set extractor reuses the existing submitted-set evidence handling.
Pinned Showdown: `e440c4a18385274f10c405d0b158b6a962ce6d94`; CPU only.

Build: `uv run --no-sync python -m fog.scripts.build_replay_team_pool --seed 0`, at `3506a39`.
One completion per side seeded by replay id, side and build seed; no retries after rejection.
A persistent Node process calls exactly the pinned CLI's `Teams.import` / `validateTeam` API.

| Filter (smoke construction counts) | Remaining / dropped |
| --- | --- |
| Imported games / sides | 22,000 / 44,000 |
| Rated at least 1700 | 9,850 sides; 34,150 dropped |
| Six known preview species | 9,849; 1 dropped |
| At least 14 revealed moves | 1,839; 8,010 dropped |
| Renderable completion | 1,837; 2 unknown-item completions dropped |
| Pinned Gen 9 OU validator accepts | 1,439; 398 dropped |
| Most-revealed valid side per species multiset | 999 compositions; 440 duplicates dropped |
| Before seeded caps (train / validation / test) | 796 / 141 / 62 |
| Committed teams and distinct compositions | 600 / 141 / 62 |

Temporal exclusion uses **all** imported known previews, including sides below 1700 or with too
few moves; no test composition appears in any non-holdout game. The fixed 2,000 holdout ids were
checked against the newest games by `(uploadtime, id)`. Test compositions occur only there.
Validation is a seeded 15% of the other compositions, before caps. Split manifests:

| Split | SHA-256 of `<split>/manifest.json` |
| --- | --- |
| test | `eddd528afe9581d1f0b4a56a0b66041f2fd4ab19bcec8870be9a1e768d2a8eca` |
| train | `35e839989802d031120cf0aff2f5b8b632cd5087ba3d02b902cdda7750fda925` |
| validation | `fa8588caad7cc52a628da650b25ca2176865afe81ca680b0fe010f1744a8badd` |

Smoke evidence completeness: the 803 committed teams average 27.2777 revealed / 62.7223 sampled
of 90 scalar fields, including six exact species; 15.7422 of 24 moves are revealed. Of the 84
non-species fields, 21.2777 (25.33%) are revealed. All nature/EV fields are sampled. Unrevealed
items, abilities, moves and Tera types are usage samples, so these teams can be less coherent
than teams built by a person. This is not a bank of exact submitted human team sheets.

Verification: all 803 teams validate; all composition groups are disjoint; the full fresh rebuild
(`--out runs/replay_team_pool/reproduced --work-dir runs/replay_team_pool/fresh-build --seed 0`)
reproduced all 807 pool/manifest files byte for byte. Exported pastes contain canonical species
names and set fields only; manifests allow only replay ids, composition ids and numeric counts,
never original nicknames or player fields. `tests-output.txt`: 34 passed (including the raw
older-preview check); `lint-output.txt`: `ruff check fog tests` passed. The pytest cache-option
warning comes from disabling the cache plugin. No hosted workflow or GPU job was started.

Anchor round-robin: pending on the validation directory. A3 is unavailable: `runs/` has no
retained imitation checkpoint. No substitute will be used. P0.7 remains unchecked until A0–A2
have completed the required games; G0's A3 clause remains unmeasured.

### Stage 2 completed: validation anchors

The round-robin ran at `66a1e29`, on the same committed validation pool (141 teams). A0 is
Random, A1 is the existing MaxDamage baseline (poke-env MaxBasePowerPlayer), and A2 is
SimpleHeuristics. Factories return the existing scripted names, so the local runner's seeded
randomness and resume behavior are unchanged. Each pair played 250 four-game blocks crossing
both team assignments and both player positions: **1,000 matched games per pair**, 3,000 total.
All games resolved; zero failures, genuine draws or truncations. Test teams were not played.

Printed rows, copied verbatim from `10-artifacts/anchors-output.txt` (full per-game results and
protocol in `10-artifacts/anchors.json`):

| Date | Candidate | Opponent | Backend | Games | Win rate [95% Wilson] | s/decision | Fallbacks | Commit | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-14 | A0 | A1 | showdown_direct | 1000 | 0.102 [0.085, 0.122] (102-898-0) | n/a | n/a | `66a1e29` | `python -m fog.eval.showdown_direct_selection --pool data/teams/replay_v1/validation --device cpu --controller A0=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A0={"name":"random"}' --controller A1=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A1={"name":"maxdamage"}' --controller A2=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A2={"name":"simpleheuristics"}' --head-to-head --games 1000 --seed 0 --max-turns 800 --progress-every 40 --resume --out runs/replay_team_pool/anchors.json` |
| 2026-11-14 | A0 | A2 | showdown_direct | 1000 | 0.034 [0.024, 0.047] (34-966-0) | n/a | n/a | `66a1e29` | `python -m fog.eval.showdown_direct_selection --pool data/teams/replay_v1/validation --device cpu --controller A0=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A0={"name":"random"}' --controller A1=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A1={"name":"maxdamage"}' --controller A2=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A2={"name":"simpleheuristics"}' --head-to-head --games 1000 --seed 0 --max-turns 800 --progress-every 40 --resume --out runs/replay_team_pool/anchors.json` |
| 2026-11-14 | A1 | A2 | showdown_direct | 1000 | 0.175 [0.153, 0.200] (175-825-0) | n/a | n/a | `66a1e29` | `python -m fog.eval.showdown_direct_selection --pool data/teams/replay_v1/validation --device cpu --controller A0=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A0={"name":"random"}' --controller A1=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A1={"name":"maxdamage"}' --controller A2=fog.eval.showdown_direct_selection:scripted_controller --controller-kwargs 'A2={"name":"simpleheuristics"}' --head-to-head --games 1000 --seed 0 --max-turns 800 --progress-every 40 --resume --out runs/replay_team_pool/anchors.json` |

Thus MaxDamage won 898/1,000 against Random; SimpleHeuristics won 966/1,000 against Random and
825/1,000 against MaxDamage. These meet the brief's minimum game count. The scripted controllers
have no decision-timing or fallback counters, so the printed columns are `n/a`; the match wall
seconds are saved in the JSON. These rows do not establish a learned-agent strength tier.

Correction to the stage-1 filter description above: the two unrenderable completions were **not
unknown-item completions**. Both were Ditto sides (`gen9ou-2139396661`, `gen9ou-2443965990`, p2):
the one-move learnset leaves three move columns unknown. The builder conservatively drops such
incomplete sheets rather than inventing moves or retrying. The counts, committed pool and hashes
are unchanged. The validator-equivalence test now explicitly checks an illegal item as well as a
legal team; final focused checks remain 34 passed and repository lint clean (updated artifacts).

P0.7 is complete under brief 10's A0–A2 acceptance rule. A3 is unavailable, with no substitute;
G0 remains open (including its A3 clause and remaining throughput work). G1 and tier claims were
not attempted. All pool files are unchanged from `3506a39`. No checkpoint training, hosted
workflow, forum fetch or test-split play was performed.

## 2026-11-14: replay blueprint on a GPU at 3,000 to 100,000 games (smoke)

Brief 05 is in progress on `work/p2-gpu-scale`; no gate is claimed. Implementation commit
`bd6db4161cb3d6d563ae4a98f9415771601d4d8c` adds CPU-only preparation and independent,
resumable best-value checkpoint selection, both default off. No model, loss, schedule or filter
changed. Affected validation: `uv run --no-sync python -m pytest -q -p no:cacheprovider --tb=short
tests/test_bc_resume.py tests/test_run_replay_blueprint.py tests/test_bc_metrics.py
tests/test_bc_scope.py tests/test_replay_holdout_report.py`: **71 passed**, 37.50 s, outside the
command sandbox; `uv run --no-sync python -m ruff check fog tests`: clean. Earlier sandbox
attempts timed out in the dependency's asynchronous exit and are retained as failures in
`docs/briefs/05-artifacts/validation-attempts.json`.

| Run (smoke) | Command | Code commit | Games / interval | Status |
| --- | --- | --- | --- | --- |
| CPU preparation, first set | `uv run --no-sync python -m fog.scripts.run_replay_blueprint --games 22000 --work-dir runs/p2_gpu_22000 --device cuda --batch-size 1024 --epochs 18 --prepare-only` | `bd6db41` | pending / n/a | import finished, conversion running; no training |
| CPU preparation, second set | `uv run --no-sync python -m fog.scripts.run_replay_blueprint --games 112000 --work-dir runs/p2_gpu_112000 --device cuda --batch-size 1024 --epochs 18 --prepare-only` | `bd6db41` | pending / n/a | import running; no training |

Both preparations began with these implementation bytes before the code commit was made.
Kaggle client authentication and job-list access succeeded. `kaggle quota --format json`
reported 0.00 GPU-hours used and 45.00 available; this brief remains capped at 20. No GPU job
submitted and no GPU checkpoint or new play result exists at this checkpoint. Historical B
reference files, budget, test results and job template: `docs/briefs/05-artifacts/`.

### Stage 0 first set completed (smoke; no training)

The first preparation command above completed under the same `bd6db41` application bytes:
22,000/22,000 games accepted, zero quarantines; train **20,000 / 1,023,525 rows**, holdout
**2,000 / 104,037 rows**, exact fixed holdout IDs. Nested 1,000 / 3,000 subsets match
**52,983 / 156,597 rows**, splitter seed 20260804. Newest training upload 1771553139 is before
oldest holdout upload 1771561053. Import / conversion / split wall: **92.066 / 1,000.359 /
3.294 s**. These are CPU preparation walls with concurrent setup/tests and the larger import,
not an idle throughput benchmark. Repeating the identical prepare command skips all stages.
Commands and complete counts/timers: `docs/briefs/05-artifacts/prepare-22000.json`, `prepare-22000-run.json`,
`prepare-22000-quality.json`, `prepare-22000-skip.txt`, plus import/split text outputs. Private upload is next.
The 112,000-game import also reached its target and conversion is running; usable training
count is not yet known. No GPU time or play result is claimed.

### Private G3 pilot submission (smoke, result pending)

The 468,727,506-byte CPU work archive uploaded privately as `fog-p2-gpu-22000-20261114` in
186.132 s. The official dataset metadata/status reads return HTTP 403,
`Permission 'datasets.get' was denied`; creation was accepted, and privacy was requested
explicitly. At code `f67c9587b75ffa838a5cffec0ca0c60b555c5c9f`,
`KaggleApi.kernels_push(acc="NvidiaTeslaT4", timeout="600")` accepted private script
`fog-p2-g3-stop-20261114`, version 1, no invalid dataset sources. Its G3 config is 3,000 games,
18 epochs, batch 1,024, CUDA, separate best-value checkpoint enabled; a completed-epoch hook
will deliberately stop after epoch 2. Initial expected session time 3–6 minutes is a planning
estimate from the prior CPU feed ceiling, **not a measured GPU rate**; hard cap is 10 minutes,
conservatively 1/3 GPU-hour if two devices are billed. Actual use and training results pending.
Source, accepted response and budget: `docs/briefs/05-artifacts/`.

### First hosted attempt failed before training; corrected retry running (smoke)

`fog-p2-g3-stop-20261114` ended **ERROR**, not a training result. Setup took **126.478 s**,
Python **3.13.16**, PyTorch **2.11.0+cu130**, CUDA **13.0**, GPU **Tesla T4**; the job
then failed at **132.475 s** because Kaggle had expanded `replay-work.tar.gz` into ordinary
files. Quota used **138.869 s = 0.0385747222 h**, including session overhead. See
`G3-stop-job-runtime.json`, `G3-stop-setup-log.txt`, `gpu-hours.json`. Full kernel metadata
confirms `is_private=true`; the owned dataset listing also confirms the dataset is private.

Restore fixes `6dcbf51` / `530ae08` support both layouts and make restored manifests writable;
**3 tests pass**, lint clean, no training/data semantics change.
`KaggleApi.kernels_push(acc="NvidiaTeslaT4", timeout="600")` accepted private retry
`fog-p2-g3-stop2-20261114`, version 1, code `530ae08cac2f27d3e53c0cfa535e44e8f56e7074`,
same two-epoch interruption plan. Expected **4–7 min** from measured setup plus still-unmeasured
training/export, capped at **10 min**. Retry is running; no training number claimed.

### Corrected pilot capped without output; safer input and export handling (smoke)

`fog-p2-g3-stop2-20261114`, code `530ae08`, ended **CANCEL_ACKNOWLEDGED** at its requested
600-second cap; no training log/checkpoint was returned (output fetch HTTP 404). The last live
event is the successful runtime probe at **98.617 s**; the stage reached afterward is unknown.
Final quota total **749.868 s / 0.208296667 h**, hence this attempt used **610.999 s** including
platform overhead. It is a failed/partial attempt, never an 18-epoch result.

`29ffeb4` adds opaque-archive input and an internal worker deadline reserving export time;
**5 local tests pass**, including preserving a saved marker after the worker is stopped.
Selected policy/value weights are now exported separately from the resume archive (`3f0f59a`).
No model, loss, schedule or filter changed. A redundant new dataset destination was rejected
by automatic approval review as outside the approved sensitive-data destination scope.
The safer update of existing private `fog-p2-gpu-22000-20261114` was permitted: same archive
bytes, filename `replay-work.bin`, old versions retained, **102.839 s** upload/version wall.
This avoids archive expansion; its effect on hosted training remains unmeasured.
See `prepare-22000-version2.json`, input tests and the exact scripts/status/log artifacts.

### Opaque-input pilot accepted (smoke; measurements pending)

The owned listing confirms versioned `fog-p2-gpu-22000-20261114` remains private and now
contains exactly one `replay-work.bin`; its bytes are identical to the original archive.
`KaggleApi.kernels_push(acc="NvidiaTeslaT4", timeout="600")` accepted private
`fog-p2-g3-stop3-20261114`, version 1, code `106034c87187d351e01873ee6e99213370ab9e3c`.
G3 remains 3,000 games / 18 epochs / batch 1,024 / CUDA, with the deliberate epoch-2 stop.
The worker deadline begins export by 480 s, ahead of the platform cap. Initial estimate
**4–7 min** uses the measured 97–126 s setup range; extraction/training speed remains unknown.
Prior ended attempts used 749.868 quota seconds. Exact source, submission, input verification
and budget are retained; no new training number is claimed.

### Opaque restore succeeded; first epoch still incomplete (smoke)

`fog-p2-g3-stop3-20261114` completed its **outer job** and exported saved work, but training
was **interrupted_for_export before epoch 1 completed**. Installation **92.656 s**, input
restore **21.109 s**, export began at **480.315 s**. The returned training log has the dataset
line and initial validation only; no completed epoch rate or GPU-memory result exists. Total
settled quota **1,295.563 s / 0.359878611 h** across the three attempts; this job used
**545.695 s**. Sources: `G3-stop3-job-runtime.json`, `G3-stop3-train.txt`, status/full log.

Private `fog-p2-g3-stop4-20261114` was accepted at code
`3b81a6154e1cc597233acba4670e266054decf8a`, version 1, same G3 18-epoch settings, now stopping
after the first completed epoch. Command: `KaggleApi.kernels_push(acc="NvidiaTeslaT4",
timeout="1800")`; internal export deadline 1,500 s. Planning range **9–27 min** follows the
observed incomplete 365-second worker attempt plus setup/restore/export, not a measured
epoch rate. Its maximum fits the roughly 38 minutes left in stage 1's one-hour allowance.
No second GPU job is active. Exact source and submission are retained.

### First completed GPU epoch and complete CPU preparation (smoke)

| Run (smoke) | Train / holdout games | Epochs | Batch/device | Holdout NLL / top-1 | Holdout value MSE | Train rows/s | Code | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G3 partial | 3000 / 2000 | 1 of 18 | 1024 / Tesla T4 | 2.016241 / 0.212143 | 0.998706 | 604.194 | `3b81a61` | `python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set device=cuda bc.manifest=/kaggle/working/replay-work/split/train_3000.parquet bc.holdout_manifest=/kaggle/working/replay-work/split/holdout_manifest.parquet bc.batch_size=1024 bc.epochs=18 bc.resume=true bc.selection_metric=action_ce bc.keep_best_value=true bc.streaming=true bc.early_stopping_patience=0 log.dir=/kaggle/working/replay-work/G3` |

This job deliberately stopped after epoch 1. Training 156,597 rows took 259.186 s; initial and
epoch validation took 160.716 / 154.323 s. Peak PyTorch allocation **2,619,046,912 bytes**
(2.439178 GiB), CUDA device 0, two T4s visible. Python 3.13.16 / PyTorch 2.11.0+cu130 /
CUDA 13.0; two CPU threads. Installation **137.037 s**, wall before export **746.991 s**;
job quota **824.787 s**, total ended-job quota **2,120.350 s = 0.588986111 h**. Raw files:
`G3-stop4-*` and `G3-first-epoch-summary.json`. Intervals are n/a for these offline diagnostics.

At the measured rate and validation time, G3 18 epochs project to **7,603.822 s**, excluding
setup/writes, above stage 1's one-hour cap; G20 projects to **33,431.119 s**. These are
projections, not completed runs. Per AGENTS.md's impossible-task rule, ROADMAP now states the
smallest clear next step: retain partial G3 and verify continuation within the unchanged hour,
then prioritize G20 with default batch 1,024 under the unchanged 20-hour total. No complete
G3-versus-B or GPU batch-256 comparison, batch-quality conclusion or gate is claimed.

Private `fog-p2-g3-resume-20261114` was accepted, code `e05d5d1`, attaching the preceding
private job's saved work. Same 18-epoch config, deliberate stop after epoch 2, internal export
deadline 1,200 s, platform cap 1,400 s. Expected 15–20 min from measured epoch/validation and
setup; that cap keeps all calibration below one quota-hour. Reports of both partial selected
checkpoints are requested. Continuation proof remains pending returned metrics.

CPU preparation of 112,000 games is complete: **101,928 / 5,217,022 training games/rows**,
**2,000 / 104,037 fixed holdout games/rows**, **8,072** newer games excluded, zero quarantines.
All 20,000 first-set training IDs are contained in the larger set; all **2,000** holdout NPZs
are byte-identical. Max training upload **1771559041** precedes min holdout **1771561053**.
Rating floor stays **1500**, no wider dates or lower floor required. Import / conversion / split
wall **789.940 / 4,452.866 / 2.740 s**, with initial raw/cache reuse and local check contention,
not an idle throughput benchmark. Commands/counts/timers: `prepare-112000*`.
The **2,385,762,558-byte** opaque archive is uploading to the brief's second private dataset.

### Real continuation verified; G3 calibration closed as partial (smoke)

`fog-p2-g3-resume-20261114` ended **COMPLETE** with training deliberately stopped at epoch 2.
The raw `[0,1]` metrics CSV is an **exact byte prefix** of `[0,1,2]`; JSON/CSV proof is in
`G3-resume-check.json` and both before/after files. Both selected checkpoints are epoch 2.
Aggregate training rate **613.027 rows/s** (epochs **604.194 / 622.123**), epoch-validation
**154.323 / 149.778 s**. Peak allocation **2,619,988,480 bytes**. Both reports: top-1
**0.300391**, policy NLL **1.895671**, value MSE **0.979114**, Brier **0.244778**, ECE
**0.033124**. This is **2 of 18 epochs**, not a completed G3 or a valid full-schedule batch/device
comparison. Command/config/source: `G3-training-runtime.json`, submitted script at `e05d5d1`.

The continuation used **1,044.105 quota seconds**; complete calibration total
**3,164.455 s = 0.879015278 h**, including every failed/partial setup. No calibration gate
or GPU-256 quality conclusion is claimed. G20 retains the brief default batch 1,024, with
that limitation explicit. Projection using the slower measured epoch and conservative
validation/overhead allowance is **9.596 h** for G20; planned platform cap **11 h**, internal
export deadline **10 h 40 min**, all within the unchanged 20-hour total. `G20-plan.json` holds
the inputs and arithmetic.

Timing clarification for the preceding row: epoch-minus-validation was **259.186 s**;
the actual training timer from rows/rate was **259.183 s**. The 0.003 s difference is other
epoch overhead. Projections already used the exact rows/rate value and are unchanged.

Both partial G3 weights were uploaded to the explicitly required private checkpoint dataset
`fog-p2-gpu-checkpoints-20261114`; the one-command download was executed and both byte sizes
and SHA-256 values match `checkpoints.json`. It contains partial G3 only; G20/G100 await training.
See `checkpoint-dataset-create.json` and `checkpoint-download-check.json`; no account identity
or weights are committed.

The second corpus upload was interrupted by a workspace restart. An owned-dataset check
confirmed no object had been committed; retry to the **same prescribed destination** succeeded
in **47.825 s**, reusing uploaded content. This is a retry wall, not a cold-transfer benchmark;
the earlier interrupted wall is unknown. Its private owned listing confirms exactly one
**2,385,762,558-byte** `replay-work.bin`. CPU preparation and both private input uploads are now
complete (`prepare-112000-{recovery-check,upload,private-check}.json`).

### G20 submitted (smoke; no completed result yet)

Private `fog-p2-g20-20261114`, version 1, was accepted at code
`7a234972abf8d4c6dde8629353588aa2558f1fd0`: **20,000 games / 1,023,525 rows**, **18 epochs**,
batch **1,024**, CUDA, both checkpoint-selection options active, no early stopping. Command:
`KaggleApi.kernels_push(acc="NvidiaTeslaT4", timeout="39600")`; internal export deadline
**38,400 s**. Expected **9.596 GPU-hours** from the conservative measured inputs in
`G20-plan.json`; cap **11 h**. All prior jobs ended before submission, at total quota
**0.879015278 h**. Exact source/submission and verified private status are retained.

`docs/replay-blueprint-scale-run.md` now gives actual hosted measurements, CPU preparation
counts/times, private input slugs, the exact job source, concrete continuation steps and the
tested checkpoint download. G20 and G100 are still pending; no curve, play or gate result is
claimed by a running job.

### Available value tables and input checks while G20 trains (smoke)

At `8b88e149161433632aadb37dd3301e5715cc58de`, read-only CPU checks used the
existing `parse_replay_policy_versions(..., imputations=1, seed=20260804)` reconstruction,
with a separate protocol-player/winner reader. A seeded sample of 64 nested G20 training
and 64 fixed holdout games contained 6,283 accepted serialized rows: no per-side outcome-sign
mismatch. Holding each legitimate imputed own sheet fixed, deleting the second half of each
log left all earlier observation fields and opponent reveal tokens identical across 3,150
rows. This is sampled smoke, not a full-corpus isolation proof. The independent full prepared
manifest/protocol check covered 101,928 training plus 2,000 holdout games with zero winner or
player-order discrepancies (24.984 s). Commands: CPU Python calling the existing parser and
NumPy shard reader, and a separate raw-log/Polars manifest comparison; no optimizer, filters
or data changed. Results and method are in `value-input-checks.json` and
`outcome-manifest-check.json` under `docs/briefs/05-artifacts/`.

The named `fog/eval/leakage_metrics.py` contains readability/reveal statistics, not a
privileged-input check. The chronological-prefix comparison above checks the intended input
isolation directly; trained-checkpoint label perturbation remains pending G100. The value
learning criterion and G2 remain unverified. Existing B and partial G3 reports now feed
`curve-value-tables.json` and the complete ten-bin `reliability.csv`; unavailable historical
best-value details stay null. The unchanged holdout-report scorer runs on CPU even when its
config says CUDA; training validation uses CUDA, and both report passes consume session quota.

The portable `uv run --no-project --with kaggle==2.2.5 --with six python -c ...` owner download
was exercised against the existing private checkpoint dataset. Both initial G3 files matched
inventory bytes and SHA-256 (`checkpoint-portable-download-check.json`). It will be repeated
after adding G20/G100. No new GPU job was started; G20 continues alone.

### G20 live progress and partial G3 label routing (smoke)

G20 has saved epochs 1 and 2 of 18, with logged (rounded) holdout value MSE 0.894 and
0.878 and roughly 32-minute epochs including validation. `G20-progress.json` preserves the
live log events and observation time; these are not final selected-checkpoint metrics. The
existing private job continues alone with the same configuration/source pin. Do not compare
rounded aggregate losses with B's full-precision policy NLL.

At `fda1a94`, CPU inference from the partial G3 own-best value checkpoint gave identical
policy/value tensors after negating outcomes, replacing terminal auxiliary labels and
injecting optional privileged truth/action fields: one fixed 256-row holdout batch, both
in evaluation mode and in training mode with matched RNG. No optimizer ran. This checks
supervision/oracle routing, not every possible feature leak. `G3-label-perturbation.json`
records the result; repeat on the selected final-run checkpoint. Command: CPU Python,
`FogAgent.from_tables`, unchanged config with `oracle=none`, paired forward passes and
`torch.equal` on log-probability and value tensors.

### Shared-account overlap and owner scheduling correction (smoke)

A later read of `work/collect-gpu` at `a6b2fad` and an authenticated listing of owned
jobs revealed that G20 began at 04:27:10 UTC while brief 06's collection attempt 2 was
still running (observed terminal 04:38:31 UTC): at most 680.625 seconds of overlap.
The submission checked the preceding replay run, but missed another active account job.
Earlier statements that G20 ran alone do not describe this initial interval. Record the
scheduling deviation; do not erase the measurements or claim the one-job rule was met.
Global quota also contains brief 06's allocations and cannot all be attributed to G20.
`shared-account-overlap.json` records the source and times.

The owner explicitly chose to finish G20 before measuring brief 06's end-to-end training
speed. G100 stays pending while the brief 06 measurement is prepared on its separate branch.
Before every later launch, list all owned RUNNING/QUEUED jobs and require none active.
G20 remains the only currently active job. Its completed epoch 4 has logged holdout value
MSE 0.861 (rounded; full selected reports remain pending).

### G20 epoch 5 and verified account-wide submission guard (smoke)

G20 has completed five of its 18 epochs; logged holdout value MSE at epoch 5 is 0.871
(rounded), with the earlier epoch-4 value 0.861. Full-precision selected checkpoint reports
remain pending. `G20-progress.json` preserves all five completed-epoch log events.

The new launch guard paginates every owned job and reads its status. It checked 20 jobs,
found only G20 active, and blocked submission (`account-wide-guard-live.json`). A larger
initial list hit HTTP 429; smaller pages, throttled reads and bounded retry resolved it.
No job was submitted by either guard check. The owner's requested order remains G20,
then brief 06 end-to-end timing, then the remaining GPU scale work.

### Projected handoff to brief 06 (planning only)

At 07:19 UTC, the five completed G20 epochs took 1,909.859–1,928.435 seconds each,
median 1,923.599. Thirteen remain. Their median projects the training endpoint to
14:08:52 UTC, then allow 10–15 minutes for both holdout reports and checkpoint export.
The unmeasured brief-06 PPO session is planned at 30–60 minutes afterward: earliest
realistic result about 14:50 UTC, planning window 15:00–15:30. These are projections,
not completion measurements or promises; a GPU-path failure would change them.
`G20-to-brief06-eta.json` records the inputs and the owner's preservation choice.

### Independent brief-06 runtime supersedes the planned handoff

Concurrent updates on `work/collect-gpu` revealed an owner-selected separate single-T4
runtime, independent of G20's Kaggle allocation. Its first CUDA probe failed from host
RAM exhaustion while transferring unused probe buffers; no throughput is claimed for that
failure. The other worker added a tested counters-only probe result, retaining real buffer
transfer in PPO, and reserved a bounded replacement through 10:21 UTC. The combined
brief-06 allocation upper bound is 2.959444 GPU-hours. Its raw results and current source
are on `work/collect-gpu`; `brief06-independent-runtime.json` records the scheduling facts.

The prepared duplicate private job was never submitted and is retired. G20 continues
unchanged. The prior 14:50 brief-06 estimate applied to waiting for G20's Kaggle slot;
this independent run may produce its full-loop result earlier, but PPO timing is still
pending. G100 still waits for completed G20 curve/value reports and its CPU play smoke.
G20's quota delta contains the earlier shared-account collection work, so its own session
wall upper bound, not that contaminated delta, must drive G100's overhead projection.

### Independent end-to-end speed result, G20 preserved (smoke)

`work/collect-gpu` at `bc5bd85` published the short timing result from its independent
single T4, released by 09:49:54 UTC. Two completed PPO updates at training commit
`ac75355` retain 26,268 learner decisions / 1,024 games in 700.389269 seconds, including
startup and checkpoint writes: **37.504858 decisions/s / 1.462044 games/s**. The whole
995.824165-second attempt includes an unfinished third update and yields **26.378151
retained decisions/s / 1.028294 games/s**. Native update time is 71.897% of completed-prefix
wall time. The worker reports a separate owner instruction to shorten the learning run;
100k decisions, the 200-game evaluation and the full self-play matrix remain unverified.
No learning/strength gate is claimed. Exact published training command, native timers and
source pin are copied in `brief06-short-run-summary.json`; independent division/count
checks are in `brief06-endtoend-recomputed.json`. Cumulative brief-06 allocation upper
bound is 2.441111 GPU-hours, separate from the brief-05 Kaggle budget. No duplicate
allocation was submitted. G20 remains running unchanged, with epoch 10 of 18 saved.
G100 can follow G20 curve/value reports and CPU play without a further brief-06 GPU job.

### Completed G20, separate policy/value selections, and conservative final-slot plan (smoke)

Private `fog-p2-g20-20261114` completed and exported by 14:17:26 UTC, at training
source `7a234972abf8d4c6dde8629353588aa2558f1fd0`. Actual execution is the retained
`fog-p2-g20-20261114.py` worker calling unchanged `bc.train`; submission command was
`KaggleApi.kernels_push(acc="NvidiaTeslaT4", timeout="39600")`. The complete resolved
training command/overrides are in `G20-training-runtime.json`: CUDA, original 20k manifest,
fixed 2k holdout, batch 1024, 18 epochs, resume, policy-CE selection and the independently
saved value checkpoint, streaming on, early stopping off. CSV contains exactly epochs 0–18;
training status is complete. No loss, schedule, model or filter changed.

| Run (smoke) | Train games / rows | Epochs / batch | Policy / value epochs | Train rows/s | Holdout top-1 / policy NLL | Policy-checkpoint value Brier / ECE | Own-best value MSE / Brier / ECE | Commit |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| G20, complete | 20,000 / 1,023,525 | 18 / 1024 | 12 / 4 | 583.241649 | 0.544368683 / 1.278605058 | 0.254355678 / 0.153632606 | 0.861281370 / 0.215320343 / 0.016158573 | `7a23497` |

Per-epoch train rates are 579.961717–585.537242 rows/s; validation is 161.165194–166.059520 s,
with peak learner allocation 2,621,320,192 bytes (2.441295 GiB). Training including validation
is 34,692.389136 s; hosted setup is 92.034458 s. The selected policy has 103,981 legal-policy
holdout rows; value reports cover all 104,037. Own-best early/middle/late Brier is
0.244753784 / 0.222778563 / 0.179070357: late is 0.065683427 below early. That value
checkpoint is eight epochs before the policy checkpoint. Its train MSE is 0.832760363,
versus 0.543420724 at the later policy selection; the latter's holdout MSE has worsened to
1.017422711. The policy NLL improves by 0.300412805 from historical B. These are diagnostic
smoke results. They do not establish the brief's strict **100k** reading or any strength gate.
Both full holdout reports, phase metrics, extreme shares/win rates and all ten reliability
bins are in `G20-holdout-*.json`, `curve-value-tables.json` and `reliability.csv`.

G20's own session upper bound is 35,426.124028 s = 9.840590 h, from conservative 04:27:00
start to observed terminal. Its global quota delta includes the earlier other brief and
is not attributed wholly to G20. Calibration plus the own-session bound is at most
10.719605 replay quota-hours; the account's observed counter is 11.118819. `gpu-hours.json`
records both scopes, using Kaggle quota units, one CUDA device used, two T4s visible.

Both G20 selected weights have been added to the **same private checkpoint dataset**,
version 2, retaining version 1. The literal portable owner command was run in a fresh ignored
folder: all four G3/G20 file sizes and SHA-256 values match the public inventory
(`checkpoint-version-G20.json`, `checkpoint-portable-download-G20.json`). No weights or
account identity are committed. CPU label/oracle perturbation on 256 holdout rows leaves
policy/value tensors identical in evaluation and matched-RNG training modes
(`G20-label-perturbation.json`); the earlier sampled prefix/sign checks remain bounded smoke.

`G100-plan.json` uses the slowest G20 train rate, longest validation, measured own-session
setup/export overhead scaled to the full archive, and a two-hour forecast reserve. All
101,928 prepared games for six epochs project to 57,054.752635 s (15.848542 h), which does
not fit. The largest seeded nested prefix that fits is **41,071 games / 2,096,584 rows**,
six epochs, projecting 24,772.228739 s (6.881175 h). It contains every original G20 game
and no holdout game. The new manifest was made on CPU; rating floor and dates are unchanged.
The projected conservative account total is 17.999994 quota-hours. Planned hard platform
cap 28,200 s, internal deadline 27,000 s, keeps even the cap total below 20; the forecast's
two-hour reserve supplies the overrun allowance. No G100 GPU job has been submitted yet.
Its exact 20k-to-100k P2.4 branch and strict 100k value test will remain unverified because
this is the brief-authorized fallback. G20 CPU play is running before G100 training.

### G20 CPU play complete and input-layout correction before G100 (smoke)

Command, from the unchanged application at `325a46f` (the dirty flag covers new docs/results):
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -m
fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config
configs/replay_blueprint_cpu.yaml --checkpoint blueprint=runs/p2_gpu_checkpoints/G20-bc_best.pt
--checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml --opponent simpleheuristics
--opponent maxdamage --games 400 --temperature 0 --action-stats --resume
--out runs/p2_gpu_play/G20-play.json`. Model source is `7a23497`, policy epoch 12.
Pinned Showdown `e440c4a`; the original matched teams/sides/seeds recipe is unchanged.
Raw command, 349.939293-second total wall and exit 0 are in `G20-play-command.json`.

| Candidate (smoke) | Opponent | Games | W-L-D | Win rate [95% Wilson] | Failures / truncations | Seconds/decision / fallbacks | Commit |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| G20 policy epoch 12 | simpleheuristics | 400 | 208-192-0 | 0.5200 [0.471083, 0.568537] | 0 / 0 | not emitted / not emitted | `325a46f` |
| G20 policy epoch 12 | maxdamage | 400 | 347-53-0 | 0.8675 [0.830754, 0.897255] | 0 / 0 | not emitted / not emitted | `325a46f` |

The result file is `G20-play.json`, separate from historical B and the later model's output.
These 400-game rows are smoke, not T1/G2 evidence. B's reference rates were 0.360
[0.314, 0.408] and 0.7725 [0.729, 0.811]. G20's action statistics are retained; per-decision
wall and fallback counters were not emitted, so no zero is invented.

An intermediate CPU repack included the conversion cache. Its 4,075,781,022-byte
archive was never uploaded or attached to a job; it is explicitly discarded in
`prepare-112000-G100-bundle-with-cache.json`. Rebuild only the original top-level layout
(`raw`, `decisions`, `split`, `run.json`, `output`), omitting that CPU-only cache.
No GPU time was used. The initial 41,071-game forecast remains a superseded preliminary
plan in `G100-plan-before-archive-check.json`.

The final planner allows 2 MiB for the new manifest in the original full-corpus archive,
rounds setup/report/export allowance up to a whole minute (1,980 s), and verifies that the
actual upload fits that byte bound. The largest fitting nested prefix is now **40,955 games /
2,090,705 rows**, six epochs, projected **24,771.827035 s = 6.881063 h**. The exact 100k
rules stay unverified; no filters or training settings changed. The two-hour forecast reserve
and 28,200/27,000-second platform/internal caps stay unchanged. G20 curve/value reports,
private persistence and CPU play are all complete before the final GPU slot is submitted.

The corrected CPU archive is **2,386,510,456 bytes**, SHA-256
`5457cc642d2ecc63152a36ec4d2d78cf150c8efeaa1c37209807be1429fbfe9f`, below the
2 MiB added-manifest planning allowance. Its CPU packaging wall is 81.327875 s; it includes
the original layout and excludes the conversion cache. The same private large-input
dataset now has ready **version 2**, with version 1 retained; upload/processing/readiness
took 704.252935 s and used no GPU. Evidence: `prepare-112000-G100-{bundle,version}.json`.
The account quota remained 40,027.748 seconds before final submission.

### Final bounded fallback submitted (smoke forecast; no G100 result yet)

Private `fog-p2-g100-20261114` version 1, kernel 145012268, was accepted from source
`46e9c0da2c1b6caa01ab3da0b5bba2e103029662` at 14:42:59 UTC: **40,955 games /
2,090,705 rows**, **six epochs**, batch **1,024**, same model/loss/optimizer/schedule recipe,
CUDA, both independent checkpoint selections, streaming, no early stop. Command:
`KaggleApi.kernels_push(acc="NvidiaTeslaT4", timeout="28200")`. Its internal export deadline
is 27,000 s. Expected **6.881063 quota-hours** from the completed G20 measurements; the
forecast conservative account total is 17.999882 h, reserving two hours. Hard-cap account
total is at most 18.952152 h. Inputs are ready private large-corpus version 2, including
the CPU nested manifest; no GPU import, conversion or splitting is run.

Immediately before submission, the paginated owned list plus each job's status checked
**20 jobs**, found no RUNNING/QUEUED job, and allowed launch; fresh quota was unchanged at
40,027.748 s. `fog-p2-g100-20261114-account-guard.json` records that check. GetKernel confirms
private `NvidiaTeslaT4`, initially QUEUED. Exact submitted source, receipt, status and plan
are retained. The new run remains unmeasured until its CSV and reports return. G20's
curve/value/CPU play and four-file private-download round trip were completed first.

### Queued final slot: distinguish elapsed wait from GPU quota

The final slot remains QUEUED with no failure message. The official quota read after the
prolonged wait still shows **11.118819 h used, 0 reserved, 45 h allowed**; the status
observation is retained in `G100-queue-observation.json`. No training rate or completed
epoch exists for this run yet. The earlier 14:42:59 timestamp is the submission request
start; acceptance returned 2.455918 s later. Queued wall time has consumed no observed
GPU quota. Its submission-to-terminal wall will be a loose upper bound, not an exact
GPU charge. Keep both that bound and the final account-counter delta; the latter also
conservatively includes any unrelated account activity. No duplicate job was submitted.

Historical B play counts and full Wilson intervals are copied into
`05-artifacts/B-play-summary.json`, with its original brief-03 source path and commit
stamp retained. This is the existing 144/400 and 309/400 CPU smoke, not a new evaluation.
It makes the baseline numbers cited by the brief-05 report available in that artifact directory.

### Final fallback allocated after the uncharged queue (smoke startup)

The final slot is now RUNNING, observed around 18:35 UTC after the long queue. Its
startup log confirms **Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0,
Node v22.23.4**, matching G20's measured stack. Setup reached the training interpreter
at about 91.5 s of script log time. `G100-startup.json` preserves the startup events and quota
observation. No completed epoch or throughput is claimed yet. The conservative 6.881063-hour
forecast after allocation puts export around **01:30 UTC on 15 November**; refine it
from actual completed epochs. Queue time remains separate from charged GPU time, using
the final account-counter delta as a conservative bound rather than counting the wait.

### First final-slot epoch saved (smoke progress, no final selection yet)

G100 fallback has saved epoch 1/6. The log confirms **2,090,705 training rows /
104,037 holdout rows**, two torch threads, with input restore **104.605 s**. First-epoch
wall from the baseline log through its completed save is **3,403.646891 s**, including
validation and checkpoint overhead. Raw events and the one-epoch projection are retained
in `G100-first-epoch-progress.json`; the broader communicated completion window is
**00:40–01:00 UTC on 15 November**. This is a projection, not a completed run or measured
steady-state throughput. The CSV/selected-checkpoint reports remain pending; no NLL is
inferred from the rounded joint-loss log. The dataset/epochs/configuration are unchanged.

### Progress refresh and local monitor recovery (smoke, 22:13 UTC)

A fresh direct status query confirms the same private final-slot job is RUNNING with
**three of six epochs saved** and no failure message. The local watcher and waiting
collector had stopped; they were restarted, and the remote training continued throughout.
No GPU job was canceled or duplicated. The latest meter is about **14.76 account quota-hours**
against the 20-hour conservative budget. Epoch walls are **3,403.646891 / 3,397.826248 /
3,389.233859 s**, including validation and save, so the **00:40–01:00 UTC on 15 November**
export projection remains conservative. `G100-progress.json` retains raw epoch events and
the timing assumptions. Full CSV, selection reports, final checkpoint persistence and CPU
play still wait for the six-epoch run to finish; no final G100 NLL or strength is claimed.

### Owner ends the foreground wait; final slot continues in the background

The owner asked to report progress without waiting for G100. G20 and the short brief-06
throughput result are complete. The existing final GPU job continues unchanged at three
of six saved epochs; no cancel or duplicate submission was made. The local read-only
watcher and completion collector now have independent process sessions and file-backed
logs, verified alive after detachment. On completion the collector downloads raw metrics
and both reports, versions the same private checkpoint dataset and tests all six hashes,
then runs the specified CPU diagnostics and 400-game/opponent smoke. These final G100
results remain pending; this entry makes no completed-run or gate claim.

### 2026-11-15: final fallback and all follow-up evidence complete (smoke)

Private `fog-p2-g100-20261114` completed and exported by **00:33:06 UTC**; the detached
collector finished checkpoint persistence, diagnostics and CPU play by **00:40:56 UTC**.
No further GPU allocation was made. Actual source is `46e9c0da2c1b6caa01ab3da0b5bba2e103029662`;
`fog-p2-g100-20261114.py` calls unchanged `bc.train`, CUDA, `train_40955.parquet`, fixed
2k holdout, batch1024, six epochs, resume, policy-CE selection, independently saved best
value, streaming, early stopping off. Submission command was
`KaggleApi.kernels_push(acc="NvidiaTeslaT4", timeout="28200")`; the full resolved config
command is retained in `G100-training-runtime.json`.

| Run (smoke) | Games / rows | Epochs / batch | Policy / value epochs | Train rows/s | Top-1 / NLL | Policy value Brier / ECE | Own-best value MSE / Brier / ECE | Commit |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| G100 fallback, complete | 40,955 / 2,090,705 | 6 /1024 | 6 / 5 | 643.919560 | 0.545474654 / 1.263436895 | 0.210989323 / 0.024287480 | 0.835634712 / 0.208908678 / 0.009065196 | `46e9c0d` |

CSV contains exactly epochs0–6, with 2,090,705 rows per training epoch and
runtime status complete. Rates are 641.717205–646.036499 rows/s;
validation 144.275986–147.351477 s; peak
2,621,156,352 bytes. Training/validation wall is
20507.284637 s; installation 90.763706 s;
the two CPU holdout reports total 417.246273 s. The per-epoch CSV, last log lines,
runtime metadata and both reports are retained without changing raw metric bytes.

Own-best value epoch5 is one before policy epoch6. Its early/middle/late Brier is
0.243595556 / 0.216147087 / 0.167666934, gap **0.075928622**; train MSE 0.800861495,
holdout MSE 0.835634712, ECE 0.009065196. Low/high shares are 0.026673203 / 0.036304392,
observed win rates 0.078198198 / 0.930103257. Both numerical value thresholds are met
at **40,955**, not at the required 100k point. Strict 100k learning remains unverified.
The observed 20k→40,955 NLL gain is **0.015168163**; train/holdout NLL at the larger point
is 1.225838184 / 1.263436895. Neither exact 100k P2.4 branch is established; no wide model,
loss, schedule or data-filter change was made. `curve-value-tables.json` and `reliability.csv`
contain the complete available diagnostics. B's historical own-best weights remain absent.

Final G100 label/oracle perturbation on 256 holdout rows has identical policy/value tensors
in evaluation and matched-RNG training modes (`G100-label-perturbation.json`). Earlier
full-manifest winner and sampled per-side/prefix checks apply to this nested subset;
these are bounded smoke checks, not exhaustive isolation proof. The named leakage module's
readability-only limitation remains recorded. No new loss or architecture was introduced.

CPU play command, unchanged application at `20fbada8f3cf6307262b9ba314c29ea81087cf55`; model source `46e9c0d`:

```sh
/workspace/fog/.venv/bin/python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint blueprint=runs/p2_gpu_checkpoints/G100-bc_best.pt --checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --action-stats --resume --out /workspace/fog/runs/p2_gpu_play/G100-play.json
```

| Candidate (smoke) | Opponent | Games | W-L-D | Win rate [95% Wilson] | Failures / truncations | Seconds/decision / fallbacks | Commit |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| G100 fallback policy epoch6 | simpleheuristics | 400 | 209-191-0 | 0.5225 [0.473569, 0.571003] | 0 / 0 | not emitted / not emitted | `20fbada` |
| G100 fallback policy epoch6 | maxdamage | 400 | 343-57-0 | 0.8575 [0.819837, 0.888362] | 0 / 0 | not emitted / not emitted | `20fbada` |

Total CPU play command wall: **324.211583 s**. Separate output `G100-play.json`;
pinned Showdown `e440c4a`, same pool_b matched recipe, greedy. G20 was 0.520/0.8675 and B
0.360/0.7725; the two larger models' intervals overlap. No play-strength gate is claimed.

Private checkpoint dataset **version3** retains all **six** selected files and earlier
versions. The literal portable owner command was executed in a fresh directory; all sizes
and SHA-256 values match `checkpoints.json` (`checkpoint-version-G100.json`,
`checkpoint-portable-download-G100.json`). No weights, account identity or token are committed.

GPU accounting uses Kaggle quota-hours. Calibration used **0.879015278 h**; G20's own-session
bound is **9.840590008 h**. The final account delta from its immediate pre-submission meter
is **5.959728333 h**, a conservative G100 charge bound excluding its uncharged queue and
including any unrelated account activity. The submission-wall bound is 9.835289352 h and
must not be treated as a GPU charge. Replay-job bounds sum to **16.679333619 h**; the
conservative whole-account meter is **17.078547222 h**, below 20 with **2.921452778 h** left.
`gpu-hours.json` preserves each scope, including failed calibration attempts and the earlier
shared-account overlap. All P2 boxes stay unchecked; the exact100k point and full G3/batch
calibration remain the next missing evidence, with every original threshold unchanged.

## 2026-11-15: review of brief 10

The branch `work/replay-team-pool` (head `3f929cc`) was reviewed and merged into
`work/roadmap`. Checked in review, on the reviewer's machine:

- **Rebuilt from nothing.** A fresh import of the pinned revision (22,000 games, the same
  rejection counts as 2026-11-13) and `python -m fog.scripts.build_replay_team_pool` gave all
  803 team files and the three split manifests byte for byte as committed.
- All 803 committed teams pass pinned Showdown's Gen 9 OU validator. No composition is shared
  between splits. All 62 test teams come from the 2,000 newest games and their compositions
  are first seen there. The split manifests' SHA-256 values match the top manifest.
- The script and the eleven-line evaluation change, read line by line; no player name or
  nickname is exported. `tests/test_replay_team_pool.py` passes (15); `ruff` is clean.
- The three round-robin rows against `docs/briefs/10-artifacts/anchors.json`
  (102, 34 and 175 wins of 1,000; no failures or truncations). Not re-run.

| Fact | Value |
| --- | --- |
| Teams (one composition each) | 600 train, 141 validation, 62 test |
| Where sides were lost | 44,000 sides; 9,850 rated 1700 or more; 1,839 with 14 or more moves revealed; 1,439 pass the validator; 999 compositions; 803 after the caps |
| How much of a team is real | 27.3 of 90 fields revealed on average; 15.7 of 24 moves |
| MaxDamage against Random | 0.898 |
| SimpleHeuristics against Random | 0.966 |
| SimpleHeuristics against MaxDamage | 0.825 |

Three cautions for whoever uses the pool.

1. **Spreads are sampled without regard to the moves.** `validation/team_000.txt` has a
   Talonflame with a Bold nature (lowers Attack) carrying Flare Blitz and Swords Dance. Both
   sides of a matched block draw from the same pool, so comparisons between agents are fair,
   but absolute win rates may differ from those on teams people built.
2. **For a model trained on replays, only the test split is unseen.** Validation compositions
   are 15% of those that occur in older games, so they are in the replay training data.
   Brief 13 now drops training rows whose acting side has a validation composition. Search
   without learning (brief 11) is not affected.
3. The test split has 62 teams, not the 150 the cap allows.

P0.7 is ticked. G0 as written also asks for anchor `A3`, which is in no clone; that clause is
recorded as not measurable until the owner finds the checkpoint (queue, "Owner"). Brief 11
can start.

## 2026-11-15: review of brief 05

The branch `work/p2-gpu-scale` (head `538ef7c`) was reviewed and merged into `work/roadmap`.
**Every model in it was trained and scored on the old rows, which gave away the label through
move order** (review of brief 07, stage 1). Its policy NLL and top-1 are therefore inflated
and are not comparable with numbers on corrected rows. Its timing, value and play numbers
stand as measured. Everything is smoke. Checked in review:

- The two tables, the rates, epochs and play counts in `docs/briefs/05-report.md` against
  `docs/briefs/05-artifacts/{G20,G100}-summary.json`, `{G20,G100}-play.json` and
  `gpu-hours.json`. All agree (for example 208 and 209 wins of 400 against SimpleHeuristics).
- The code change: `bc.keep_best_value` (default off) saves the checkpoint with the lowest
  holdout value loss, and `run_replay_blueprint --prepare-only`. Nothing in the model or the
  losses changed. The three covering test files pass (38); `ruff` is clean.
- No account name or token was found in the artifacts; commits are authored as the owner.
- Not re-run: any training or play.

| Question | Answer | Evidence |
| --- | --- | --- |
| How fast is replay training on a T4? | 583 and 644 rows/s | 20,000 games (1.02 million rows): 29 minutes an epoch plus 2.7 of validation; 18 epochs cost 9.8 GPU-hours. 40,955 games, 6 epochs: 6.0 GPU-hours. All 101,928 prepared games for 6 epochs projected to 15.9 hours and did not fit the budget |
| Does the value head work with more games? | Yes | Own-best holdout Brier 0.215 (20,000 games, epoch 4) and 0.209 (40,955, epoch 5), ECE 0.016 and 0.009, against 0.345 at 3,000 games. Predictions below 0.1 or above 0.9 were won 7.8% and 93.0% of the time |
| Does it need its own checkpoint? | At 18 epochs, yes | 20,000 games: value best at epoch 4, policy at epoch 12, where value Brier is back to 0.254 |
| Does doubling the data improve the policy? | A little, offline | Holdout NLL 1.2786 to 1.2634 (inflated rows; 18 against 6 epochs) |
| Does it improve play? | No | Against SimpleHeuristics 0.520 and 0.5225; against MaxDamage 0.8675 and 0.8575; 400 games each on `pool_b`, intervals overlap |

Reading. Two things are settled for the old rows: the value head's failure at 3,000 games was
too few games and too many epochs, not the architecture; and twice the data bought nothing in
play. The P2.4 decision rule cannot be applied (the larger run is 40,955 games, not 100,000,
with a third of the epochs, and the gain of 0.015 falls between its two thresholds). Play at
0.52 against SimpleHeuristics is far from G2's 0.65. More of the same rows is not what is
missing; what a row lacks or gets wrong compared with play is the more likely limit
(briefs 09 and 12), which is a hypothesis until brief 13 measures it.

The hosted account's meter stood at 17.1 GPU-hours when the work ended.

## 2026-11-14: PPO update speed on a GPU after the belief-loop fix (smoke)

Brief 08, `work/update-speed`. Every result here is smoke. The primary measurement ran at
`e23682f`; production files under `fog/`, configs and tests are identical to `work/roadmap`
`f36bc11`. Machine: four logical Intel Xeon 2.00 GHz CPUs (CPU quota 4), two 15,360 MiB T4s
allocated, model work on `cuda:0`, PyTorch 2.11.0+cu130. Brief 06 used two CPUs, one T4 and
PyTorch 2.14.1+cu130. Thus these are measured before/after observations, not an isolated causal
ablation of the vectorisation fix. No learning or strength gate is claimed.

Primary entry: `FOG_COMMIT=e23682fd9dbf8f9f13c20cd2fbdfb4dcfb8ba47f python
 docs/briefs/08-artifacts/gpu_job.py` on the GPU platform, with the converted replay input
attached. It reuses brief 06's installer, timing shim and effective training command:
`python -m fog.train --config configs/collect_gpu_check.yaml --trainer ppo --set
log.dir=runs/collect_gpu league.snapshot_every_decisions=8000 env.rollout_workers=4`.
`measure.py ppo` installs the timing hooks before executing that command and returns the
trainer's normal halt signal after two completed, saved updates. No model, loss or trainer
source changed. Exact effective config/commands are in `08-artifacts/ppo-config.json`,
`gpu_job.py`, `measure.py` and the execution JSON records.

| PPO smoke metric | Brief 06 before | After fix | Evidence under `docs/briefs/08-artifacts/` |
| --- | ---: | ---: | --- |
| Mean optimizer-step seconds | 6.084865 | 1.314772 | `baseline-ppo.json`, `ppo-summary.json` |
| Processed optimizer rows/s | 159.886676 | 770.241811 | same |
| Collection decisions/s | 144.979601 | 293.829273 | same |
| End-to-end decisions/s, through saved checkpoint | 37.504858 | 122.802031 | same |
| Collection / update / other wall share | 25.869 / 71.897 / 2.234% | 41.794 / 52.320 / 5.887% | same |
| Sampled peak device memory, MiB | 11,561 | 11,333 | same; old whole attempt versus new prefix through PPO |

The completed primary prefix contains two PPO updates, 26,330 learner decisions from 1,024
training games, 78 optimizer steps and 78,990 processed rows in 214.410135 s. Means include
partial minibatches; the separately profiled minibatch has exactly 1,024 rows. Its synchronized
parts total 1.350055 s: trunk 0.139314, opponent belief 0.217050, self-belief 0.204925,
policy/value 0.003880, other forward 0.016592, loss assembly 0.023661, backward 0.734500,
gradient clipping/metrics 0.004789, optimizer 0.005344. **Backward is the largest component,
54.4052% of this sampled step.** The two perspectives' backbone and shared latent are counted
in trunk. Evidence: `ppo-parts.json`; each part is synchronized before and after. Primary
optimizer timings include part hooks on one minibatch and the failed stack-attempt overhead.

Replay entry: the same job invokes `python docs/briefs/08-artifacts/measure.py replay` with
`configs/replay_blueprint_cpu.yaml`, `device=cuda`, batch 1,024 and random initial weights.
The current importer/converter prepared 1,000 qualifying games (49,478 accepted rows; no
quarantines), using the pinned import and `--imputations 1 --workers 4 --resume` command in
`preparation.json`. This is an allowed small replay corpus, not the canonical brief 09 split.
The measurement reproduces `bc.py`'s full-scope CUDA loss, autocast, GradScaler, clipping,
AdamW and cosine schedule. That CUDA branch uses autocast even though the recipe says
`mixed_precision: false`. Partial epoch tails are skipped so all measured batches have 1,024
rows; there is no warmup and no skipped nonfinite optimizer step.

| Replay smoke metric | Brief 05 before | After fix | Evidence |
| --- | ---: | ---: | --- |
| Processed rows/s | 583.241649 | 1850.260853 | `baseline-replay.json`, `replay-summary.json` |
| Measured optimizer steps / row-visits | longer epoch run | 200 / 204,800 | `replay-summary.json`, `replay-metrics.csv` |
| Seconds per step / total measured seconds | not separately measured there | 0.553435 / 110.687096 | same |

Replay's measured wall includes batch building and checkpoints every 20 steps; model/dataset
loading is excluded. The synchronized sample is 0.417411 s, so it is not the aggregate step
average: trunk 0.029940, opponent belief 0.164006, policy/value 0.001942, other forward
0.004942, loss 0.012858, backward 0.196528, clipping/metrics 0.003561, optimizer 0.003636.
Self-belief is absent because replay rows have no mirror input. Evidence: `replay-parts.json`.
The 3.1724× replay comparison also changes corpus and runtime; it is not a learning result.

Stack recovery and GPU budget: the first startup aborted when a requested T4 runtime actually
allocated two devices; no training ran. The primary job completed every timing stage, but all
three `py-spy dump` attempts were denied permission. Both failures remain in the artifacts.
A CPU-only platform check verified a per-process grant to the exact profiler child, revoked
after each dump, without changing host policy. A final bounded PPO-only GPU run at `ee058d8`
then captured three successful main-process stacks during update two:
`08-artifacts/profiling-followup/update-stack-{1,2,3}.{txt,json}`. Their main threads were in
opponent belief heads, the history transformer, and the self-belief chain, respectively;
these three snapshots do not quantify time shares. Their separate timing files do not replace
the primary results above. First/main/follow-up GPU jobs used hard limits 3,300/1,740/900 s;
finished attempts were charged by their observed terminal cost before reserving the next.
All jobs were serial. Counting both allocated T4s and platform overhead conservatively gives
**0.468986 GPU-hours total**, below the one-hour budget (`gpu-budget.json`). The final job is
complete and quota reservation is zero.

Recovery: private notebook `fog-update-speed-20261114`, version 2, retains `checkpoints.zip`
with native PPO weights/state pairs and replay optimizer/RNG state. A local copy passed ZIP
integrity checking; the public record is `primary-checkpoints.json`. Actual runtime-drop
recovery was not exercised. Only text/JSON is committed; attached-input paths are normalized,
and no private account name, token or private input name is committed.

Validation: 30 focused PPO, measurement, belief-permutation, rollout-probe and GPU-job tests
passed; repository and artifact Ruff checks pass. The CPU timing-hook check preserves native
metrics, parameters and RNG exactly; arithmetic checks reconcile rows, steps, shares and both
sampled timing partitions. Commands/results: `preparation.json`, `timing-equivalence.json`,
`profiler-child-check.json`, `validation.txt`. Full suite and strength evaluation were not run.

Reading for the next brief: the update still consumes more of the measured PPO wall than
collection (52.3% versus 41.8%); backward, rather than forward Python, is now the largest
sampled step component. This meets brief 14's numerical condition in a short smoke, subject to
review of brief 08. No phase gate or P0.5 completion is claimed, and no later brief was started.

## 2026-11-15: the policy-only training step (ROADMAP P2.3)

Asked for by the owner: find what makes training slow that a profile alone does not show.
Branch `work/policy-only-step`. Machine: 4 CPU cores, no GPU, PyTorch 2.13.0 CPU. Smoke.

**What the profile says once it is read by purpose.** In one PPO update on the CPU
(2026-11-14 profile, after the loop fix; 20.3 s of forward passes over 10 optimizer steps):

| Part of the forward | Seconds | Needed for |
| --- | ---: | --- |
| Opponent-belief and self-belief models, four passes per forward | 12.1 | belief losses only |
| History encoder, own view and opponent view | 3.7 | half for policy and value, half only for self-belief |
| Shared trunk, own view and opponent view | 3.7 | half for policy and value, half only for self-belief |
| Policy and value heads | 0.3 | policy and value |

About 80% of the self-play update's forward and about 60% of the replay trainer's (which has
no opponent view) is spent on belief models that the policy and value never read. The replay
trainer could not avoid it: `bc_loss` ran every belief pass even with all belief weights at
zero.

**What changed.** `loss.skip_unweighted_belief` (default false). When it is true and every
belief weight is zero, `bc_loss` calls the model with `need_belief=False`. With the key off
the call is unchanged. `tests/test_bc_skip_belief.py` covers both.

**Does the policy lose anything without the belief losses?** Brief 07's recipe exactly:
corrected rows, the same 1,000 training games (52,955 rows) and 2,000 holdout games
(103,944 rows), 8 epochs, seed 20261201, `configs/replay_blueprint_cpu.yaml`, plus
`loss.belief_coef=0 loss.belief_chain_coef=0 loss.skip_unweighted_belief=true`
(`python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set
bc.manifest=runs/blueprint/split/train_1000.parquet
bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet bc.epochs=8
bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce seed=20261201 ...`).

| Run | Belief losses | Best holdout NLL (epoch) | Top-1 at that epoch | NLL at epoch 8 | Value MSE at epoch 8 | Train rows/s | Whole run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brief 07 R1 (another machine, 5 threads) | 0.05 and 0.05 | 1.9239 (8) | 26.48% | 1.9239 | about 1.57 (Brier 0.392) | 205 to 236 | 3,134 s |
| Brief 07 R1, second seed | 0.05 and 0.05 | 1.9293 (7) | 26.67% | | about 1.59 | | 3,144 s |
| Policy-only, this section | none, passes skipped | 1.9248 (6) | 26.15% | 1.9274 | 1.58 | 412 to 436 | 1,481 s |

Per-epoch numbers are in `docs/notes/policy-only-step/B-metrics.csv`.

**Reading.** The policy-only run is inside the gap between brief 07's two seeds (0.0053 NLL):
on 1,000 games the belief losses do not help the policy or the value head. Training is about
twice as fast. The two rows come from different machines, so the speed ratio is approximate;
a control on this machine with the belief losses on follows in the next section. Not shown:
the same at 20,000 games or more, where an auxiliary loss could matter differently, and
anything on a GPU, where the belief passes are many small operations and the saving should be
larger than on the CPU.

**What follows.** Train the replay policy and value without belief passes (brief 13 now does).
The opponent-team model is not needed until P4.1; train it then, on its own, with the trunk
frozen. For self-play the same skip is in brief 14.

## 2026-11-15: the policy-only training step, control on one machine

The control promised in the previous section: the same recipe, data, seed and machine (4 CPU
cores, PyTorch 2.13.0 CPU), run one after the other, differing only in the belief losses.
Smoke; one seed each.

| Run | Belief losses | Best holdout NLL (epoch) | Top-1 at that epoch | NLL / top-1 at epoch 8 | Value MSE at epoch 8 | Train rows/s | Seconds per epoch | Whole run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.05 and 0.05 (config values) | 1.9235 (8) | 26.60% | 1.9235 / 26.60% | 1.61 | 193 to 202 | 398 | 3,361 s |
| B | none, passes skipped | 1.9248 (6) | 26.15% | 1.9274 / 26.47% | 1.58 | 412 to 436 | 176 | 1,481 s |

Per-epoch numbers: `docs/notes/policy-only-step/{A,B}-metrics.csv`. Run A reproduces brief
07's R1 from another machine (1.9239).

The difference in best NLL is 0.0013, against 0.0107 for twice the gap between two seeds
(brief 07), so no difference is claimed. Training is 2.1 to 2.2 times as fast in rows per
second and the whole run, with its nine holdout passes, 2.27 times. The reading of the
previous section stands, now measured on one machine. Still not shown: 20,000 games or more,
and a GPU.


## 2026-11-14: order-blind model, value targets and move features (smoke)

Brief 09, branch `work/order-blind-model`, implementation `b824430` from `70cc5ed`.
Preparation and stage 2 representation checks only; R1 is running and R2 has not started.
No value/feature/lead result or strength gate is claimed. Date is America/Los_Angeles.

| Check | Result (smoke) | Producing commit and command / evidence |
| --- | --- | --- |
| Corrected corpus | 22,000 games, zero quarantines, 1,126,792 rows; fixed holdout 2,000 games / 103,944 rows; training 1,000 games / 52,955 rows | Converter source `70cc5ed`; `python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou_r1500 --out data/metamon_gen9ou_r1500_decisions --imputations 1 --workers 4 --resume`; split with `--holdout-ids data/replay_splits/p2_holdout_2000_ids.txt --train-sizes 1000`; `09-artifacts/preparation-execution.json`, `conversion-quality.json`, `data-counts.json` |
| Permutation acceptance | Every parameter perturbed; 64 holdout, 64 live and 64 synthetic observations; all four permutations within 1e-5; key-off acting-move and own-bench checks fail. Ten acceptance tests pass | `b824430`; `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 FOG_ORDER_BLIND_TEST_REPORT=docs/briefs/09-artifacts/permutations-holdout.json .venv/bin/python -m pytest -q tests/test_order_blind_model.py`; `09-artifacts/tests-stage2-acceptance.txt`, `permutations.json` |
| Default compatibility | Strict original checkpoint load and bit-identical policy/value on 64 live rows; all three same-machine rollout digests match | Original `70cc5ed`, new `b824430`; `09-artifacts/compatibility-stage2.json`, `digests-summary.json` (exact probe commands) |

The initial conversion dispatcher was corrected before baseline training; its converter was
interrupted and resumed from 9,227 per-game checkpoints, as `09-artifacts/preparation-restart.json`
records. Report: `docs/briefs/09-report.md`. Measurements and stage 2 completion follow below.

Stage 2 completed on 2026-11-15 UTC. All results below are smoke; both perspectives of a
game stay together in the 2,000-draw percentile bootstrap. Policy uses 103,933 valid rows;
value uses all 103,944 rows from the same 2,000 games. Eight epochs, 1,000 training games.

| Run | Result (smoke; 95% game intervals) | Producing commit and command / evidence |
| --- | --- | --- |
| R1 | Policy e8: shuffled NLL 1.9253 [1.9189, 1.9316]; top-1 26.64 [26.34, 26.95]%; policy Brier 0.4038 [0.3887, 0.4176], ECE 0.3808 [0.3653, 0.3956]. Value e0: Brier 0.2533 [0.2532, 0.2534], ECE 0.0573 [0.0567, 0.0579] | `70cc5eda12345edffe11a998e2c4782f4ed7b161`; `python -m fog.training.bc --config /workspace/fog-order-baseline/configs/replay_blueprint_cpu.yaml --set bc.manifest=runs/blueprint/split/train_1000.parquet bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet bc.epochs=8 bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce log.dir=runs/order_blind_model/R1 seed=20261201`; `09-artifacts/R1-metrics.csv`, `R1-selection.json`, `R1-best-{policy,value}-slot-check.json`; exact execution in `preparation-execution.json` |
| R2 | Policy e8: shuffled NLL 1.8799 [1.8735, 1.8860]; top-1 28.24 [27.92, 28.57]%; policy Brier 0.3978 [0.3839, 0.4117], ECE 0.3744 [0.3596, 0.3893]. Value e0: Brier 0.2500 [0.2500, 0.2500], ECE 0.0015 [0.0009, 0.0021] | `c7007c8e2fdd7c63abe0299a0b6b64059c783168`; `python -m fog.training.bc --config configs/replay_order_blind_cpu.yaml --set bc.manifest=runs/blueprint/split/train_1000.parquet bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet bc.epochs=8 bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce log.dir=runs/order_blind_model/R2 seed=20261201`; `09-artifacts/R2-metrics.csv`, `R2-selection.json`, `R2-best-{policy,value}-slot-check.json`; exact execution in `stage2-execution.json` |

R1 matched the required reference within 0.0016 NLL. R2 stored/shuffled results match to
four decimals for all/move/unmoved groups. `09-artifacts/stage2-summary.json` compares
the observed differences with twice the historical R1 two-seed difference; no smaller
difference is claimed. Both value selectors remain at or above Brier 0.25.
The required perturbed-weight invariance evidence completes P2.2a; P2.3 and all gates remain open.

Stage 3 completed (smoke). The first dense-label conversion was stopped and rejected before training.
At source `699bd3f`, 84/500 checked games disagreed across terminal perspectives (maximum HP difference error 1/6); 7,122 per-game conversion checkpoints had completed. The checker used the first 500 sorted completed shards; no population interval is inferred from this deterministic integrity check. `09-artifacts/value-public-targets-rejected-check.json`, `value-conversion-restart.json` and `value-data-execution-rejected.json` retain the failed run.

The Ogerpon-only correction at `d86caf1` was too narrow: a full raw-log scan found 347 switch events involving other public forms; four representative public end states confirmed missed identities. R3 was stopped after two completed epochs, and its metrics/checkpoints are excluded from the final comparison (`value-forms-restart.json`, `R3-rejected-forms-metrics.csv`, `R3-rejected-forms-execution.json`).

The final corrected source `69b9d19b503702087d2d69a504b25019d2541fa3` reads terminal targets only from public opponent views and matches all forms through unique public base-species metadata, rejecting unmatched identities. `python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou_r1500 --out data/metamon_gen9ou_r1500_terminal_decisions --imputations 1 --workers 4 --resume` rebuilt all 22,000 games with no quarantines. Full cross-view checks have zero error; all old arrays are identical for the 3,000 fixed-split games. Counts/commands: `value-data-counts.json`, `value-data-execution.json`; 95 focused tests and Ruff pass.

| Run | Result (smoke; 95% game intervals) | Producing commit and command / evidence |
| --- | --- | --- |
| Rejected dense-label conversion | 84/500 checked games inconsistent; max HP error 1/6; stopped after 7,122 completed games, no training. Interval n/a (deterministic coherence check) | `699bd3f`; `python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou_r1500 --out data/metamon_gen9ou_r1500_value_decisions --imputations 1 --workers 4 --resume`; `09-artifacts/value-data-execution-rejected.json`, `value-public-targets-rejected-check.json` |
| Rejected partial R3 | Stopped after 2/8 epochs; 1,000 train / 2,000 holdout games. Auxiliary targets missed non-Ogerpon forms. No final comparison or interval claimed | `d86caf1`; original fixed command in `09-artifacts/R3-rejected-forms-execution.json` (exit -15), metrics in `R3-rejected-forms-metrics.csv`; original `runs/order_blind_model/value_split` dataset retained |
| R3 | Policy e8: shuffled NLL 1.8806 [1.8744, 1.8869]; top-1 28.51 [28.16, 28.84]%; Brier 0.4065 [0.3928, 0.4201], ECE 0.3841 [0.3702, 0.3981]. Value e0: Brier 0.2500 [0.2500, 0.2500], ECE 0.0015 [0.0009, 0.0021] | `69b9d19b503702087d2d69a504b25019d2541fa3`; `python -m fog.training.bc --config configs/replay_value_cpu.yaml --set bc.manifest=runs/order_blind_model/terminal_split/train_1000.parquet bc.holdout_manifest=runs/order_blind_model/terminal_split/holdout_manifest.parquet bc.epochs=8 bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce log.dir=/workspace/fog/runs/order_blind_model/R3 seed=20261201`; 1,000 train / 2,000 holdout games; `09-artifacts/R3-{metrics.csv,selection.json}`, `R3-best-{policy,value}-slot-check.json`, `R3-execution.json` |

Both checkpoint diagnostics include p<0.1 / p>0.9 shares and conditional observed win frequencies, with game-resampled intervals. They are decision-weighted, not unique-game frequencies. Native value selection includes epoch 0 and is resumable; synthetic interruption tests preserve the independent policy/value selections and RNG exactly. No gate is claimed.

Stage 4 completed (smoke). Mechanical features default off; public table facts and exactly eleven candidate columns are qualified by 89 affected tests, excluded-column NaN/gradient checks, and all-parameter-noise permutation checks (64 synthetic/holdout/live).

| Run | Result (smoke; 95% whole-game intervals) | Producing commit and command / evidence |
| --- | --- | --- |
| R4 | Policy e7: NLL 1.8041 [1.7976, 1.8108]; top-1 33.25 [32.91, 33.58]%; policy Brier 0.4017 [0.3877, 0.4162], ECE 0.3731 [0.3586, 0.3880]; value e0: Brier 0.2567 [0.2566, 0.2568], ECE 0.0819 [0.0813, 0.0825] | `4066edc1d756e41a28b136156daba5f57060fdae`; `python -m fog.training.bc --config configs/replay_mechanical_cpu.yaml --set bc.manifest=runs/order_blind_model/terminal_split/train_1000.parquet bc.holdout_manifest=runs/order_blind_model/terminal_split/holdout_manifest.parquet bc.epochs=8 bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce log.dir=/workspace/fog/runs/order_blind_model/R4 seed=20261201`; 1,000 train / 2,000 holdout games; `09-artifacts/R4-{metrics.csv,selection.json,execution.json}`, `R4-best-{policy,value}-slot-check.json` |
| R4-seed2 | Policy e6: NLL 1.8139 [1.8072, 1.8208]; top-1 32.80 [32.45, 33.15]%; policy Brier 0.3878 [0.3731, 0.4020], ECE 0.3601 [0.3449, 0.3747]; value e0: Brier 0.2707 [0.2705, 0.2708], ECE 0.1438 [0.1432, 0.1444] | `4066edc1d756e41a28b136156daba5f57060fdae`; `python -m fog.training.bc --config configs/replay_mechanical_cpu.yaml --set bc.manifest=runs/order_blind_model/terminal_split/train_1000.parquet bc.holdout_manifest=runs/order_blind_model/terminal_split/holdout_manifest.parquet bc.epochs=8 bc.early_stopping_patience=0 bc.resume=true bc.selection_metric=action_ce log.dir=/workspace/fog/runs/order_blind_model/R4-seed2 seed=20261202`; 1,000 train / 2,000 holdout games; `09-artifacts/R4-seed2-{metrics.csv,selection.json,execution.json}`, `R4-seed2-best-{policy,value}-slot-check.json` |
| Rare-move comparison | Never / 1–19 / 20+ training labels; full-14-action NLL/top-1 and game intervals for R3/R4/R4-seed2; twice-seed-gap comparisons in stage4-summary | `4066edc1d756e41a28b136156daba5f57060fdae`; `/workspace/fog/.venv/bin/python -m fog.eval.replay_move_frequency_report --config R3=configs/replay_value_cpu.yaml --checkpoint R3=runs/order_blind_model/R3/checkpoints/bc_best.pt --config R4=configs/replay_mechanical_cpu.yaml --checkpoint R4=runs/order_blind_model/R4/checkpoints/bc_best.pt --config R4-seed2=configs/replay_mechanical_cpu.yaml --checkpoint R4-seed2=runs/order_blind_model/R4-seed2/checkpoints/bc_best.pt --train-manifest runs/order_blind_model/terminal_split/train_1000.parquet --holdout-manifest runs/order_blind_model/terminal_split/holdout_manifest.parquet --out docs/briefs/09-artifacts/rare-moves.json`; 1,000 train / 2,000 holdout games; `09-artifacts/rare-moves.json` |
| R1 vs simpleheuristics | 59–341, 0 true draws, 0 unresolved, 0 failures; decisive win rate 14.75 [11.61, 18.56]% (Wilson); native primary/block scores retained | `4066edc1d756e41a28b136156daba5f57060fdae`; `python -u -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint R1=runs/order_blind_model/R1/checkpoints/bc_best.pt --checkpoint R4=runs/order_blind_model/R4/checkpoints/bc_best.pt --checkpoint-config R1=configs/replay_blueprint_cpu.yaml --checkpoint-config R4=configs/replay_mechanical_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --action-stats --resume --out runs/order_blind_model/play.json`; 400 games; `09-artifacts/play.json`, `play-execution.json` |
| R1 vs maxdamage | 159–241, 0 true draws, 0 unresolved, 0 failures; decisive win rate 39.75 [35.07, 44.62]% (Wilson); native primary/block scores retained | `4066edc1d756e41a28b136156daba5f57060fdae`; `python -u -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint R1=runs/order_blind_model/R1/checkpoints/bc_best.pt --checkpoint R4=runs/order_blind_model/R4/checkpoints/bc_best.pt --checkpoint-config R1=configs/replay_blueprint_cpu.yaml --checkpoint-config R4=configs/replay_mechanical_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --action-stats --resume --out runs/order_blind_model/play.json`; 400 games; `09-artifacts/play.json`, `play-execution.json` |
| R4 vs simpleheuristics | 154–246, 0 true draws, 0 unresolved, 0 failures; decisive win rate 38.50 [33.86, 43.36]% (Wilson); native primary/block scores retained | `4066edc1d756e41a28b136156daba5f57060fdae`; `python -u -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint R1=runs/order_blind_model/R1/checkpoints/bc_best.pt --checkpoint R4=runs/order_blind_model/R4/checkpoints/bc_best.pt --checkpoint-config R1=configs/replay_blueprint_cpu.yaml --checkpoint-config R4=configs/replay_mechanical_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --action-stats --resume --out runs/order_blind_model/play.json`; 400 games; `09-artifacts/play.json`, `play-execution.json` |
| R4 vs maxdamage | 308–92, 0 true draws, 0 unresolved, 0 failures; decisive win rate 77.00 [72.63, 80.86]% (Wilson); native primary/block scores retained | `4066edc1d756e41a28b136156daba5f57060fdae`; `python -u -m fog.eval.showdown_direct_selection --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml --checkpoint R1=runs/order_blind_model/R1/checkpoints/bc_best.pt --checkpoint R4=runs/order_blind_model/R4/checkpoints/bc_best.pt --checkpoint-config R1=configs/replay_blueprint_cpu.yaml --checkpoint-config R4=configs/replay_mechanical_cpu.yaml --opponent simpleheuristics --opponent maxdamage --games 400 --temperature 0 --action-stats --resume --out runs/order_blind_model/play.json`; 400 games; `09-artifacts/play.json`, `play-execution.json` |

Play uses the same global seed-0 schedule for both policies/opponents, with teams and sides crossed, temperature 0. Play overlaps second-seed training; timings are contended. Pool B is not a held-out gate. Per-run differences below twice the R4 seed gap are not claimed. All results are smoke; no gate changed.

Rare-group detail from the preceding comparison command/source, 2,000 holdout games; intervals resample games, both perspectives together. Each cell is NLL [95%] / top-1 % [95%].

| Training labels / holdout rows | R3 | R4 | R4-seed2 |
| --- | --- | --- | --- |
| never / 130 | 1.8840 [1.7479, 2.0249] / 25.38 [12.50, 37.98] | 2.0163 [1.8794, 2.1504] / 15.38 [8.47, 23.48] | 2.0197 [1.8650, 2.1617] / 22.31 [14.50, 30.52] |
| 1_to_19 / 2171 | 2.0192 [1.9813, 2.0595] / 11.93 [10.18, 13.72] | 1.8193 [1.7775, 1.8638] / 29.25 [26.86, 31.81] | 1.7940 [1.7471, 1.8433] / 31.74 [29.12, 34.51] |
| 20_or_more / 72552 | 1.7233 [1.7173, 1.7290] / 30.98 [30.57, 31.38] | 1.6757 [1.6693, 1.6822] / 36.62 [36.20, 37.04] | 1.6735 [1.6670, 1.6799] / 36.84 [36.42, 37.25] |

The 1–19 and 20+ gains exceed twice their R4 seed differences; unseen-move NLL worsens in both seeds with only 130 rows and broad game intervals. No unseen-move gain is claimed. All standard all/move/unmoved NLL/top-1 improvements of R4 over R3 exceed twice their own R4 seed gap.

Matched-block bootstrap (10,000 draws, seed 0): R4−R1 score difference +0.2375 [0.1900, 0.2850] vs SimpleHeuristics and +0.3725 [0.3075, 0.4375] vs MaxDamage. Individual primary-score block intervals and the exact analysis command/source are in `09-artifacts/play-block-intervals.json` and `stage4-summary-driver.txt`. These 400-game cells remain smoke, with no seed-robust play-strength claim.

Stage 5 and brief 09 completed (smoke). Preview records are separate from decisions, two per game from imputation zero. Inputs stop at public preview; labels use public physical identities, masking unresolved Illusion. No R5 play.

`c87b5198b5b48c9c9e4b24e586e8ab174bd343fa`: `python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou_r1500 --out data/metamon_gen9ou_r1500_preview_decisions --imputations 1 --workers 4 --resume`; same fixed split. All 22,000 games, no quarantines, 1,126,792 ordinary rows and 44,000 preview records. Old arrays including terminal labels are bit-identical in all 3,000 fixed-split games. `09-artifacts/preview-data-{counts,execution}.json`, `preview-conversion-quality.json` record commands/checks/eligible counts.

The initial full conversion completed successfully, but its integrity driver incorrectly required HP=1 in empty slots of incomplete previews. Validation was resumed with present HP=1 / empty HP=0; no source or data changed, and training waited for all checks. `09-artifacts/preview-checker-restart.json` and `preview-data-execution-initial.json` retain the failed check and original full-conversion command/timing.

| Run | Result (smoke; 95% whole-game intervals) | Source, command and evidence |
| --- | --- | --- |
| R5 | Preview CE-selected epoch 8; top-1 34.90 [33.43, 36.44]%; NLL 1.6031 [1.5826, 1.6234] | `c87b5198b5b48c9c9e4b24e586e8ab174bd343fa`; `python -m fog.training.bc --config configs/replay_preview_cpu.yaml --set seed=20261201 log.dir=runs/order_blind_model/R5`; 1,000 train / 2,000 holdout games, 1985 / 3986 valid previews; `09-artifacts/R5-{metrics.csv,selection.json,scope.json}` |
| R5-seed2 | Preview CE-selected epoch 8; top-1 34.95 [33.51, 36.44]%; NLL 1.6025 [1.5820, 1.6231] | `c87b5198b5b48c9c9e4b24e586e8ab174bd343fa`; `python -m fog.training.bc --config configs/replay_preview_cpu.yaml --set seed=20261202 log.dir=runs/order_blind_model/R5-seed2`; 1,000 train / 2,000 holdout games, 1985 / 3986 valid previews; `09-artifacts/R5-seed2-{metrics.csv,selection.json,scope.json}` |
| Lead baselines | Chance 16.6667%; training-frequency top-1 23.93 [22.58, 25.17]%; R5−frequency 10.96 [9.46, 12.60] points | `c87b5198b5b48c9c9e4b24e586e8ab174bd343fa`; `/workspace/fog/.venv/bin/python -m fog.eval.replay_preview_report --config configs/replay_preview_cpu.yaml --checkpoint initial-preview=runs/order_blind_model/R4/checkpoints/bc_best.pt --checkpoint R5=runs/order_blind_model/R5/checkpoints/bc_best.pt --checkpoint R5-seed2=runs/order_blind_model/R5-seed2/checkpoints/bc_best.pt --train-manifest runs/order_blind_model/preview_split/train_1000.parquet --holdout-manifest runs/order_blind_model/preview_split/holdout_manifest.parquet --out docs/briefs/09-artifacts/preview-holdout.json`; 2,000 holdout games; `09-artifacts/preview-holdout.json` |

Frequency uses the most-led training species present with alphabetical ties; no holdout outcomes. The initial-preview comparator uses R4 weights with the new pooled preview path, not the old R4 preview math used in play. Twice the R5 seed gap is 0.100351 points; qualified effects are in `stage5-summary.json`. Counts, label shares and exclusions: `preview-data-counts.json`, `preview-holdout.json`.

166 affected tests and Ruff pass. Native preview resume preserves model/optimizer/scheduler/RNG/best checkpoint after interruption. Initial development tests corrected a too-strict public single-ability assertion and moved invalid-option validation before manifest reads (`tests-stage5-preview-initial.txt`, `checks-stage5.txt`).

| Check | Result | Source, command and evidence |
| --- | --- | --- |
| Final compatibility | Strict original checkpoint load; policy/value bit-identical on 64 frozen live observations. R4 preview math also bit-identical on 64 prefix observations with the new preview flag off | `c87b5198b5b48c9c9e4b24e586e8ab174bd343fa`; `python runs/order_blind_model/check_compatibility.py`; `09-artifacts/compatibility-final.json`, `preview-off-compatibility.json`; deterministic checks, intervals n/a |
| Final sequential digest | `0d026fed7728c0c4` → `0d026fed7728c0c4`; all new keys absent/off | `c87b5198b5b48c9c9e4b24e586e8ab174bd343fa`; `.venv/bin/python -m fog.eval.showdown_rollout_probe --games 6`; `09-artifacts/digest-sequential-final.txt`, `digests-final-summary.json`; deterministic regression, intervals n/a |
| Final batched digest | `ef414aee0fe96bad` → `ef414aee0fe96bad`; all new keys absent/off | `c87b5198b5b48c9c9e4b24e586e8ab174bd343fa`; `.venv/bin/python -m fog.eval.showdown_rollout_probe --batch 32 --games 64 --threads 2`; `09-artifacts/digest-batched-final.txt`, `digests-final-summary.json`; deterministic regression, intervals n/a |
| Final maxdamage digest | `40644081cca71ebe` → `40644081cca71ebe`; all new keys absent/off | `c87b5198b5b48c9c9e4b24e586e8ab174bd343fa`; `.venv/bin/python -m fog.eval.showdown_rollout_probe --batch 32 --games 96 --opponent MaxDamage --pool data/teams/pool_a`; `09-artifacts/digest-maxdamage-final.txt`, `digests-final-summary.json`; deterministic regression, intervals n/a |

All stages complete; only P2.2a acceptance is closed. P2.3 and every phase gate stay open. No search, PPO or belief source was changed; no playing-strength gate claimed.

## 2026-11-14: EngineMCTS fidelity, budget curve and gate G1

Brief 11, branch `work/g1`, based on `f36bc11`. CPU only: four-core quota on AMD EPYC 9V74,
Python 3.12.14, locked environment, pinned Showdown `e440c4a`. Validation split only;
no test-split files read. Exact commands are in `docs/briefs/11-artifacts/commands.json`.

Fidelity baseline (`f36bc11`): `python -m fog.scripts.fuzz_recon_showdown --pool
 data/teams/replay_v1/validation --games 200 --max-turns 80 --seed 0 --out
 docs/briefs/11-artifacts/fuzz-showdown-before.json` checked 14,362 positions; 1,214
with any mismatch, zero reconstruction exceptions. `python -m fog.scripts.fuzz_recon
--pool data/teams/replay_v1/validation --games 2000 --max-turns 40 --extended --seed 0`
checked 3,526 engine positions; 11 with core differences, 2,234 with extended differences.
Before files and the environment are retained in `11-artifacts/`.

Reconstruction fixes at `aa38e46`: map Ogerpon-Teal-Tera back to Ogerpon for matching a
submitted-world slot; read pending charge moves from public events, including completion by
Power Herb / immediate animation. Regressions cover both player views, HP/active/used-Tera
across re-entry, all four Ogerpon forms, charge-only engine root options and charge completion.
The same Showdown command with `--out docs/briefs/11-artifacts/fuzz-showdown-final.json`
has 1,043 mismatching positions and zero exceptions. Opponent HP (96), Tera (56), active
identity (10), and missing Phantom Force (52) / Meteor Beam (34) differences are gone.
Fields overlap. Most remaining positions are hidden benched Regenerator healing (821).
The remaining exact labels/counts are in the JSON; they are not fallback rates.

Checks: 45 focused tests, 84 integration tests, `ruff check fog tests` clean (commands and
captured results retained). A 200-game default-budget validation smoke is now running in two
100-game matched-block shards, seeds 11000/11001, K=8, 1600 ms, one search thread, no SPRT,
max turns 800. No match result yet. P1.3/P1.4 remain unchecked, G1 open; budget curve and
1,000-game/opponent gate rows have not started. See `docs/briefs/11-report.md`.

### Partial smoke checkpoint (run continues)

At the saved 80/56-game blocks: 136 games, 127 wins, 3,516 decisions, zero fallbacks,
rejected actions, failed games or truncations; 1.7555 / 1.7516 seconds per decision in the
individual shards. **Partial smoke, not a gate or completed Stage 1.** The per-game JSON
checkpoints and captured shard progress are pushed so another runtime can resume the same
`fidelity_smoke` command. Source remains `aa38e46`; no measurement code changed.

The engine round-trip was also repeated at `aa38e46` using the command in
`11-artifacts/commands.json`: 3,526 positions, 11 core mismatches and 2,229 extended mismatches
(`fuzz-engine-final.txt`). This briefly used a spare core during the two-shard smoke on the
four-core quota; budget/gate jobs will run without another simulator job.

### 2026-11-14: fidelity smoke complete

Stage 1 passes: default-budget 200-game smoke, fallbacks below 5%, no rejected actions. P1.3 complete. This is smoke, not G1.

| 2026-11-14 | mcts | simpleheuristics | showdown_direct | 200 | 0.935 [0.892, 0.962] (187-13-0) | 1.776 | 2/5341 | `aa38e46` | `python -m fog.eval.run_shards --shards 2 --games 200 --seed 11000 --out-dir docs/briefs/11-artifacts/fidelity-smoke -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":1600,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### 2026-11-14: budget 800 ms K=8

400-game smoke; same seed blocks across cells.

| 2026-11-14 | mcts | simpleheuristics | showdown_direct | 400 | 0.917 [0.886, 0.941] (367-33-0) | 0.896 | 27/10674 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-800-k8 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":800,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### 2026-11-14: budget 200 ms K=8

400-game smoke; same seed blocks across cells.

| 2026-11-14 | mcts | simpleheuristics | showdown_direct | 400 | 0.910 [0.878, 0.934] (364-36-0) | 0.261 | 28/10738 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-200-k8 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":200,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### Partial 3,200 ms / K=8 checkpoint (run continues)

Observed saved blocks: 200 games, 182 wins, 5487 decisions, 13 fallbacks, 0 rejected actions, 0 failed games and 0 truncations. **Partial smoke, not a completed curve cell or gate.** The curve allocation at observation is 9.354326 core-hours. Raw matched-block reports and the allocation checkpoint are pushed for recovery. Source remains `aa38e46`; the exact command and launch commit remain in `11-artifacts/measurement-progress.json`.

### 2026-11-15: budget 3200 ms K=8

400-game smoke; same seed blocks across cells.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct | 400 | 0.905 [0.872, 0.930] (362-38-0) | 3.636 | 21/10875 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-3200-k8 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":3200,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### 2026-11-15: budget 800 ms K=1

400-game smoke; same seed blocks across cells.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct | 400 | 0.782 [0.739, 0.820] (313-87-0) | 0.904 | 13/12388 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-800-k1 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":1,"budget_ms":800,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### 2026-11-15: budget 800 ms K=32

400-game smoke; same seed blocks across cells.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct | 400 | 0.915 [0.884, 0.939] (366-34-0) | 1.027 | 13/10785 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-800-k32 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":32,"budget_ms":800,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### 2026-11-15: budget 200 ms K=1

400-game smoke; same seed blocks across cells.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct | 400 | 0.840 [0.801, 0.873] (336-64-0) | 0.225 | 23/11550 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-200-k1 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":1,"budget_ms":200,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### 2026-11-15: budget 200 ms K=32

400-game smoke; same seed blocks across cells.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct | 400 | 0.865 [0.828, 0.895] (346-54-0) | 0.405 | 14/11268 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-200-k32 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":32,"budget_ms":200,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### Partial 3,200 ms / K=1 checkpoint (run continues)

Observed saved blocks: 176 games, 140 wins, 5618 decisions, 9 fallbacks, 0 rejected actions, 0 failed games and 0 truncations. **Partial smoke, not a completed curve cell or gate.** The curve allocation at observation is 30.282541 core-hours. Raw matched-block reports and the allocation checkpoint are pushed for recovery. Source remains `aa38e46`; the command and launch commit are in `11-artifacts/measurement-progress.json`.

### 2026-11-15: budget 3200 ms K=1

400-game smoke; same seed blocks across cells.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct | 400 | 0.775 [0.732, 0.813] (310-90-0) | 3.419 | 28/12836 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-3200-k1 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":1,"budget_ms":3200,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### 2026-11-15: budget curve stage complete

Budget charged 40.0000 allocated core-hours. Complete cells: ['800 ms K=8', '200 ms K=8', '3200 ms K=8', '800 ms K=1', '800 ms K=32', '200 ms K=1', '200 ms K=32', '3200 ms K=1']. Unfinished/not run: ['3200 ms K=32']. The 40-core-hour cap stops further cells.



### Capped final budget cell: 3,200 ms / K=32

Stopped at 40.0000 allocated core-hours as required by brief 11. This is a partial smoke, not a completed 400-game curve cell or a gate. The saved whole matched blocks contain 96 games; unfinished games at the cap are excluded. Derived totals are in `11-artifacts/budget-capped-summary.json`; the literal shard checkpoints remain available for inspection. There are eight complete cells, one partial cell, and no never-started cells. P1.4 stays unchecked because all nine full cells were not measured.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct (partial/capped) | 96 | 0.927 [0.857, 0.964] (89-7-0) | 3.543 | 11/2283 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 400 --seed 12000 --out-dir docs/briefs/11-artifacts/budget-3200-k32 -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":32,"budget_ms":3200,"threads":1}' --opponent simpleheuristics --max-turns 800 --progress-every 4` |

### Partial default-budget gate checkpoint (run continues)

Saved whole matched blocks only; these rows are partial and do not establish G1. The literal shard reports and allocation state are committed for recovery; `gate-partial-summary.json` holds derived totals from the staged checkpoint. Native evaluations continue to the fixed 1,008 games per opponent; no sequential stopping.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct (partial) | 604 | 0.935 [0.913, 0.952] (565-39-0) | 1.813 | 31/17443 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 1008 --seed 13000 --out-dir docs/briefs/11-artifacts/gate -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":1600,"threads":1}' --opponent simpleheuristics --opponent maxdamage --max-turns 800 --progress-every 4` |

### Default-budget SimpleHeuristics gate opponent complete

All four shards completed 252 games against SimpleHeuristics, for 1,008 matched games. This opponent meets its G1 rate and fidelity requirements. MaxDamage is still running, so G1 is not yet established. `gate-simpleheuristics-summary.json` contains derived totals from the committed shard rows.

| 2026-11-15 | mcts | simpleheuristics | showdown_direct | 1008 | 0.929 [0.911, 0.943] (936-72-0) | 1.801 | 53/28020 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 1008 --seed 13000 --out-dir docs/briefs/11-artifacts/gate -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":1600,"threads":1}' --opponent simpleheuristics --opponent maxdamage --max-turns 800 --progress-every 4` |

### Partial MaxDamage gate checkpoint (run continues)

SimpleHeuristics is complete; MaxDamage is only halfway. This partial row does not establish G1. The staged whole matched blocks and state are pushed for recovery, with derived totals in `gate-maxdamage-half-summary.json`. The fixed 1,008-game MaxDamage run continues without sequential stopping.

| 2026-11-15 | mcts | maxdamage | showdown_direct (partial) | 504 | 0.972 [0.954, 0.983] (490-14-0) | 1.830 | 7/12492 | `aa38e46` | `/workspace/fog/.venv/bin/python -m fog.eval.run_shards --shards 4 --games 1008 --seed 13000 --out-dir docs/briefs/11-artifacts/gate -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":1600,"threads":1}' --opponent simpleheuristics --opponent maxdamage --max-turns 800 --progress-every 4` |

## 2026-11-16: review of brief 08

The branch `work/update-speed` (head `d0efe84`) was reviewed and merged into `work/roadmap`
(merge `2be53f3`). Everything in it is smoke. Checked in review:

- Every number in `docs/briefs/08-report.md` against `docs/briefs/08-artifacts/ppo-summary.json`,
  `replay-summary.json`, `ppo-parts.json` and `replay-parts.json`. All agree.
- The claim that production code equals `f36bc11`: `git diff f36bc11 e23682f -- fog tests configs`
  is empty, and outside `docs/` the branch changes only `ROADMAP.md`.
- No account name or token in the artifacts; all eight commits are authored as the owner.
- Not re-run: any GPU measurement.

| Question | Answer | Evidence |
| --- | --- | --- |
| How fast is the PPO update on a T4 after the belief-loop fix? | 1.315 s per 1,024-row optimizer step (6.085 before), 122.80 decisions/s end to end (37.50 before) | `ppo-summary.json`: two updates, 26,330 decisions, 78 steps, 214.4 s. Four CPU cores against two before, and a different PyTorch, so this is not an isolated ablation of the fix |
| Is the update still most of the wall time? | Narrowly: 52.3% update, 41.8% collection | Same file. Two updates only |
| Where does a PPO step go? | Backward 0.735 s (54%); of the 0.582 s forward, the two belief models are 0.422 s (73%) | `ppo-parts.json`. **The measured step still ran every belief pass**: the code is from before `loss.skip_unweighted_belief`, and PPO has no such switch yet |
| How fast is replay training on a T4? | 1,850 rows/s (583 in brief 05) | `replay-summary.json`: 200 steps of 1,024 rows on random weights, belief passes on, 0.553 s a step |
| How much hosted GPU time is left? | **3.69 hours until the meter resets on 2026-11-21 00:00 UTC** | `gpu-budget.json`: 62,326.9 s used of 75,600 s when the last job ended. Nothing in the repository records a later job |

What follows.

- Brief 14's start condition ("the update still takes more than half the wall time") is met,
  narrowly and on a two-update smoke. Because belief passes are most of the forward, brief 14
  is worth running, but self-play is frozen until G3 and the remaining GPU time this week is
  3.69 hours. **Brief 14 waits for the meter reset; the remaining time goes to brief 13**
  (queue).
- Brief 13 was written for 20 GPU-hours. That is not available before 2026-11-21. Brief 13 is
  split into a part that fits now (20,000 games) and a part for after the reset (all prepared
  games); see the brief.

## 2026-11-16: review of brief 09

The branch `work/order-blind-model` (head `54d2418`) was reviewed and merged into
`work/roadmap` (merge `a40aa54`). Everything in it is smoke: 1,000 training games, 400 play
games a cell on `pool_b`.

Checked in review, on an Intel Xeon 2.10 GHz, four cores, PyTorch 2.13.0, Python 3.13:

- Headline numbers in `docs/briefs/09-report.md` against `09-artifacts/R{1,2,3,4,4-seed2}-best-policy-slot-check.json`,
  `stage4-summary.json`, `stage5-summary.json` and `play.json` (59, 159, 154 and 308 wins of
  400). All agree.
- The code diff outside `docs/`, read in full. Every new behaviour is behind a key that
  defaults to off. Terminal targets are labels only; no opponent truth enters a policy or
  value input.
- **Permutation test** (`tests/test_order_blind_model.py`): it perturbs every parameter, checks
  five permutations (acting moves, another own Pokémon's moves, an opponent's moves, own team,
  opponent team) and has a key-off control that must fail. On this machine the synthetic and
  live-Showdown cases pass; the replay-holdout case was skipped (no replay data here) and is
  the worker's claim.
- **Defaults unchanged.** With every new key off, the model built from
  `configs/gen9ou_typed_token_trunk.yaml` has the same 376 state tensors, 5,853,084 parameters
  and seeded initial values before and after the merge (one hash over names, shapes and
  values), so old checkpoints load strictly. The three rollout digests are equal before
  (`824a5f5`) and after the merge: `23405989f4665988`, `0efd4319547a4b3b`, `338cfca82f31b181`.
- 81 tests in the ten covering files pass (5 skipped for missing replay data);
  `.venv/bin/ruff check fog tests` is clean. Commits are authored as the owner; no account
  name or token in the artifacts.
- Not re-run: any training, the holdout permutation case, the play smoke.

Merge conflicts and how they were resolved (the only code written in review):

| Conflict | Resolution |
| --- | --- |
| `bc_loss`: brief 09's preview-only path against `loss.skip_unweighted_belief` | Both kept: preview-only returns first, then the belief skip decides how the model is called |
| `bc.keep_best_value` (brief 05) and `bc.save_best_value` (brief 09): two keys, two state layouts, one file `bc_best_value.pt` | One tracker. Either key keeps the best-value checkpoint; `save_best_value` also adds the value Brier and calibration columns and writes `value_selection.json`. The state file stores the score under both old names, and a state file from either branch still resumes. Both branches' tests pass unchanged |
| `loss` and `bc` key lists in `fog/config_schema.py` | Union |

One new test, `tests/test_bc_skip_belief.py::test_outcome_auxiliary_heads_survive_the_belief_skip`:
with `loss.skip_unweighted_belief` on and `loss.outcome_aux_coef` 0.1, the real model is called
with `need_belief=False`, the auxiliary loss is positive and equal to the loss without the
skip, and both auxiliary heads receive gradient. Brief 13 trains with both keys.

| Question | Answer | Evidence |
| --- | --- | --- |
| Can the model still read slot order? | No, with `model.order_blind_entities` | Permutation test above; R2 stored and shuffled metrics equal to four decimals |
| Does that help the policy? | Yes: shuffled NLL 1.9253 to 1.8799, top-1 26.64% to 28.24% | R1 against R2; larger than twice the seed gap |
| Do logistic value loss and dense outcome targets help at 1,000 games? | No measurable effect | R3 against R2: NLL 1.8806 against 1.8799. The best-value checkpoint is the untrained epoch 0 in every run, as expected from brief 05 (the value head needs about 20,000 games) |
| Do mechanical move features help? | Yes: NLL 1.8806 to 1.8041 (second seed 1.8139), top-1 28.5% to 33.3% | R4 against R3. Moves seen 1 to 19 times improve most; the 130 rows with never-seen moves get worse (1.884 to 2.016, wide intervals, cause unknown) |
| Does it change play? | R1 14.75% and 39.75%, R4 38.50% and 77.00% against SimpleHeuristics and MaxDamage | `play.json`, 400 greedy games a cell on `pool_b`, smoke |
| Can the lead head learn human leads? | 34.9% top-1 against 16.7% chance and 23.9% for the most-led species | R5, R5b |

**A gap the report does not mention.** The lead stage (`bc.scope=preview`) also trains the
shared embeddings and the private and current-state encoders, which the policy reads
(`09-artifacts/R5-scope.json`: `"policy_frozen": false`, 39 changed tensors including
`backbone.embeds.*`). R5 was never scored on the policy holdout and never played, so whether
lead training damages the policy is unknown. Brief 13 measures it before any lead-trained
checkpoint is used in play.

Decisions for other briefs.

- **Model keys for brief 13:** `configs/replay_mechanical_cpu.yaml`, that is R4:
  `model.order_blind_entities`, `model.mechanical_features`, `loss.value_loss=logistic`,
  `loss.outcome_aux_coef=0.1`, `bc.save_best_value`. The two value keys showed no benefit at
  1,000 games and none was expected there; R4 is the only configuration with a play result,
  and 20,000 games is where the value head can be judged (reference: Brier 0.215 on the old
  rows with the old loss).
- P2.2a is ticked: rows no longer carry the answer in their order, and the model cannot read
  order. P2.3 and G2 stay open.
- Brief 12 can start (it needed this merge).

## 2026-11-16: review of brief 11 and gate G1

The branch `work/g1` (head `1bfdef6`) was reviewed, its gate run completed in review, and
the result merged into `work/roadmap` (merge `194f534`).

**Gate G1 passes.** Validation teams of `data/teams/replay_v1` (141), pinned Showdown
`e440c4a`, default budget (K = 8, `budget_ms` 1,600, one search thread), matched four-game
blocks, no sequential stop, four shards with seeds 13000 to 13003, `fog` source `aa38e46`
(the merged tree has the same `fog/` search and reconstruction code).

| Date | Agent | Opponent | Games | Score [95% Wilson] | W-L-D | s per decision | Fallbacks | Rejected | Failed or truncated games |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-15 | EngineMCTS | SimpleHeuristics | 1,008 | 0.9286 [0.9110, 0.9429] | 936-72-0 | 1.801 | 53 of 28,020 (0.19%) | 0 | 0 |
| 2026-11-15 and 16 | EngineMCTS | MaxDamage | 1,008 | 0.9732 [0.9613, 0.9815] | 981-27-0 | 1.814 | 41 of 24,112 (0.17%) | 0 | 0 |

Command (both rows):
`python -m fog.eval.run_shards --shards 4 --games 1008 --seed 13000 --out-dir docs/briefs/11-artifacts/gate -- --pool data/teams/replay_v1/validation --device cpu --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":1600,"threads":1}' --opponent simpleheuristics --opponent maxdamage --max-turns 800 --progress-every 4`.
Files: `docs/briefs/11-artifacts/gate/merged.json`, `gate/shard_{0..3}.json`,
`gate-complete-summary.json`.

Gate clauses: at least 85% against SimpleHeuristics, met; at least 95% against MaxDamage,
met (the lower end of the interval, 96.1%, is above the bar); fallbacks under 5% of
decisions, met; zero rejected actions, met.

**How the run was completed.** The worker's session ended with the MaxDamage row at 504 of
1,008 games (490 wins) and the shard files committed. In review the same command was run
again; each shard resumed from its last complete block. So the MaxDamage row was played on
two machines:

| Part | Machine | Games | Wins | Score | s per decision |
| --- | --- | --- | --- | --- | --- |
| SimpleHeuristics, all | AMD EPYC 9V74, four-core quota | 1,008 | 936 | 0.9286 | 1.801 |
| MaxDamage, first half | the same | 504 | 490 | 0.9722 | 1.830 |
| MaxDamage, second half | Intel Xeon 2.10 GHz, four cores | 504 | 491 | 0.9742 | 1.797 |

The search budget is wall-clock, so a different processor is a different number of search
iterations. The two halves agree (490 and 491 wins), and the budget curve is flat from 200 ms
up, so the change of machine is recorded and not treated as a defect. During the second half
the reviewer ran test files and three short rollout probes at the lowest scheduling priority
for a few minutes in total; that can only have taken time from the search.

Checked in review:

- Every budget-curve cell and the SimpleHeuristics gate row in `docs/briefs/11-report.md`
  against `11-artifacts/budget-*/merged.json` and `gate-simpleheuristics-summary.json`; the
  fuzzer counts (14,362 positions, 1,214 to 1,043 with a mismatch, 821 of them hidden
  Regenerator healing) against `fuzz-showdown-{before,final}.json`. All agree.
- The code diff (`fog/env/state_recon.py`: pending charge turns from the public log, and the
  Teal-Tera Ogerpon identity) and its two test files. No search variant was added and no
  threshold changed. `git diff aa38e46 1bfdef6 -- fog tests` is empty, so the gate ran on the
  reviewed source.
- On the merged tree: 49 tests in six covering files pass, `ruff` is clean, and the three
  rollout digests are unchanged (`23405989f4665988`, `0efd4319547a4b3b`, `338cfca82f31b181`).
- All 23 commits are authored as the owner; no account name or token in the artifacts.
- Not re-run: the fuzzers, the budget curve, the SimpleHeuristics row.

Budget curve against SimpleHeuristics (400 games a cell, smoke; `11-artifacts/budget-*/merged.json`):

| `budget_ms` | K = 1 | K = 8 | K = 32 |
| --- | --- | --- | --- |
| 200 | 0.840 (0.23 s) | 0.910 (0.26 s) | 0.865 (0.41 s) |
| 800 | 0.783 (0.90 s) | 0.9175 (0.90 s) | 0.915 (1.03 s) |
| 3,200 | 0.775 (3.42 s) | 0.905 (3.64 s) | 96 games only: 0.927 |

Reading. Sampling several opponent teams matters (K = 8 beats K = 1 at every budget by 7 to
13 points); more search time does not, between 200 ms and 3,200 ms. One world searched for
longer gets worse, not better, which is what over-fitting a single guessed team looks like.
The agent is limited by what it assumes about the opponent and by the engine's fidelity, not
by search depth. Two consequences: labels for distillation can be bought at 200 ms (P3.2), and
P4's learned world sampler is aimed at the right thing.

Known failures inside the search, all handled by dropping the world or falling back: an engine
panic on Encore (424 and 320 dropped worlds in the two gate rows) and "no world with positive
weight and kept visits" (53 and 41 fallbacks). Residual reconstruction approximations are
listed in `docs/briefs/11-report.md`.

What follows. P1.3 is ticked. P1.4 stays unticked (eight of nine cells; nothing waits on the
ninth). EngineMCTS is anchor `A4`. P3 may start: brief 15 (pilot) is the next step, and brief
16 measures the agent once on the test teams for tier T1.

## 2026-11-16: search distillation pilot (P3, generation 0)

Brief 15, run in the planner's session on `work/roadmap`. Intel Xeon 2.10 GHz, four cores,
no GPU. Everything here is smoke (400 games a cell). Rounds are appended below as they
finish (one subsection a round).

Code: `fog/training/search_distill.py` (`e68f22a`, `5cf0780`, `e6ef687`), tests
`tests/test_search_distill.py` (a stored row replayed through the network reproduces the
log-probabilities recorded in play to 5e-7; targets sum to one, are zero on illegal actions,
and peak at the action the teacher plays). Model: `configs/search_distill_cpu.yaml` (brief
09's R4 keys, random initial weights, seed 20261201). Teacher: EngineMCTS, K = 8, 200 ms.

### Round 0: the teacher plays

`python -m fog.training.search_distill collect --pool data/teams/replay_v1/train --out runs/distill/round0 --games 1100 --beta 1 --seed 15000 --agreement-every 40 --workers 4`
(source `5cf0780`; started as `--games 1600` at `e68f22a` and restarted when two of four
processes turned out to hold all the long mirror games; finished games were kept, which is
why 111 games numbered above 1,099 are in the set).

| Quantity | Value |
| --- | --- |
| Games | 1,211: 550 teacher against teacher (both sides recorded), 306 against SimpleHeuristics, 355 against MaxDamage; train teams only; 1 truncated at 300 turns |
| Labelled decisions | 60,862 rows, 59,706 with a search distribution (the rest had one legal action or a fallback); 44,001 rows from mirror games |
| Cost | 19,565 core-seconds, 0.32 s a row including the label check |
| Teacher fallbacks | 51 of 60,862 decisions |
| Teacher results while collecting (train teams, not matched blocks, diagnostic only) | 267 of 306 against SimpleHeuristics, 347 of 355 against MaxDamage |
| Stored size | 40 MB, about 600 bytes a row |

**Label check** (roadmap G3, "if it fails": agreement of the reduced-budget teacher with
the default budget should exceed 70%). On 542 decisions the search was repeated at the same
budget with another seed and at 1,600 ms: the 200 ms teacher's chosen action equals the
second-seed choice on 86.5% and the 1,600 ms choice on **73.8%**. Limit: in this round the
checked decisions were the 40th and 80th of a side in a game (turn 29 at the earliest, median
turn 38), so they are late-game positions; from `e6ef687` on the check samples decisions at
random. `docs/briefs/15-artifacts/round0-summary.json`.

### Network 0: trained on round 0 only

`python -m fog.training.search_distill train --data runs/distill/round0 --out runs/distill/net0 --threads 4`
(source `5cf0780`): 58,160 training rows, 2,702 held out by game (61 games), 8 epochs, batch
256, 3e-4 to 3e-5, 285 rows/s, 28 minutes. `15-artifacts/net0-metrics.json`.

| Epoch | Held-out cross-entropy | KL to the teacher | Top-1 agreement with the teacher | Value MSE to outcome |
| --- | --- | --- | --- | --- |
| 0 (untrained) | 2.170 | 0.403 | 6.4% | 1.085 |
| 1 | 1.969 | 0.202 | 45.3% | 0.666 |
| 4 | 1.942 | 0.175 | 47.6% | 0.699 |
| 8 (best) | 1.935 | 0.168 | 49.3% | 0.750 |

The teacher's distributions are flat (cross-entropy minus KL is about 1.77 nats), and the
teacher agrees with its own second seed on 86.5% of choices, so 49% leaves a lot of room.

Play: the network alone, greedy, no search, lead from the untrained lead head, validation
teams, matched blocks, 400 games a cell (smoke), seed 15100:
`python -m fog.eval.run_shards --shards 4 --games 400 --seed 15100 --out-dir runs/distill/eval/net0 -- --pool data/teams/replay_v1/validation --device cpu --config configs/search_distill_cpu.yaml --checkpoint net0=runs/distill/net0/policy_best.pt --opponent simpleheuristics --opponent maxdamage --max-turns 300 --progress-every 20`

| Network | Against SimpleHeuristics | Against MaxDamage | File |
| --- | --- | --- | --- |
| Untrained (same seed) | 0.0075 [0.003, 0.022] (3-397) | 0.0475 [0.031, 0.073] (19-381) | `15-artifacts/untrained-play.json` |
| Network 0 (58,160 rows) | **0.450 [0.402, 0.499] (180-220)** | **0.815 [0.774, 0.850] (326-74)** | `15-artifacts/net0-play.json` |

No draws, truncations or failed games. For orientation only, because the team pools differ:
brief 09's R4, trained on 52,955 human replay rows, scored 0.385 and 0.770 on `pool_b`; brief
05's models on 20,000 and 40,955 replay games (old rows) scored 0.52 and 0.86 on `pool_b`.
Checkpoint `runs/distill/net0/policy_best.pt` is not in the repository (SHA-256
`5ed3c3ae8c326495cfaede9e1c0b92a275df278e5939c639d789324045a8d4ad`); the commands above
regenerate it.

### Round 1: network 0 plays, the teacher labels

`python -m fog.training.search_distill collect --pool data/teams/replay_v1/train --out runs/distill/round1 --checkpoint runs/distill/net0/policy_best.pt --games 1300 --beta 0.25 --seed 15001 --agreement-every 40 --workers 4`
(source `e6ef687`), stopped by hand once 60,000 rows were stored; games in progress were
discarded.

| Quantity | Value |
| --- | --- |
| Games | 1,058: 528 mirror (network 0 with the teacher's move one time in four, on both sides), 265 against SimpleHeuristics, 265 against MaxDamage; train teams only; none truncated |
| Labelled decisions | 63,785 rows, 62,080 with a search distribution; the teacher's move was the one played on 15,826 (24.8%) |
| Cost | 22,443 core-seconds, 0.35 s a row |
| Teacher fallbacks | 175 of 63,786 decisions (0.27%; 0.08% in round 0, so positions the network reaches are a little harder to rebuild) |
| Network 0 against the teacher on its own positions | same top choice on 47.3% of labelled rows (49.3% on held-out teacher positions) |
| Results while collecting (sampled play at temperature 1 with teacher moves mixed in; diagnostic only) | 83 of 265 against SimpleHeuristics, 156 of 265 against MaxDamage |

**Label check, now on decisions sampled at random over the whole game** (1,552 decisions,
median turn 18): the 200 ms teacher's choice equals its second seed's on 87.8% and the 1,600 ms
teacher's on **75.7%**. Above the roadmap's 70%. `15-artifacts/round1-summary.json`.

### Network 1: trained on rounds 0 and 1

`python -m fog.training.search_distill train --data runs/distill/round0 runs/distill/round1 --out runs/distill/net1 --threads 4`
(source `e6ef687`): random initial weights, 118,518 training rows, 6,129 held out by game (113
games), 8 epochs, about 7.3 minutes an epoch on four cores. `15-artifacts/net1-metrics.json`.

| Epoch | Held-out cross-entropy | KL to the teacher | Top-1 agreement with the teacher | Value MSE to outcome |
| --- | --- | --- | --- | --- |
| 0 (untrained) | 2.038 | 0.341 | 9.3% | 1.045 |
| 1 | 1.880 | 0.183 | 46.3% | 0.797 |
| 4 | 1.842 | 0.146 | 53.8% | 0.969 |
| 8 (best) | 1.829 | 0.132 | 56.7% | 0.974 |

The held-out sets of networks 0 and 1 differ, so compare the play rows, not these columns.
The value head is no better than a constant on held-out outcomes after the first epoch
(a constant scores 1.0 on this scale); it needs its own checkpoint or its own targets, as
the replay trainer did. Nothing below uses it.

### Play, gate-size rows: the network alone, no search

Greedy, lead from the untrained lead head, validation teams of `data/teams/replay_v1`,
matched four-game blocks, no sequential stop, seed 15200 (shards 15200 to 15203):
`python -m fog.eval.run_shards --shards 4 --games 1008 --seed 15200 --out-dir runs/distill/eval/<net>-1008 -- --pool data/teams/replay_v1/validation --device cpu --config configs/search_distill_cpu.yaml --checkpoint <net>=runs/distill/<net>/policy_best.pt --opponent simpleheuristics --opponent maxdamage --max-turns 300 --progress-every 40`
(`fog` commit `06a4615`).

| Date | Agent | Training rows | Opponent | Games | Score [95% Wilson] | W-L-D | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | Network 0 | 58,160 (teacher positions) | SimpleHeuristics | 1,008 | 0.448 [0.418, 0.479] | 452-556-0 | `15-artifacts/net0-play-1008.json` |
| 2026-11-16 | Network 0 | | MaxDamage | 1,008 | 0.815 [0.790, 0.838] | 822-186-0 | |
| 2026-11-16 | Network 1 | 118,518 (teacher positions and network 0's own) | SimpleHeuristics | 1,008 | **0.615 [0.585, 0.645]** | 620-388-0 | `15-artifacts/net1-play-1008.json` |
| 2026-11-16 | Network 1 | | MaxDamage | 1,008 | **0.909 [0.889, 0.925]** | 916-92-0 | |

No truncated or failed games. Policy play costs about 0.2 s a game on four cores.

Reading. Doubling the rows, with the second half taken from positions the network itself
reaches, moved the network from 44.8% to 61.5% against SimpleHeuristics and from 81.5% to
90.9% against MaxDamage; the intervals are far apart. One seed, and the two effects (more
rows; on-policy rows) are not separated. Network 1 is the strongest policy without search
in the repository: the replay models of brief 05 scored 0.52 and 0.86 on another pool, and
gate G2 asks a replay blueprint for 0.65 against SimpleHeuristics. It is still far below its
teacher (0.929 and 0.973) and below tier T1 (0.90 and 0.97). Strength is rising steeply with
rows at 120,000, so the run at scale (P3.2) is worth its cost. Checkpoint SHA-256
`f729ac9684bbc25f98d55a24e6c7ea7a785451f462ba5c1b938b25eb68246c59` (not in the repository).

## 2026-11-16: hosted GPU time, corrected

The review of brief 08 read 3.69 hours of GPU time left from
`docs/briefs/08-artifacts/gpu-budget.json` (62,326.9 s used of 75,600 s). The account's quota
endpoint, read on 2026-11-16 08:15 UTC, says otherwise: **17.31 hours used, 27.69 hours
remaining of 45.00, reset 2026-11-21 00:00 UTC**. The used time agrees with the artifact
(62,327 s is 17.31 h); the total is 45 hours, not 21. Brief 13 no longer needs to be split:
its 20 hours fit this week. Brief 14 (2 hours) fits too.

## 2026-11-16: search distillation pilot, round 2

Continues "search distillation pilot (P3, generation 0)" above; same machine and settings.

### Round 2: network 1 plays, the teacher labels

`python -m fog.training.search_distill collect --pool data/teams/replay_v1/train --out runs/distill/round2 --checkpoint runs/distill/net1/policy_best.pt --games 1300 --beta 0.25 --seed 15002 --agreement-every 40 --workers 4`
(source `06a4615`), stopped by hand at 60,000 rows.

| Quantity | Value |
| --- | --- |
| Games | 1,021: 510 mirror, 256 against SimpleHeuristics, 255 against MaxDamage; train teams only; none truncated |
| Labelled decisions | 61,680 rows, 60,056 with a search distribution; teacher's move played on 15,475 (25.1%) |
| Cost | 21,878 core-seconds, 0.35 s a row |
| Teacher fallbacks | 140 of 61,680 (0.23%) |
| Network 1 against the teacher on its own positions | same top choice on 55.8% of labelled rows (network 0 in round 1: 47.3%) |
| Label check (1,474 decisions sampled over the whole game) | second seed 87.4%, 1,600 ms **77.6%** |

`docs/briefs/15-artifacts/round2-summary.json`. The three rounds (186,327 rows) are also kept
in the owner's private hosted dataset, so later sessions need not regenerate them.

### Network 2: trained on rounds 0 to 2

`python -m fog.training.search_distill train --data runs/distill/round0 runs/distill/round1 runs/distill/round2 --out runs/distill/net2 --threads 4`
(source `94cbb98`): random initial weights, 178,317 training rows, 8,010 held out by game, 8
epochs, about 10.6 minutes an epoch on four cores. `15-artifacts/net2-metrics.json`.
`policy_best.pt` SHA-256 `6eea4684a48c70c94b8d9e2da083ff8ea9185e9960fdfc59073069c599cede29`.

| Epoch | Held-out cross-entropy | KL to the teacher | Top-1 agreement with the teacher | Value MSE to outcome |
| --- | --- | --- | --- | --- |
| 0 (untrained) | 2.017 | 0.355 | 8.3% | 1.056 |
| 1 | 1.852 | 0.190 | 47.5% | 0.776 |
| 4 | 1.806 | 0.144 | 56.2% | 0.877 |
| 8 (best) | 1.794 | 0.132 | 58.8% | 0.874 |

The value head's held-out error is lowest after the first epoch (0.776; a constant scores
1.0) and rises to 0.874 by epoch 8, the pattern of networks 0 and 1. Nothing uses it.

### Play: network 2 alone, no search

Same command as the gate-size rows above (`fog.eval.run_shards --shards 4 --games 1008 --seed
15200`, validation teams, greedy, `--checkpoint net2=runs/distill/net2/policy_best.pt`),
source `94cbb98`. One game in each row reached the 300-turn limit and is not counted.

| Date | Policy | Training rows | Opponent | Games | Win rate [95% interval] | W-L-D | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | Network 2 | 178,317 (rounds 0 to 2) | SimpleHeuristics | 1,008 | **0.682 [0.653, 0.710]** | 687-320-0 | `15-artifacts/net2-play-1008.json` |
| 2026-11-16 | Network 2 | | MaxDamage | 1,008 | **0.923 [0.904, 0.937]** | 929-78-0 | |

The learning curve against SimpleHeuristics is 0.448, 0.615, 0.682 at 58k, 119k and 178k
training rows: the third round added 6.7 points, the second 16.7. Against MaxDamage: 0.815,
0.909, 0.923. Still rising, more slowly. Rounds 3 and 4 follow the same recipe.

## 2026-11-16: what the teacher's value is worth (distillation pilot, rounds 0 to 2)

Measured on the stored rows, no training involved. The teacher's root value (EngineMCTS,
K = 8, 200 ms, mapped to [-1, 1]) against the result of the game the row came from, for the
side that searched. A constant 0 scores 1.0 on this scale.

| Round | Rows with a value | MSE to the result | Sign agrees with the result | Correlation |
| --- | --- | --- | --- | --- |
| 0 | 59,706 | 0.923 | 58.6% | 0.277 |
| 1 | 62,080 | 0.918 | 58.6% | 0.290 |
| 2 | 60,056 | 0.907 | 59.7% | 0.306 |

By quarter of the game (round 2; rounds 0 and 1 are within 0.03): MSE 1.004, 0.996, 0.977 in
the first three quarters and 0.668 in the last; the value's standard deviation is 0.18, 0.21,
0.22 and 0.52. The search value says almost nothing about who wins until the last quarter.

Reading. The network's value head scores 0.776 on held-out results after one epoch (network
2), so it is already a better predictor of the result than the search that labels it; what
it does after that epoch is memorise which games were won (training error 0.29, held-out
0.87 at epoch 8, from 3,290 games). Mixing the search value into the value target
(`--value-mix 0.5`, the pilot's setting) therefore pulls the head toward a weaker signal.
The earlier ledger sentences calling the head "no better than a constant" describe the last
epoch, not the best one. `value_best.pt` keeps the best epoch. Consequence for P4: leaf
values from this head are worth trying against the engine's evaluation; the number of games
(about 3,300), not rows, is what limits it.

Figures: `summarize` in `fog/training/search_distill.py` now writes the MSE and the sign
agreement into each round's `summary.json` (source `887bf60`); the quarters and correlations
came from a one-off pass over the same arrays (`teacher_value`, `outcome`, `turn`).

## 2026-11-16: search distillation, round 3 on this machine

Continues the pilot sections above with the same recipe.

`python -m fog.training.search_distill collect --pool data/teams/replay_v1/train --out runs/distill/round3 --checkpoint runs/distill/net2/policy_best.pt --games 1000 --beta 0.25 --seed 15003 --agreement-every 40 --workers 4`
(source `79283c9`). The run was interrupted once: the container was suspended 63 games in
and the command was restarted (it resumes; the games in flight were replayed). While it ran,
the test file `tests/test_search_distill.py` was run three times at low priority (about half
a minute each) and a pass over the stored rounds once; labels made in those minutes had less
than a full core.

| Quantity | Value |
| --- | --- |
| Games | 1,000: 500 mirror, 250 against SimpleHeuristics, 250 against MaxDamage; train teams only; none truncated |
| Labelled decisions | 60,938 rows, 59,265 with a search distribution; teacher's move played on 15,198 (24.9%) |
| Cost | 21,874 core-seconds, 0.36 s a row |
| Teacher fallbacks | 142 of 60,939 (0.23%) |
| Network 2 against the teacher on its own positions | same top choice on 58.7% of labelled rows (network 1 in round 2: 55.8%) |
| Label check (1,556 decisions) | second seed 87.9%, 1,600 ms **76.2%** |

`docs/briefs/15-artifacts/round3-summary.json`.

## 2026-11-16: search distillation, round 3 on hosted CPU machines

Same collection command as round 3 (network 2 acts, `--beta 0.25`, teacher K = 8 and 200 ms,
`--agreement-every 40`), one seed and one folder a machine, through
`docs/briefs/15-artifacts/hosted_cpu_job.py` (sources `887bf60` to `199fc6a`). The hosted
machine reports four CPUs, which are two cores with two threads each (Xeon 2.20 GHz). "This
machine" is the planner's four-core Xeon 2.10 GHz, one thread a core.

| Folder | Machine, processes | Seed | Games | Rows | Simulations a label | Core-s a row | Label check: second seed | Label check: 1,600 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `round3` | this machine, 4 | 15003 | 1,000 | 60,938 | not counted | 0.36 | 87.9% | 76.2% of 1,556 |
| `round3d` | this machine, 4 (25 minutes) | 18003 | 303 | 17,566 | 118,853 | 0.34 | 83.7% | 79.1% of 435 |
| `round3b` | hosted, 4 | 16003 | 600 | 35,825 | 43,018 | 0.55 | 88.6% | 74.5% of 842 |
| `round3c` | hosted, 4 | 17003 | 600 | 37,079 | 44,972 | 0.53 | 88.2% | 76.9% of 909 |
| `round3e` | hosted, 2 (80 minutes) | 19003 | 369 | 22,319 | 72,206 | 0.43 | 87.8% | 74.5% of 541 |
| `round3f` | hosted, 2 (80 minutes) | 20003 | 386 | 22,087 | 60,818 | 0.44 | 88.9% | 74.0% of 557 |
| `round3g` | hosted, 2 (80 minutes) | 21003 | 388 | 22,980 | 75,015 | 0.42 | 88.8% | 79.3% of 547 |

One game in `round3b` reached the 300-turn limit. Summaries:
`docs/briefs/15-artifacts/round3*-summary.json`. With network 2 acting there are now 218,794
rows; with rounds 0 to 2, 405,121.

What this says:

- **A hosted core is slower than a core here, and a hosted "CPU" is half a core.** With one
  process a hosted CPU a label gets 37% of the simulations it gets here; with one process a
  core, 51% to 63%. A hosted machine yields 7.4 rows a second with four processes and 4.7
  with two; this machine 11.6.
- **The label check does not see the difference.** Pooled, the 200 ms label agrees with the
  1,600 ms label of the same machine on 76.8% here (1,991 checks), 75.7% hosted with four
  processes (1,751) and 75.9% hosted with two (1,645). The check is relative (each machine's
  reference is eight times its own label budget), so it shows that agreement between a budget
  and eight times that budget is about the same from 43,000 to 119,000 simulations; it does
  not show that a 43,000-simulation label is as good as a 119,000-simulation one.
- Hosted collection therefore runs with two processes a machine from here on (nearer the
  teacher that brief 11's budget curve measured), and the rows of `round3b` and `round3c`
  are kept and used.
- A search opponent on a hosted machine gets fewer simulations than its budget in
  milliseconds implies, so gate and tier rows against EngineMCTS are not to be played there
  with four processes; with two, the opponent gets about 60% of the simulations it gets here.

The first hosted GPU job of the day (12:58 UTC, re-submitted 14:23) waited in the host's
queue and had not started by 15:55; network 3 is being trained on this machine instead.

## 2026-11-16: search distillation, network 3

`python -m fog.training.search_distill train --data runs/distill/round0 runs/distill/round1 runs/distill/round2 runs/distill/round3 --out runs/distill/net3cpu --threads 4`
(source `a8c9555`), on this machine because the hosted GPU job never left the host's queue:
random initial weights, 233,895 training rows, 13,370 held out by game, 8 epochs,
about 14.0 minutes an epoch on four cores. The hosted round-3 folders are not in it, so
it is the same recipe as networks 0 to 2 with one more round. `15-artifacts/net3-metrics.json`.
`policy_best.pt` SHA-256 `b1549dffda3ef95991912582c530a44a295b13c7acd5e6af5620e4ac35c20fcd`.

| Epoch | Held-out cross-entropy | KL to the teacher | Top-1 agreement with the teacher | Value MSE to outcome |
| --- | --- | --- | --- | --- |
| 0 (untrained) | 1.952 | 0.337 | 10.7% | 1.029 |
| 1 | 1.795 | 0.180 | 49.8% | 0.866 |
| 4 | 1.745 | 0.130 | 59.2% | 0.929 |
| 8 (best) | 1.734 | 0.120 | 61.7% | 0.923 |

### Play: network 3 alone, no search

`python -m fog.eval.run_shards --shards 4 --games 1008 --seed 15200 --out-dir runs/distill/eval/net3-1008 -- --pool data/teams/replay_v1/validation --device cpu --config configs/search_distill_cpu.yaml --checkpoint net3=runs/distill/net3/policy_best.pt --opponent simpleheuristics --opponent maxdamage --max-turns 300 --progress-every 40`
(source `1d770c4`; the same games as networks 0 to 2). One game against MaxDamage reached
the 300-turn limit and is not counted.

| Date | Policy | Training rows | Opponent | Games | Win rate [95% interval] | W-L-D | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | Network 3 | 233,895 (rounds 0 to 3) | SimpleHeuristics | 1,008 | **0.685 [0.655, 0.712]** | 690-318-0 | `15-artifacts/net3-play-1008.json` |
| 2026-11-16 | Network 3 | | MaxDamage | 1,008 | **0.925 [0.907, 0.939]** | 931-76-0 | |

**The learning curve has flattened.** Against SimpleHeuristics: 0.448, 0.615, 0.682, 0.685 at
58k, 119k, 178k and 234k training rows. Against MaxDamage: 0.815, 0.909, 0.923, 0.925. The
fourth round added 0.3 points where the third added 6.7, and the interval is 2.9 points wide,
so the gain from the last 56,000 rows is somewhere between nothing and about three points.
Agreement with the teacher's top move on held-out rows still rose (58.8% to 61.7%).
Brief 17's stopping rule needs two such rounds; this is one. What is being tried next, each
as its own row: a network on every stored row (about 450,000, the hosted folders included)
at the same number of training steps, and sharper targets (`--target-power`).

## 2026-11-16: where network 2 and the teacher differ (diagnostic)

From the rows stored when network 2 acted on this machine (`round3` and `round3d`, 76,304
labelled decisions; the network's own distribution at the moment of play is in each row, so
no forward pass is involved). "Network's choice" is its most probable action; the 1,008-game
play rows use that choice. A one-off pass over the arrays; nothing here is a gate.

| Quantity | Value |
| --- | --- |
| Same top choice as the teacher | 58.8% |
| Teacher's visit share on its own top choice, mean | 0.40 (entropy 1.61 nats over 7.6 legal actions; the network's 1.72) |
| Visits on the network's choice as a share of the visits on the teacher's top choice, mean | 0.84; below 0.10 on 1.3% of decisions |
| Agreement where the teacher's top share is under 0.3 (34% of rows), 0.3 to 0.5 (42%), 0.5 to 0.7 (16%), over 0.7 (7%) | 46.2%, 61.2%, 70.9%, **77.3%** |
| Teacher's top choice is a move, a switch, a Tera move | 74.8%, 24.8%, 0.4% of rows |
| Agreement when it is a move, a switch, a Tera move | 65.2%, **40.4%**, 3.3% |
| Network's top choice is a switch, a Tera move | 19.6%, never |
| Agreement by quarter of the game | 61.5%, 58.7%, 58.6%, 56.9% |

Reading, to be tested and not yet evidence about strength:

- The teacher's visit distribution is nearly flat, and by that distribution the network's
  choice is almost as good as the teacher's on nearly every decision. Yet the teacher wins
  93% against SimpleHeuristics and the network 68%. So visit shares say little about how much
  worse the second choice is, and a network fitted to them learns weak preferences. The
  network misses the teacher's choice on 23% of the decisions where the teacher is surest.
  Sharper targets (`--target-power`) weight exactly those decisions; two trainings with powers
  2 and 8 on rounds 0 and 1 are running on hosted CPU machines.
- Switching is where they differ most: the teacher switches on a quarter of its decisions
  and the network follows on 40% of them.
- The search itself rarely terastallizes: a Tera move is its first choice on 0.4% of
  decisions, about one game in six for a side. The network never does. Whether that is right
  play or a weakness of the engine's handling (HANDOFF section 10 lists it as unverified) is
  a question for P1, not for distillation.

## 2026-11-16: search distillation, three GPU trainings and what they answer

One hosted GPU job (`docs/briefs/15-artifacts/gpu_train_job.py`, trainer at `126f8be`,
Tesla T4, 2,332 s in all; `15-artifacts/gpu-job-2026-11-16b.json`). It was submitted at 12:58
UTC, re-submitted at 14:23 and started only at about 16:48: the host kept it queued for close
to four hours. The quota endpoint read 18.56 hours used and 26.44 remaining of 45 afterwards.
Each network was then played alone, greedy, with the command of the network 3 section
(`--seed 15200`, validation teams, 1,008 games an opponent, source `d21c2eb`).

| Network | Rows it was trained on | Change from the recipe | Top-1 with the teacher, held out | SimpleHeuristics [95%] | MaxDamage [95%] |
| --- | --- | --- | --- | --- | --- |
| 2 (earlier section) | rounds 0 to 2, 178,317 | none | 58.8% | 0.682 [0.653, 0.710] | 0.923 [0.904, 0.937] |
| `net2-p2` | rounds 0 to 2 | `--target-power 2` | 59.7% | 0.699 [0.670, 0.727] (705-303) | 0.926 [0.908, 0.940] (933-75) |
| `net2-vm0` | rounds 0 to 2 | `--value-mix 0` (value target: result only) | 58.7% | 0.665 [0.635, 0.693] (670-338) | 0.919 [0.900, 0.934] (926-82) |
| 3 (earlier section) | rounds 0 to 3, 233,895 | none, trained on the CPU | 61.7% | 0.685 [0.655, 0.712] | 0.925 [0.907, 0.939] |
| `net3g` | rounds 0 to 3 | none, trained on the GPU, same seed | 62.1% | 0.701 [0.672, 0.729] (707-301) | 0.932 [0.914, 0.946] (939-69) |

SHA-256 of `policy_best.pt`: `net3g` `e44b24c853a5a0ed9bf1e215cbe64d7c368a59f779a69be270a3d9342cf746e8`,
`net2-p2` `d056dcc367cf1c2347ffcf0364b85186a8018ce8fbecf5b0390206fcfb1aadea`,
`net2-vm0` `c82d08ff18d7c3b3565ec68ebf831e12cf2bed09a726a66cf3aca49e055893c4`. Metrics and play
files: `15-artifacts/{net3g,net2-p2,net2-vm0}-{metrics,play-1008}.json`.

What they answer:

- **Two trainings of one recipe differ by 1.6 points** against SimpleHeuristics (network 3
  on the CPU and on the GPU, same data, same seed, same held-out cross-entropy 1.7344). A
  difference between two networks smaller than about three points on these 1,008 games is
  not a finding. With both, the plateau reads 0.68 to 0.70 from 178,000 rows on.
- **Sharper targets, power 2: no gain that can be claimed** (+1.7 points, inside the above).
  Not kept. Power 8 on rounds 0 and 1 is still training on a hosted CPU machine.
- **The value head needs the search value in its target.** With the result alone the head is
  worse than a constant on held-out games from the first epoch (MSE 1.03, then 1.18; training
  error 0.07 at epoch 8). With the half-and-half target its best epoch scores 0.78 to 0.87.
  This corrects the reading in "what the teacher's value is worth" above: the search value
  is a weak predictor of the result, but as half of the target it is what keeps the head from
  memorising games. `--value-mix 0.5` stays.

### Round 4 on this machine (network 3 acts)

`python -m fog.training.search_distill collect --pool data/teams/replay_v1/train --out runs/distill/round4 --checkpoint runs/distill/net3/policy_best.pt --games 4000 --beta 0.25 --seed 15004 --agreement-every 40 --workers 4 --max-seconds 2400`
(source `d21c2eb`): 488 games, 28,158 rows, 118,190 simulations a label, label check 77.1%
of 658 against 1,600 ms and 89.8% against a second seed; network 3 has the teacher's top
choice on 60.6% of its own positions (network 2: 58.7%). `15-artifacts/round4-summary.json`.
Two hosted machines are collecting the same round; the host refused two more (it runs five
CPU jobs at once).

## 2026-11-16: network 2 against EngineMCTS (first reading on gate G3; the opponent ran below its budget)

`python -m fog.eval.run_shards --shards 4 --games 1008 --seed 15300 --out-dir net2-vs-mcts -- --pool data/teams/replay_v1/validation --device cpu --config configs/search_distill_cpu.yaml --checkpoint net2=net2.pt --controller mcts=fog.search.engine_mcts:showdown_controller --controller-kwargs 'mcts={"k":8,"budget_ms":1600,"threads":1}' --opponent mcts --max-turns 300 --progress-every 40`
on one hosted CPU machine through `docs/briefs/15-artifacts/hosted_cpu_job.py` (source
`2aa8dce`), 17,800 s.

| Date | Policy (no search) | Opponent | Games | Win rate [95% interval] | W-L | Opponent fallbacks | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | Network 2 (`6eea4684...`) | EngineMCTS, K = 8, 1,600 ms | 1,008 | **0.128 [0.109, 0.150]** | 129-879 | 53 of 34,984 decisions (0.15%); 0 rejected | `15-artifacts/net2-vs-mcts-1008.json` |

**Not a gate row.** The four shards shared two cores (the hosted machine's four CPUs are two
cores with two threads), so EngineMCTS had about 37% of the simulations it gets on a
four-core machine in 1,600 ms (ledger, "round 3 on hosted CPU machines"). The budget curve
is flat from 200 to 3,200 ms on a four-core machine, which makes this opponent about as
strong as the gated one, but that is an inference.

Reading: gate G3 asks for 30% against EngineMCTS at the default budget. A network that beats
SimpleHeuristics 68% of the time wins 13% against a search that was, if anything,
handicapped. The networks of rounds 2 and 3 score 0.68 to 0.70 against SimpleHeuristics, so
none of them is near 30% here. A proper row for the best of them (`net3g`, one evaluation
process a core, two hosted machines) is running.

## 2026-11-16: sharper targets on rounds 0 and 1 (trained on hosted CPU machines)

`python -m fog.training.search_distill train --data round0 round1 --out <name> --threads 4 --target-power P`
through `docs/briefs/15-artifacts/hosted_cpu_job.py` (source `364c724`), random initial
weights, 118,518 training rows, the rows and held-out games of network 1. A hosted machine
trains 165 rows a second (this machine 282). Play as for every network (`--seed 15200`,
1,008 games an opponent, source `4d6aebc`).

| Network | Power | Top-1 with the teacher, held out | SimpleHeuristics [95%] | MaxDamage [95%] |
| --- | --- | --- | --- | --- |
| 1 (earlier section) | 1 | 56.7% | 0.615 [0.585, 0.645] | 0.909 [0.889, 0.925] |
| `net1-p2` (`2358365e...`) | 2 | not comparable (sharpened targets) | 0.626 [0.596, 0.655] (631-377) | 0.920 [0.901, 0.935] (927-81) |
| `net1-p8` (`bf46c701...`) | 8 | 54.9% | 0.599 [0.569, 0.629] (604-404) | 0.889 [0.868, 0.907] (896-112) |

With the rounds 0 to 2 pair above (0.682 against 0.699): power 2 is +1.1 and +1.7 points,
both inside the 1.6 points two trainings of one recipe differ by; power 8, close to training
on the teacher's first choice alone, is 1.6 points lower. **Sharper targets are not the
missing strength. The switch stays off** (brief 17's first question).
`15-artifacts/{net1-p2,net1-p8}-{metrics,play-1008}.json`.

## 2026-11-16: search distillation, round 4 and network 4 (the data curve to 456,000 rows)

### Round 4 on hosted machines (network 3 acts)

Through `scripts/hosted.sh round-collect 4 4800 "a b d e"` (source `364c724`; the first run
of that command end to end): two hosted machines, two processes each, 80 minutes. The host
refused the other two (five CPU jobs at once).

| Folder | Seed | Games | Rows | Simulations a label | Label check: second seed | Label check: 1,600 ms |
| --- | --- | --- | --- | --- | --- | --- |
| `round4a` | 16004 | 359 | 22,936 | 74,847 | 90.4% | 77.0% |
| `round4b` | 17004 | 435 | 24,747 | 81,134 | 87.4% | 73.9% |

With `round4` on this machine (28,158 rows, section above) and four 40-minute hosted folders
with network 2 acting (`round3h` to `round3k`, seeds 22003 to 25003, 45,677 rows, 74,827 to
78,395 simulations a label, label check 73.9% to 79.4%), 526,639 rows are stored. Summaries:
`15-artifacts/round3[h-k]-summary.json`, `round4*-summary.json`.

### Network 4: every row stored when it started, four epochs

`python -m fog.training.search_distill train --data runs/distill/round0 runs/distill/round1 runs/distill/round2 runs/distill/round3 runs/distill/round3{b,c,d,e,f,g,h,i,j,k} runs/distill/round4 --out runs/distill/net4 --epochs 4 --threads 4`
(source `29b2a9b`, on this machine; stopped after epoch 3 and resumed, because a background
command here ends after two hours). Random initial weights, 455,501 training rows, 23,455
held out by game (8,184 games); the hosted `round4a` and `round4b` arrived later and are not
in it. Four epochs over these rows are 7,120 optimizer steps, network 3's eight epochs 7,312.
`policy_best.pt` SHA-256 `54a8db675714612c3a5c89a050795f328f77d895c23806c1350e9f429d11f165`.
`15-artifacts/net4-metrics.json`.

| Epoch | Held-out cross-entropy | KL to the teacher | Top-1 agreement with the teacher | Value MSE to outcome |
| --- | --- | --- | --- | --- |
| 0 (untrained) | 1.990 | 0.318 | 9.9% | 1.035 |
| 1 | 1.808 | 0.136 | 54.6% | 0.845 |
| 2 | 1.791 | 0.120 | 58.6% | 0.893 |
| 4 (best) | 1.778 | 0.107 | 61.5% | 0.912 |

Play, the command of every network (`--seed 15200`, validation teams, source `1d58d83`):

| Date | Policy | Training rows | Opponent | Games | Win rate [95% interval] | W-L-D | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | Network 4 | 455,501 | SimpleHeuristics | 1,008 | **0.708 [0.680, 0.736]** | 714-294-0 | `15-artifacts/net4-play-1008.json` |
| 2026-11-16 | Network 4 | | MaxDamage | 1,008 | **0.943 [0.927, 0.956]** | 951-57-0 | |

### The data curve

| Training rows | SimpleHeuristics | MaxDamage | Network |
| --- | --- | --- | --- |
| 58,160 | 0.448 | 0.815 | 0 |
| 118,518 | 0.615 | 0.909 | 1 |
| 178,317 | 0.682 | 0.923 | 2 |
| 233,895 | 0.685 and 0.701 | 0.925 and 0.932 | 3, trained twice |
| 455,501 | 0.708 | 0.943 | 4 (four epochs) |

**Strength is close to flat in data from 178,000 rows on**: 2.6 times the rows bought 2.6
points against SimpleHeuristics and 2.0 against MaxDamage, and two trainings of one recipe
differ by 1.6. Gate G3's "if it fails" line: label quality passes (74% to 79% everywhere),
data volume is now plotted to 456,000 rows, so "the model is the limit; widen it" is next.
`configs/search_distill_wide.yaml` (entity width 256, 10.1M parameters against 5.8M) is
training on network 4's rows and epochs; an eight-epoch run of the narrow model on all rows
is on a hosted machine, to say whether four epochs were too few.

## 2026-11-16: search distillation, the network at twice the width

`python -m fog.training.search_distill train --config configs/search_distill_wide.yaml --data <network 4's fifteen folders> --out runs/distill/net4w --epochs 4 --threads 4`
(source `549baea`, this machine, stopped after epoch 2 and resumed): entity width 256
instead of 128, 10.1M parameters instead of 5.8M, the rows, held-out games, seed and epochs of
network 4. 38 to 42 minutes an epoch (the narrow model: 27).
`policy_best.pt` SHA-256 `7a93b21f9289aded9b0c55dfca38ce72aafe946271c7fa6c3db23a3d53c7e5a4`.

| Epoch | Held-out cross-entropy, wide (narrow) | Top-1 with the teacher, wide (narrow) | Value MSE to outcome, wide (narrow) |
| --- | --- | --- | --- |
| 1 | 1.815 (1.808) | 52.9% (54.6%) | 0.846 (0.845) |
| 2 | 1.792 (1.791) | 58.1% (58.6%) | 0.860 (0.893) |
| 3 | 1.781 (1.782) | 60.5% (60.4%) | 0.862 (0.901) |
| 4 | 1.776 (1.778) | 61.9% (61.5%) | 0.876 (0.912) |

Play (`--seed 15200`, validation teams, `--config configs/search_distill_wide.yaml`, source
`62900c8`):

| Date | Policy | Training rows | Opponent | Games | Win rate [95% interval] | W-L-D | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | Network 4, wide | 455,501 | SimpleHeuristics | 1,008 | 0.695 [0.666, 0.723] | 701-307-0 | `15-artifacts/net4w-play-1008.json` |
| 2026-11-16 | Network 4, wide | | MaxDamage | 1,008 | 0.939 [0.923, 0.953] | 947-61-0 | |
| 2026-11-16 | Network 4 (narrow, section above) | 455,501 | SimpleHeuristics / MaxDamage | 1,008 each | 0.708 / 0.943 | | |

**Width does not help**: the wide network fits the teacher exactly as well as the narrow one
and plays the same (1.3 points lower, inside the 1.6 points two trainings differ by).

### Where this leaves gate G3

The three steps of G3's "if it fails" line have now each been taken once:

| Step | Result |
| --- | --- |
| Label quality: the 200 ms and 1,600 ms teachers agree on more than 70% | 74% to 80% in every folder |
| Data volume | 0.682 at 178,000 rows, 0.708 at 456,000 |
| Widen the model | 0.695 at twice the width |
| Also tried: sharper targets | no gain at power 2, lower at power 8 |

The policy without search sits at about 0.70 against SimpleHeuristics and 0.94 against
MaxDamage whatever is changed, agrees with the teacher's first choice on 62% of held-out
decisions, and network 2 won 13% against EngineMCTS where G3 asks for 30%. G3 is not passed
and nothing measured here points to a way of passing it with this teacher's visit
distributions as the only target. The gate row for `net3g` against EngineMCTS (two hosted
machines, one process a core) and two further trainings (eight epochs; twice the depth) are
still running and will be recorded, but none of them is expected to change this. What to do
about P4, which the roadmap opens only after G2 or G3, is the owner's decision
(`HANDOFF.md` section 4).
