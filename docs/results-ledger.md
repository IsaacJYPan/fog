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
