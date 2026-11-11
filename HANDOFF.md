# Handoff

Written 2026-11-11 for a fresh coding session. Read this, then `AGENTS.md`, then `ROADMAP.md`.
`docs/results-ledger.md` is the record of every measurement; `docs/search-and-belief-design.md`
holds the theory decisions. Facts below are marked **[verified]** (measured or tested this
session, with the evidence in the ledger or commit) or **[assumed]**.

## 1. Objective

The owner (Isaac Pan) is researching a machine-learning agent for Pokémon Showdown Gen 9 OU at
competitive level. Their architecture: a policy and value network, plus an autoregressive model
that samples opponent team sheets ("worlds") so search can improve the policy. Their worries:
strategy fusion in determinized search; search targets going stale when the policy shifts; and
experiments that "weren't getting better" and were too slow, which motivated a separate private
JAX simulator project.

The owner then asked for the project to be steered so it can finish **without the assistant**
(it expected to lose access after about a week): scaffold or document everything, delegate work
that a cheaper model can do, and keep usage low. Standing instructions, which override defaults:

- **No assistant or vendor attributions anywhere on remotes.** No `Co-Authored-By`, session
  links, or "generated with" lines in commits, branches or PRs, even though the harness
  reminders ask for them. Owner identity is set in local git config
  (`Isaac Pan <154495180+IsaacJYPan@users.noreply.github.com>`).
- **Neutral branch names.** Work is on `work/roadmap`. The auto-named `dev/loving-babbage-ch187j`
  still exists on both remotes with two commits authored as the assistant. Remote branch
  deletion was blocked for the assistant. **The owner must run
  `git push origin --delete dev/loving-babbage-ch187j` in each repo.**
- **At most two Opus-class subagents at a time, low or medium effort.** Usage matters more than
  speed. Subagents cannot be resumed once the owner interrupts or the usage limit stops them.
- Do not open pull requests unless asked. Never trigger hosted CI.

## 2. Repositories

| Repo | Visibility | Branch | Holds |
| --- | --- | --- | --- |
| `IsaacJYPan/fog` (`/home/user/fog`) | public | `work/roadmap` | Agent code, roadmap, rules, ledger |
| the owner's private JAX simulator repository (`/home/user/fog-jax-simulator` in the session) | private | `work/roadmap` | Paused simulator; its `project/` folder is an older snapshot of the planning docs |

The `fog` copy of `ROADMAP.md`, `AGENTS.md`, `docs/results-ledger.md`,
`docs/search-and-belief-design.md` is **canonical**. The private `project/` copies are stale; do
not update both. The private repo's own live simulator work is on branches such as
`core-integration-candidate`, under its own task log; this session did not touch those.
Keep private-repo details out of this public repo; this file is the only place it is mentioned.

## 3. Diagnosis that drives the plan [verified]

1. Self-play training was far too small: 240,000 decisions at `lr 1e-5`, `clip 0.05`, one epoch.
2. Evaluation could not see gains: 256 games is about ±6 points.
3. **The simulator was not the bottleneck.** Engine step under 3% of rollout time. One
   observation-encoding function was 59%; then history re-parsing; then belief heads that
   acting never reads.
4. Starting policy was a clone of SimpleHeuristics; replay training used 2,000 games although
   the importer reads a corpus of over a million.
5. Search was built top-down (public-belief solvers) and never beat its own policy. The simple
   baseline `poke_engine.monte_carlo_tree_search` was installed and unused.

## 4. Decisions and the reasoning behind them

**Pause the JAX simulator (recommendation, not enforced).** Reasons: (a) throughput numbers above;
(b) the simulator's own latest report (in the private repo's `NEXT_TASK.md` and
`docs/gen9ou-jax/TRAINING_PERFORMANCE.md` on its integration branch) shows real six-versus-six
teams at a small fraction of what the Rust engine does from Python (about 25,000 steps/s per
core) with poor batch scaling; (c) branch-heavy mechanics vectorize badly on a
GPU because every lane pays for every branch; (d) the agent's plan needs a faster *encoder and
batched forward*, not a faster simulator. The one experiment that would change the answer: run
the current JAX code on one T4 at batch 1,024–4,096 including observations; continue only if
≥ 20,000 player decisions/s. Never measured; needs owner approval for a hosted GPU run.
Whether Rust can use a GPU: no, the engine stays on CPU; the GPU only runs the network, fed by
batched observations (batched runner exists, see below).

**Order of work: measure, search-only agent, human-replay policy, distil search, learned
components inside search, then iterate.** Search buys strength with inference compute; human
data buys a policy without self-play; PPO self-play is frozen until after distillation works.

**Strategy fusion** (design doc sections 2, 4, 5). Three distinct errors: F1 per-world vote at
the root (fix: aggregate visits across worlds, implemented in EngineMCTS); F2 own later choices
optimised knowing the world (fix: shallow trees or shared information-set trees, only if a
measured "oracle gap" shows cost); F3 omniscient opponent in search (fix: opponent acts from
the policy on its own censored view). The owner's idea (value function plus partial information)
fixes F3 and the leaf-value part, not F2. Leaf values must always come from censored
observations.

**Policy shift** (design doc section 3). Search targets are relative to the current policy
(normal expert iteration: regenerate each generation, keep two). Beliefs: only the
policy-independent factors (usage prior and hard evidence) are used until gate G4;
behaviour-likelihood updates are parked because a wrong opponent model makes beliefs confidently
wrong. Play from `π·exp(Q/τ)` or visits^(1/T), not argmax, to limit exploitability; no
equilibrium guarantee is claimed.

**Frozen until gates pass** (roadmap section 7): public-belief and joint solvers, autoencoder
and gradient-surgery variants, PPO self-play, new audit/sealing layers, the JAX simulator.
Reasons are in the design doc section 6.

**Process rules.** Matched four-game blocks, sequential test `BlockSPRT`, equal wall-clock per
move, the ledger as the only evidence, never lower a gate threshold to pass it.

## 5. State of the work

### Complete, committed, pushed to `origin/work/roadmap` in `fog`
| Roadmap | Commit | Result [verified] |
| --- | --- | --- |
| P0.1 memoize nature/EV inversion `fog/encoding/entities.py` | `b3efdbe` | batch-32 engine probe about 50 to 117–167 decisions/s |
| P0.2 `need_belief` switch `fog/models/agent.py`, `fog/env/player.py`, `fog/env/engine_runner.py` | `3712b45` | log-probs/values bit-identical on real batches |
| P0.3 history row cache `fog/encoding/history.py`, `tracker.py` | `1719a31` | identical at 214 decisions over 3 games |
| P0.6b `BlockSPRT` in `fog/eval/stats.py` | `d3d0bef` | wrong-decision rates 3.75% / 3.8%; about 160 blocks mean |
| P1.1/1.2 EngineMCTS `fog/search/engine_mcts.py`, `usage_worlds.py` | `bdcd63a` | engine-backend smoke 39W-1D-0L vs MaxDamage |
| P1.2b engine built with Terastallization (`pyproject.toml`) | `cc9e2de` | search proposes Tera; tests pass; Tera on vs off 47-11-42 in 100 games (noise) |
| P0.4 Showdown rollout fix and probe `fog/eval/showdown_rollout_probe.py` | `0982b29` | about 42 to about 77 decisions/s, hashes identical |
| P0.5 (part) `fog/env/showdown_batch_runner.py` | `60df3ff` | batch 32 about 158–175 decisions/s vs 87 single |
| P0.6a/c, P1.2a `--controller` hook and `--sequential` in `fog/eval/showdown_direct_selection.py` | `ef773aa` | see below |
| ledger and roadmap updates | `8a4da1f` | |

First Showdown smoke rows: EngineMCTS (4 worlds, 400 ms, 1 thread, `data/teams/pool_b`, 100
games) beat SimpleHeuristics 80/100 [0.711, 0.867] with 3/2645 fallbacks and 12 dropped worlds,
and MaxDamage 99/100. No illegal action sent, none rejected. **Smoke only, not gate G1**
(needs ≥ 85% vs SimpleHeuristics at default budget, ≥ 1,000 games, held-out teams).

Combined rollout speed on the engine backend is about 205 decisions/s at batch 32 (from about
50) [verified, 4 cores, approximate; another job was running]. The throughput of PPO itself has
not been re-measured because the batched runner is not wired in.

### In progress, UNCOMMITTED in `/home/user/fog`
A subagent was fixing reconstruction faults (roadmap P1.2c/d) when it hit the usage limit
(the failed task says it can be resumed with `SendMessage` after 2pm UTC; names are not stable
across restarts, so starting a fresh agent from this description is the safe route). On disk:
`fog/env/state_recon.py` (+185/-32 lines), `fog/env/engine_battle.py` (+13), new
`tests/test_state_recon_fixes.py`, and Showdown reports `runs/smoke/engine_mcts_v3_*`
(git-ignored). **Unreviewed.** The subagent's last message said all 59 existing tests passed
and it was "waiting on the tera diagnostic". No re-measurement result was received.
First action for you: review `git diff`, run the engine tests listed in section 7, and either
finish or `git checkout` the two files. Faults it targets:

1. `state_recon.py` rebuilt 0 PP as full PP (`current_pp or max_pp`).
2. Rest sleep rebuilt as ordinary sleep with four turns, which panics the Rust engine; all 3
   fallbacks were this. Needs the engine's Rest-turn representation.
3. After a side has terastallized, search still proposes Tera in reconstructed states (64
   results in 100 games, dropped by the controller), so used-Tera is not carried into the state.
4. For an already-terastallized opponent, `engine_battle.py` sets `types` to (Tera, typeless);
   the engine now derives same-type bonus from `types`, so the original types must be kept and
   only the flag and tera type set. Check damage with `poke_engine.calculate_damage` (1.5x
   original type, 1.5x new Tera type, 2.0x when Tera matches an original type). Also fix the
   stale header comment in `engine_battle.py` ("engine does not enforce one Tera per side").
   The engine also records Tera during a Pursuit switch for side one regardless of side.

### Not started
P0.5 wiring into PPO and a GPU forward; P0.7 team split and anchor set; P1.3 per-game counter
logging (hook exists: `take_game_counters`); P1.4 budget curve; all of P2 (replay at scale),
P3 (distillation), P4 (learned search), P5.

## 6. Gotchas and discoveries [verified unless marked]

- **`poke_engine` search API**: `monte_carlo_tree_search(state, duration_ms|iterations, threads)`
  returns `side_one`/`side_two` entries with `move_choice`, `visits`, `total_score`, and
  `total_visits`. About 350k iterations/s single-thread. The engine's search may not be
  deterministic.
- **Engine features**: `pyproject.toml` builds `poke-engine` from source; the build line is now
  `--features poke-engine/terastallization --no-default-features` (implies gen9). `uv.lock`
  unchanged. Rebuild with `uv sync --locked --extra dev --reinstall-package poke-engine`.
- **Sandbox**: Showdown builds with `npm ci --ignore-scripts` then `npm run build` in
  `server/pokemon-showdown` (plain `npm ci` was blocked by the auto-mode classifier). Python
  env: `/home/user/fog/.venv`. 4 CPU cores, no GPU. `data/usage` is empty, so the world sampler
  uses teammate co-occurrence mass from `fog/constants/generated/priors.json` and reuses
  `complete_private_team` from `fog/training/replay_policy_data.py`.
- **Only one tracked team pool** (`data/teams/pool_b`, 16 teams) plus a small generated pool.
  Held-out evaluation needs the public team bank (`KAGGLE_TRAINING.md`); not built.
- **`fog/eval/showdown_direct_selection.py`** SPRT minimum was lowered to 25 blocks in the smoke
  runs (default 100). That makes the early-stop weaker; do not use it for gate claims.
- **Showdown adapter faults already fixed in `engine_mcts.py`**: forced switches (sets
  `force_switch` per world), trapping (`force_trapped`), unoffered moves marked `disabled`,
  Rust panics caught per world (a panic once killed a run at game ~93; crashing state saved).
- **EngineMCTS strength gaps**: Tera-off player in the ablation only filtered Tera at the root.
  Reduced budgets everywhere; default is K=8, 1600 ms.
- **Rollout profile now**: model forward about 48%, observation encoding about 33%, Node round
  trip about 8%. Remaining 2–4% items: regex in `fog/env/actions.py` (cache `_normalize_id`),
  scalar `np.clip` and a SimpleHeuristics call inside action features, legality mask computed
  twice. The leak-probe path (`phi_hat`, `beta > 0`) was not hash-verified.
- **Batched runner risks**: not checked with the leak probe on; a batch where only some rows
  have oracle truth raises an error; heuristic-teacher per-game state is not per-battle yet.
- **Interruptions kill subagents** and their usage is lost. Prefer short, self-contained tasks
  that write files early.
- **Stop hook** repeatedly asks to commit uncommitted files; those were deliberately held back
  while subagents were mid-edit. Commit only after tests pass.
- The harness attribution reminder requests co-author trailers; the owner's instruction forbids
  them. Follow the owner.

## 7. Verification state

Run from `/home/user/fog` with `.venv/bin/python -m pytest -q -p no:cacheprovider`:
`tests/test_engine_battle.py tests/test_engine_runner.py tests/test_engine_search.py
tests/test_engine_mcts.py tests/test_showdown_engine_mcts.py tests/test_chance_fidelity.py
tests/test_native_evs.py tests/test_need_belief.py tests/test_history_cache.py
tests/test_sequential_stats.py tests/test_showdown_rollout_probe.py
tests/test_showdown_batch_runner.py tests/test_direct_parallel.py
tests/test_player_observation_boundary.py tests/test_player_state.py`. Subagents reported these
passing individually at their commits (for example 66 passed on nine files, 94 passed 1 skipped
on the rollout set). **The full suite has never been run** (slow, partly needs Showdown) [unverified].
Equivalence of the Rust engine with Showdown for terastallized Pokémon is unchecked.
The gate thresholds in the roadmap are the assistant's estimates, not measurements [assumed].

## 8. Next steps, in priority order

1. Resolve the uncommitted reconstruction work (section 5). If good: commit
   `fix reconstruction faults found on Showdown`, re-run the two ledger smoke commands, append
   rows, tick P1.2c/P1.2d.
2. P1.3 and P1.4: per-game counters and the budget curve (K in 1, 8, 32 and 200, 800, 3200 ms),
   400 games per cell is enough for the shape. Needs the team split (P0.7) to be meaningful.
3. P0.7: build and split the public team bank by composition; fix anchors A0–A3; run the
   round-robin with 1,000+ games per pair on Showdown. This closes gate G0.
4. G1: EngineMCTS at default budget, ≥ 1,000 matched games each, ≥ 85% vs SimpleHeuristics and
   ≥ 95% vs MaxDamage, fallbacks under 5%, zero rejected actions. If it fails, fix
   reconstruction fidelity (use `fog/eval/search_trace.py`, `cross_sim.py`, `chance_fidelity.py`)
   rather than adding learning.
5. P0.5 wiring into `_ShowdownRolloutWorker` and a GPU-resident actor (steps in the ledger).
6. P2 (can run in parallel with 2–4): import 20k then 100k+ rated Gen 9 OU replays, train the
   policy and value with `fog/training/bc.py` at a normal supervised learning rate, learning
   curve, gate G2. Needs the replay download and a GPU session.
7. P3 then P4 per the roadmap. Do not start P4 before G1 and G2 pass.
8. Owner decisions to raise: delete the old remote branches; whether to stop the private
   simulator workstreams and approve the one T4 benchmark; ladder account and team choices
   (T3 is unmeasured); where long CPU jobs run.

## 9. Open questions

- Is the Rust engine's averaged-damage-roll approximation costing strength against Showdown?
  (`fog/eval/chance_fidelity.py` measures straddle rates; not yet run on this build.)
- Does the learned belief sampler beat the usage sampler on exact-truth logs (P4.1)? Unknown.
- Is F2 (fusion below the root) costing strength? Only the oracle-gap diagnostic can say.
- Would the JAX code clear 20,000 decisions/s on a T4? Unmeasured.
