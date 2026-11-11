# FOG roadmap

This is the single source of truth for what to do next on the whole project. It was written on
2026-11-11 after a full review of the code, the recorded experiments, and fresh
measurements. It is meant to be executed by people or coding agents who have no access to the
conversation that produced it.

Reusable prompt for any agent session:

> Continue the next task in ROADMAP.md and follow AGENTS.md.

Read [AGENTS.md](AGENTS.md) first (rules of evidence, scope freeze), then this file, then
[docs/search-and-belief-design.md](docs/search-and-belief-design.md) before touching search or belief
code. Every measured result goes in [docs/results-ledger.md](docs/results-ledger.md).

## 1. Goal

A Gen 9 OU agent that is competitive with strong human ladder players. Progress is measured on
three tiers, in order. Do not claim a tier without the stated evidence.

| Tier | Claim | Evidence |
| --- | --- | --- |
| T1 | Clearly beats scripted bots | ≥ 90% vs SimpleHeuristics and ≥ 97% vs MaxDamage, held-out teams, pinned Showdown, ≥ 1,000 matched games each |
| T2 | Beats search without learning | > 55% vs the EngineMCTS anchor (task P1) at equal time per move, ≥ 2,000 matched games |
| T3 | Human-competitive | Stable ladder rating over ≥ 300 games with fixed teams: 1500, then 1700, then 1900+ |

T3 at 1900+ is a research outcome, not a schedule. T1 and T2 are engineering and should be
reached by following this file.

## 2. Diagnosis: why earlier runs did not improve

These are measured facts (details and commands in the ledger), not opinions.

1. **Training was 3–4 orders of magnitude too small.** The self-play recipe trains for 240,000
   decisions (a few thousand games) at `lr 1e-5`, `clip 0.05`, one epoch. The policy cannot move
   measurably under those settings.
2. **Evaluation could not see a gain if there was one.** 256 games resolves about ±6 points of
   win rate at 95%. A real 3-point gain is invisible.
3. **The simulator was never the bottleneck.** On the Rust engine, which steps a turn in
   microseconds, self-play still ran at about 50 decisions/s on four CPU cores. Profiling showed
   59% of the time in one observation-encoding function (now fixed), then history re-parsing,
   then belief heads that the acting policy does not read. The engine step itself was under 3%.
4. **The starting policy is weak and the human data is barely used.** The initial checkpoint
   imitates SimpleHeuristics. Replay training used 2,000 games; the public corpus the importer
   already reads contains over a million Gen 9 OU games.
5. **Search was built top-down.** The repository has a two-sided public-belief solver, a joint
   counterfactual solver and depth-two expansion, but no search variant has ever beaten its own
   policy on Showdown. The sophisticated variants depend on a value model that failed its ablation.
6. **An unused strong baseline is already installed.** `poke_engine.monte_carlo_tree_search` runs
   about 350,000 iterations/s on states built by `fog.env.engine_battle.EngineBattle`. Nothing in
   the repository calls it.

The plan below fixes these in dependency order: measure, get a strong agent without learning,
get a strong policy from human data, then combine them, then iterate.

## 3. Strategy in one paragraph

Two approaches are known to work for competitive singles: imitation and offline RL on large human
replay corpora (no search), and determinized tree search on a fast approximate engine with
usage-based set inference (no learning). They fail in complementary ways. FOG's thesis is the
combination: a learned policy prior, value and set-belief model *inside* search on the Rust
engine, with pinned Showdown as the only judge. Under a small compute budget this ordering
matters: search buys strength with inference-time compute you already have, and human data buys a
policy without self-play. Self-play reinforcement learning comes last and only behind a gate.

## 4. Phases, tasks and gates

Dependencies: `P0 → P1`, `P0 → P2`, `P1 → P3`, `(P2 or P3) and P1 → P4`, `P4 → P5`.
P1 and P2 are independent and can be done in parallel. A gate is a number recorded in the ledger
with its command and commit. **Do not start a phase whose gate prerequisites are not met.** If a
gate fails, follow the "if it fails" line; do not improvise a new architecture.

Mark tasks `[x]` in this file in the same commit that records their ledger row. The ledger row
cites the `fog` commit that produced the result.

### P0. Measurement and throughput

- [x] **P0.1 Memoize nature/EV inversion** in `fog/encoding/entities.py` (`_invert_nature_evs`).
      Engine probe at batch 32 went from about 50 to about 120–165 decisions/s on four CPU cores.
- [x] **P0.2 Skip belief heads while acting.** Done in `fog` commit `3712b45`. `FogAgent.forward` computes the belief and
      self-belief heads whenever `opp_revealed` is in the batch (`fog/models/agent.py`); they were
      about 55% of forward time and the policy does not read them. Add an explicit
      `need_belief: bool` switch (default keeps today's behaviour) and turn it off in rollout
      actors (`fog/env/engine_runner.py`, `fog/env/player.py`) unless leakage diagnostics or
      search are enabled. Acceptance: action log-probabilities and values are bit-identical with
      the switch on and off on 1,000 recorded decisions; probe throughput recorded.
- [x] **P0.3 Stop re-parsing history every decision.** Done in `fog` commit `1719a31`. `encode_history` and
      `encode_history_sidecars` (`fog/encoding/history.py`) call `parse_turn_events` for every past
      turn at every decision, which is quadratic in game length. Cache the parsed per-turn records
      on the tracker; completed turns never change. Acceptance: encoded history arrays identical
      on the golden battles in `tests/golden`; probe throughput recorded.
- [x] **P0.4 Profile the `showdown_direct` rollout path** (done; profile and result in the ledger) with `cProfile` the same way (the
      command is in the ledger). It could not be profiled in the session that wrote this file
      because Showdown could not be built there. Record the top ten cumulative entries. Fix
      anything above 15% that is not model inference or the Node step. Likely suspects: per-step
      JSON state serialization in `fog/env/showdown_worker.js`, poke-env message parsing, and
      `fog/env/mirror_recording.py`.
- [ ] **P0.5 Batched inference across battles.** Today each rollout process plays one battle at
      a time with batch-1 CPU inference (`_ShowdownRolloutWorker` in `fog/training/ppo.py`). Change
      to N concurrent battles per process with one batched forward per wave, as
      `EngineRunner._agent_wave` already does for the engine backend, and allow the forward to run
      on the GPU. Target: ≥ 150 decisions/s on four CPU cores plus one GPU on `showdown_direct`,
      ≥ 400 on the engine backend. If the target is missed, record the new profile and stop; do
      not start a simulator rewrite (see section 7).
- [ ] **P0.6 One evaluation command.** Part (b), `BlockSPRT` in `fog/eval/stats.py`, is done
      (`fog` commit `d3d0bef`); parts (a) and (c) remain. Extend `fog/eval/showdown_direct_selection.py` (it already
      plays four-game matched blocks crossing teams and sides) so that:
      (a) any controller can be an opponent, not only three scripted names and checkpoints: accept
      a `module:factory` spec returning an object with `choose(view) -> action`;
      (b) it supports a sequential stopping rule: a two-sided SPRT on the block mean with
      `p0 = 0.50`, `p1 = 0.55`, `alpha = beta = 0.05`, minimum 100 blocks, maximum 1,000 blocks
      (4,000 games); put the test in `fog/eval/stats.py` with unit tests on simulated coins that
      check the realised error rates;
      (c) it writes its JSON report and prints one ready-to-paste Markdown ledger row (candidate,
      opponent, games, win rate, 95% interval, mean seconds per decision, fallback count, commit,
      command). Paste that row into `docs/results-ledger.md`.
- [ ] **P0.7 Fix the anchor set and the team split.** Anchors: `A0` Random, `A1` MaxDamage, `A2`
      SimpleHeuristics, `A3` the retained imitation checkpoint, `A4` EngineMCTS (from P1), `A5`
      any public learned Gen 9 OU agent that can be run locally (optional). Teams: build the
      public team bank per `KAGGLE_TRAINING.md`, split once into train/validation/test by
      composition, commit the three manifests' hashes to the ledger, and never evaluate on train.
      The test split is touched only for tier claims.

**Gate G0:** P0.6 and P0.7 done; `A0–A3` round-robin recorded with ≥ 1,000 games per pair;
measured `showdown_direct` throughput recorded after P0.2–P0.5.

### P1. EngineMCTS: a strong agent with no learning

Build the search-only anchor. It is also the first teacher and the base that P4 improves.

- [x] **P1.1 World sampler without a network.** Done in `fog/search/usage_worlds.py`
      (`fog` commit `bdcd63a`); it reuses `complete_private_team` and `priors.json`. Given the public reveal grid for the opponent,
      sample `K` complete team sheets `[6, 15]` from usage statistics (`data/usage`, spreads from
      `fog/constants/build_spreads.py`), with revealed fields clamped and candidates filtered by
      `fog.search.evidence.keep_mask_from_reveals`. Reuse the usage completion already used by the
      replay imputer (`fog/eval/replay_imputation.py`, `fog/scripts/build_replay_candidates.py`)
      rather than writing a second prior. Unrevealed team members are sampled from team-mate
      usage. Unit-test: revealed facts are never contradicted; outputs pass
      `truth_tokens`-style legality checks.
- [ ] **P1.2 EngineMCTS controller.** Built and tested on the engine backend (`fog` commit
      `bdcd63a`); the two follow-ups P1.2a and P1.2b below keep this open. New file `fog/search/engine_mcts.py`. At each decision:
      sample `K` worlds; reconstruct an engine state for each with
      `fog.env.state_recon.reconstruct_battle` (as `ShowdownWorldStepper` in
      `fog/search/engine_step.py` does); call
      `poke_engine.monte_carlo_tree_search(state, duration_ms=budget_ms // K)`; map each result's
      `side_one`/`side_two` entries (`move_choice`, `visits`) to FOG action indices; sum visit
      fractions across worlds weighted by world weight; play the argmax. If reconstruction or
      mapping fails for a world, drop it; if all fail, fall back to SimpleHeuristics and count it.
      Defaults `K = 8`, `budget_ms = 1600`. Expose the aggregated visit distribution and the
      visit-weighted mean score: P3 uses them as training targets. Terastallization needs care:
      the header of `fog/env/engine_battle.py` notes that the engine does not enforce one Tera
      per side and that `EngineBattle` masks it. Test what the built-in search does after a side
      has already used Tera; if it still proposes Tera moves, drop those entries when mapping.
- [ ] **P1.2a Run the Showdown adapter.** `ShowdownEngineMCTS` in `fog/search/engine_mcts.py`
      has never been executed. Run it on `showdown_direct`, fix what breaks, and change its
      all-worlds-failed fallback from the first legal move to SimpleHeuristics.
- [x] **P1.2b Terastallization in the built-in search.** Done in `fog` commit `cc9e2de`: the engine
      is now built with its `terastallization` feature and the search proposes Tera moves.
- [ ] **P1.2c Tera typing in sampled worlds.** `fog/env/engine_battle.py` builds a search world
      for an already-terastallized opponent Pokémon by setting its `types` to (Tera type,
      typeless). The engine now models Tera itself and derives the same-type attack bonus from
      `types`, so in those worlds the opponent loses the bonus on its original types and gets a
      doubled bonus on Tera-type moves. Keep the original types and set only the terastallized
      flag and Tera type; then check damage for a terastallized attacker against pinned Showdown
      with `fog/eval/cross_sim.py` or `fog/eval/chance_fidelity.py`. Also update the header
      comment in that file, which still says the engine does not enforce one Tera per side.
      The engine itself records a Tera used during a Pursuit switch for side one regardless of
      side; note any effect seen.
- [ ] **P1.3 Fidelity counters.** Log per game: decisions, worlds dropped, fallbacks, and the
      number of chosen actions Showdown rejected. Run `fog/scripts/fuzz_recon.py` on the
      validation teams and fix the top reconstruction failures until the gate's limit is met.
- [ ] **P1.4 Budget curve.** Win rate vs `A2` at `budget_ms` 200, 800, 3200 and `K` 1, 8, 32
      (nine cells, 400 games each is enough to see the shape). Record it; it sizes every later
      equal-time comparison.

**Gate G1:** EngineMCTS at the default budget scores ≥ 85% vs SimpleHeuristics and ≥ 95% vs
MaxDamage on pinned Showdown over ≥ 1,000 matched games each, with fallbacks on < 5% of
decisions and zero rejected actions.
**If it fails:** the cause is reconstruction fidelity or action mapping, not missing learning.
Inspect losses with `fog/eval/search_trace.py`, compare engine and Showdown outcomes with
`fog/eval/cross_sim.py` and `fog/eval/chance_fidelity.py`, and fix the largest mechanical
divergence. Do not proceed to P3 or P4 until G1 passes.

### P2. A blueprint policy and value from human replays at scale

- [ ] **P2.1 Import.** With `fog/scripts/import_metamon_replays.py` import Gen 9 OU games rated
      ≥ 1500 uploaded on or after 2024-02-01 (after the last major roster expansion) from the
      revision pinned in `docs/metamon-gen9ou.md`. Do it in three nested sizes: 2,000 (exists),
      20,000 and ≥ 100,000 games. Keep one fixed chronological holdout of 2,000 games that is
      newer than every training game, and exclude it from the candidate bank as
      `docs/kaggle-candidate-training.md` describes.
- [ ] **P2.2 Convert** with `fog/scripts/convert_replay_decisions.py` (one imputation per game is
      enough at this scale; three triples storage for correlated rows). Time the conversion on
      1,000 games first and record games/s and stored bytes per decision. If 100,000 games would
      take more than two days on the available cores, profile the converter before scaling; it
      shares the encoder hot paths fixed in P0. If the extrapolated dataset exceeds the storage
      available where training runs, do not shrink the corpus: store one compressed record per
      game and build decision rows in the data loader. If fewer than 100,000 games pass the
      filters, lower the rating floor to 1300 before widening the dates, and record the counts.
- [ ] **P2.3 Train policy and value together** with `fog/training/bc.py` on the typed-token
      model. Value target is the game outcome from the acting player's side for every decision.
      Belief losses stay at their current small coefficients. Use a normal supervised schedule
      (start `lr 3e-4` cosine, batch ≥ 1,024, until holdout loss stops improving), not the PPO
      continuation settings.
- [ ] **P2.4 Learning curve.** Report holdout policy top-1, policy NLL, value Brier score and a
      ten-bin value reliability table for the 2k, 20k and 100k models.
      Decision rule: if NLL still improves by > 0.02 from 20k to 100k, import more data before
      anything else; if it improves by < 0.005 and training NLL is close to holdout NLL, the model
      is the limit: train the `gen9ou_typed_token_wide.yaml` variant and compare.
- [ ] **P2.5 Play strength.** Evaluate the best model greedy (temperature 0) and at temperature
      0.5 against `A1–A3` with P0.6.

**Gate G2:** the replay blueprint beats `A3` head-to-head (> 55%) and scores ≥ 65% vs
SimpleHeuristics, and its value head has expected calibration error < 0.05 on the holdout.
**If it fails on play strength but the offline metrics are good:** this is covariate shift from
imputed own-team sheets and teacher-forced states. Go to P3 (on-policy relabelling fixes exactly
this) and use the P2 model as its initial weights. **If value calibration fails:** check for
leaked privileged inputs with `fog/eval/leakage_metrics.py` and for outcome-sign errors per side
before changing the architecture.

### P3. Distil search into the policy (expert iteration, generation 0)

- [ ] **P3.1 Soft-target relabelling.** `fog/training/dagger_heuristic_distill.py` already runs
      the learner in the simulator and asks a teacher for a label at each visited state. Extend
      the teacher interface from one integer label to `(visit distribution over legal actions,
      scalar search value in [-1, 1])` and add EngineMCTS as a teacher. Loss: cross-entropy to
      the visit distribution plus value regression to `0.5 * outcome + 0.5 * search value`.
      Generate on the `showdown_direct` backend so states are authoritative; the teacher uses its
      own reconstruction as in play.
- [ ] **P3.2 Generate** ≥ 1,000,000 labelled decisions with the learner acting (mix 25% teacher
      actions so strong lines are visited). Use a reduced teacher budget (`K = 4`,
      `budget_ms = 400`) chosen from the P1.4 curve to keep labelling under 0.5 s per decision.
      Run data generation on CPU workers; it is embarrassingly parallel.
- [ ] **P3.3 Train** from the P2 blueprint if G2 passed, otherwise from `A3`. Mix one human
      replay batch per three distilled batches when P2 data exists, to keep human-like play in
      distribution.
- [ ] **P3.4 Evaluate** the distilled policy with no search against `A2`, `A3`, the P2 blueprint
      and EngineMCTS.

**Gate G3:** the distilled policy without search beats its initial weights (> 55%) and scores
≥ 30% vs EngineMCTS at the default budget.
**If it fails:** check label quality first (top-1 agreement between the teacher at the reduced
and default budgets should exceed 70%), then data volume (plot strength at 250k, 500k, 1M
decisions). If strength is flat in data, the model is the limit; widen it.

### P4. Learned components inside search

This is the research contribution. Each item is an A/B against EngineMCTS at **equal wall-clock
per move**, using P0.6. Read the design document first; it defines the algorithms and the
information-isolation rules these tasks must not break. Do them in this order and keep each one
only if its own gate passes.

- [ ] **P4.1 Learned world sampler.** Replace the usage sampler with `BeliefModel` samples
      (`sample_opponent_worlds` in `fog/search/pb_search.py`), still filtered by hard evidence.
      Offline check first with `fog/eval/replay_belief_quality.py` on exact-truth self-play logs:
      the learned sampler must put more probability on the true set fields than the usage
      sampler. Then the A/B. Keep if ≥ 52% at 2,000 games; otherwise keep the usage sampler and
      move on. This item is not on the critical path.
- [ ] **P4.2 FogMCTS.** New file `fog/search/fog_mcts.py`: decoupled PUCT over engine states
      with the policy as prior for both players (each from its own censored observation) and the
      value head at leaves, per sampled world, root visits aggregated across worlds. The exact
      algorithm, its batching scheme and its test checklist are in the design document, section 4.
      `poke_engine.generate_instructions` runs about 25,000 calls/s from Python on one core, so
      network evaluation is the budget: design for one batched forward per search wave.
- [ ] **P4.3 Shared-tree variant** (information-set search), only if the fusion diagnostic in the
      design document, section 5, shows a measurable cost.
- [ ] **P4.4 Budget and ablations.** For the best variant: prior off, value off (engine
      evaluation at leaves), learned sampler off. Each is 1,000 games.

**Gate G4:** the best hybrid beats EngineMCTS ≥ 55% at equal time per move over ≥ 2,000 matched
games, and beats its own network without search ≥ 65%.
**If it fails:** run the ablations in P4.4 before anything else. Value-off better than value-on
means the value head is the problem (retrain on P3 data, check calibration by game phase).
Prior-off better than prior-on means the prior is too sharp (raise the prior temperature) or
wrong off-distribution (more P3 relabelling on search-visited states). If no configuration
passes after both remedies, stay on EngineMCTS plus the distilled policy for fast moves, and
record that result; it is a legitimate finding. In that case P5.2 and P5.3 still proceed, with
EngineMCTS (or whichever agent tops the anchor round-robin) as the champion.

### P5. Iterate, then the ladder

- [ ] **P5.1 Expert-iteration loop.** Generation `n + 1` = P3 with the best G4 searcher as the
      teacher, initialised from generation `n`. Promote a generation only if it beats the
      previous one (SPRT accept at `p1 = 0.55`) and does not lose more than 2 points against any
      fixed anchor. Keep targets from the two most recent generations only. Stop when two
      consecutive generations fail to promote.
- [ ] **P5.2 Exploiter check.** Before any tier-3 claim, train a best response for a fixed budget
      against the frozen champion (P3 pipeline with the champion as the opponent) and record how
      far it gets. A large gain means the champion is too deterministic; raise the mixing
      temperature described in the design document, section 3.
- [ ] **P5.3 Ladder protocol.** Needs the owner (section 8). Fixed set of three to five proven
      public teams, one account per configuration, move budget capped well inside the turn
      timer, ≥ 300 games per configuration, report rating, GXE and the rating trajectory.
- [ ] **P5.4 Optional: PPO fine-tuning arm.** Only after G3, from the champion, with ≥ 20 million
      decisions, `lr` ≥ `1e-4` with a KL constraint to the champion, against a league of past
      champions and anchors. Kill it if it has not passed a promotion test after 20 million
      decisions. Expert iteration is the default because its targets are dense.

## 5. Compute assumptions

Written for: a few CPU cores, one hosted notebook GPU with a weekly quota of tens of hours, and
no budget for hosted CI. Under that budget:

- Data generation and search are CPU-bound. Any machine with more cores helps more than a faster
  GPU. P3.2 at 0.5 s per label is about 170,000 decisions per core-day.
- Training the current 5.9 million-parameter model on 100,000 games fits in one notebook
  session per epoch or less. Make every training job resumable from its last checkpoint.
- Evaluation: 2,000 games of policy-only play is minutes; 2,000 games against EngineMCTS at
  1.6 s per move is about 30 core-hours. Use the sequential test and the reduced budgets.

If more compute appears, spend it in this order: P2 data scale, P3 label count, P4 search budget.

## 6. If this happens, do that

| Situation | Action |
| --- | --- |
| A result looks good on < 1,000 games | It is not a result. Run the sequential test to a decision. |
| Training metrics improve but win rate does not | Trust win rate. Check evaluation teams differ from training teams, then value calibration, then action-family statistics (`fog/eval/action_diagnostics.py`). |
| The agent beats engine opponents but not Showdown ones | It is exploiting engine approximations. Run `fog/eval/cross_sim.py`; generate P3 data on `showdown_direct` only. |
| Search is slower than the turn timer allows | Lower `K` before lowering per-world budget; the P1.4 curve shows the trade. |
| Showdown or the metagame changes (bans, new usage month) | Re-pin deliberately: update `pins` in `configs/default.yaml`, rebuild tables (`fog-build-constants`), download the new usage month, re-run the `A0–A4` round-robin, and start a new ledger section. Old checkpoints remain valid to initialise from; old win rates do not carry over. |
| The replay corpus revision changes | Keep the pinned revision until a phase ends. Never mix revisions inside one learning curve. |
| A notebook session dies mid-run | Jobs must be resumable (P2/P3 acceptance). Never restart from scratch silently; record the interruption in the ledger. |
| An agent cannot build Showdown in its sandbox | Develop and unit-test on the engine backend there; run Showdown evaluation where `npm ci` works. Do not replace a Showdown gate with an engine result. |
| Two gates in a row fail after their remedies | Stop adding components. Write the negative result in the ledger and return to the last passing configuration. |
| Someone proposes a new model family, loss or solver | Not before G4, unless a gate's "if it fails" line calls for it. Write the idea in section 9. |
| A unit test blocks an unrelated task | Fix the cause or mark the test with a reason and a ledger note. Never delete or skip silently. |

## 7. Frozen until further notice

Do not extend, tune or build on these until the named gate passes. Keep them compiling and keep
their tests; do not delete them.

- **A from-scratch accelerator simulator.** Not needed for any phase here. Reconsider only
  after G4, and only if a profile shows the simulator step above 50% of rollout wall-clock.
- **Public-belief and joint solvers** (`fog/search/pb_*.py`, `two_sided.py`, `joint.py`,
  `selective_depth.py`, `belief_table.py`, `direct_two_sided.py`): until G4. Reason in the design
  document, section 6.
- **Reconstruction and autoencoder trunks, gradient-surgery variants, matchup-prior and
  soft-value recipes** (`configs/gen9ou_contextual_*`, `*_flat_autoencoder_*`, `*_pcgrad*`,
  `*_matchup_prior*`, `*_soft_value*`): until G2.
- **PPO self-play** (`configs/gen9ou_typed_token_selfplay.yaml` and the automated league jobs):
  until G3, then only as P5.4.
- **New audit, sealing or provenance machinery.** The existing preflight checks stay. The ledger
  plus the commit hash is sufficient provenance for everything in this file.

## 8. Decisions only the owner can make

- Ladder play: which server, account names, and whether the account is declared as a bot. Read
  the server's current rules first. Until then, T3 is unmeasured.
- Which three to five teams represent the agent on the ladder.
- Spending on compute, and where long CPU jobs run.
- Whether any public learned agent is installed as anchor `A5`, subject to its licence.
- Un-freezing anything in section 7 ahead of its gate.

## 9. Parking lot

Ideas that are deliberately not scheduled. Add to this list instead of starting them.

- Behaviour likelihoods in the belief update (design document, section 3.3).
- Rating-conditioned policy and opponent models from replay ratings.
- A learned value at engine leaves inside the Rust search (needs an engine fork).
- Team preview and lead selection as a separate small search.
- Team building.
