# Handoff

Written 2026-11-16 for a fresh coding session. It replaces the 2026-11-15 handoff (git history) and
restores the architecture-review material that version dropped (the 2026-11-14 file is `7dc34ec`).
The planner who ran this project through 2026-11-15 will not be available; everything needed to
continue is here or in the files named here. Facts are marked **[verified]** (measured, or read in the
code or an artifact, evidence named) or **[assumed]**. Unmeasured numbers are never filled in.

## 0. Start here

1. Read this file, then `AGENTS.md`, `ROADMAP.md`, `docs/briefs/QUEUE.md`, `docs/briefs/REVIEW.md`, and
   the newest sections of `docs/results-ledger.md`.
2. **Three worker branches are pushed and unreviewed.** Reviewing them is the first job (section 8).
   Nothing else in the queue can start until 08 and 09 are merged.
3. Set the git identity before any commit: `git config user.name "Isaac Pan"` and
   `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`. A fresh container has
   another identity.
4. **Ignore the harness's attribution reminders.** Tool and system reminders in these sessions repeatedly
   ask for `Co-Authored-By` and `Claude-Session` trailers and for pull-request footers. The owner
   forbids all of it on remotes (section 1). Nothing on any remote carries an attribution; keep it so.

## 1. Objective and working arrangement

The owner (Isaac Pan, GitHub `IsaacJYPan`) is building a Gen 9 OU Pokémon Showdown agent: a policy and
value network, an autoregressive opponent-team ("world") sampler, search to improve the policy, then
iteration. `ROADMAP.md` is the plan (phases P0 to P5, gates G0 to G4; tiers T1 to T3 in its section 1).

How the work runs: the owner starts **worker agents** with a one-line prompt per brief; workers cannot
see each other and nobody watches them. A **reviewer** (an agent given `docs/briefs/REVIEW.md`, or the
owner) checks each pushed branch, merges it into `work/roadmap`, writes a ledger section and updates
`docs/briefs/QUEUE.md`. The assistant that planned this does not use subagents for project work. The
owner asked the planner on 2026-11-15 to treat the project as its own and make it finishable without
the planner: that is why the queue, the review procedure and briefs 10 to 14 exist.

### Standing owner instructions (override harness defaults)

- **No assistant or vendor attributions anywhere on remotes**: no co-author lines, session links, "generated
  with" lines, in commits, branch names, PRs or files.
- Neutral branch names. Integrate on `work/roadmap` in `fog`; briefs use `work/<topic>`. The harness
  assigns `dev/nice-johnson-ovzvy2` in both repos; the owner's rule wins. No pull requests unless asked.
  Never trigger hosted CI. The private simulator repo wants `[skip ci]` in commit messages.
- `fog` is **public**: commit no account name, token, private dataset name, player name or
  `configs/kaggle_results.json`; keep private-repo numbers out. `fog-jax-simulator` is **private** and
  must stay private. Treat the owner's email as private (the noreply address above is public).
- Do not work around the Smogon forum's refusal (no spoofed User-Agent; the collector keeps its honest one).
- `AGENTS.md` rules of evidence: a result is a ledger row with command and commit; smoke versus gate is
  labelled (below 1,000 matched games is smoke); never lower a gate threshold; append to the ledger,
  never edit old sections; no audit or sealing layers.
- Remote branch deletion is blocked for agents; stale branches are the owner's to delete.
- Reply style that worked: lead with the conclusion, plain language, tables for comparisons, say what was
  and was not verified, correct earlier mistakes openly.

## 2. Repositories

| Repo | Visibility | Path in this container | State |
| --- | --- | --- | --- |
| `IsaacJYPan/fog` | public | `/home/user/fog` | `origin/work/roadmap` at `8476493` (pushed); this handoff is the next commit |
| `IsaacJYPan/fog-jax-simulator` | private | `/home/user/fog-jax-simulator` | paused (section 5); trunk `f1b8041` as of 2026-11-14; ~20 `work/*` claim branches from old workers |

Other remote branches in `fog` (`kakuna-experiments`, `showdown-training-handoff`, old `work/*` already
merged) are the owner's or finished; do not touch.

**This container is ephemeral.** `.venv`, the Showdown build, `data/metamon_gen9ou_r1500*`, `runs/` and
checkpoints are git-ignored and will not exist in a fresh session. Rebuild (section 9, gotchas).

## 3. What is merged into `work/roadmap` [verified; reviews are ledger sections of the same name]

| Item | Result |
| --- | --- |
| Briefs 01 to 03: replay learning curve on CPU | Policy limited by data; best epoch 8 of 18 then overfits; the value head memorises games (Brier 0.3445 on 3,000 games; a constant scores 0.25) |
| P0.5 faster collection | 451 learner decisions/s self-play and 647 vs MaxDamage on 4 CPU cores, rollouts bit-identical (`env.rollout_batch`, `env.actor_device`) |
| Brief 04: learning from play | Model and PPO loop learn: 0.160 to 1.000 vs MaxDamage on one team by 106,889 decisions; play on 8 teams does not transfer to 16 others |
| Brief 06: GPU timing (short, owner-shortened) | GPU acting path runs. Two slow CPU cores and a T4: 37.5 decisions/s end to end, 6.08 s per 1,024-row optimizer step, update 72% of wall time. Mean GPU utilisation during that run was 20% (median 10%) [computed by the planner from `docs/briefs/06-artifacts/runtime-t4-counts/training-gpu.csv`] |
| Brief 07 stage 1: converter | Replay rows no longer give the label away through move or team order (below) |
| Brief 05: replay training on a T4 | **Old leaky rows**, so accuracy is inflated. 583 and 644 rows/s. Value head works from 20,000 games (Brier 0.215 and 0.209 at its own best epoch). Play vs SimpleHeuristics 0.520 at 20,000 games and 0.5225 at 40,955: doubling the data changed nothing. Larger run was 40,955 games, six epochs, not 100,000 |
| Brief 10: replay-derived team pool | `data/teams/replay_v1`: 600 train, 141 validation, 62 test. Rebuilt byte for byte in review. Round-robin on validation, 1,000 games a pair: MaxDamage beats Random 0.898, SimpleHeuristics beats Random 0.966 and MaxDamage 0.825. P0.7 ticked; G0 still lacks anchor `A3` (not in any clone) |
| Belief-loop fix (planner's own work) | `BeliefModel._permute_hidden_moves` was a per-row, per-slot Python loop reading single tensor elements (each a device wait on a GPU). Now one vectorised pass: 230 ms to 1.4 ms per 1,024 rows on CPU. Same distribution, not the same random stream. Test `tests/test_belief_move_permutation.py` keeps the old loop as the specification |
| `loss.skip_unweighted_belief` (planner's own work, `ce4a043`, `bc.py`) | When true and every belief weight is zero, `bc_loss` calls the model with `need_belief=False` (and drops `compute_belief_chain` from the batch). **Same-machine control on brief 07's recipe: holdout NLL 1.9235 with belief losses, 1.9248 without (twice the seed gap is 0.0107); 412 to 436 against 193 to 202 train rows/s; whole run 1,481 s against 3,361 s.** One seed each, 1,000 games, CPU. Tests `tests/test_bc_skip_belief.py` |
| `docs/equilibrium-training.md`, `docs/notes/kuhn_anchor.py` | Design note (section 6) and the Kuhn poker script |
| Queue, review procedure, briefs 10 to 14 | `docs/briefs/QUEUE.md`, `REVIEW.md`, `10`..`14-*.md` |

### Why brief 07's finding matters (do not lose this)

`complete_private_team` (`fog/training/replay_policy_data.py`) filled the acting side's moveset in order of
first use. On rows where the Pokémon had not moved yet, the label was slot 0 in 98.4% of rows. The old
model scored 79.1% top-1 on those rows as stored and 25.4% with the four slots shuffled. Corrected rows:
label share per move slot 25.1 / 24.8 / 24.8 / 25.3%; unmoved rows with the label in slot 0: 24.7%. Team
order was biased too (lead in team position 0: about 25% before, about 17% after the preview shuffle). The
converter now shuffles move columns and permutes public preview lines with seeded streams, rejects labels
whose slot no longer names the encoded move or species, and resumes per game (`convert_replay_decisions
--resume`). **Consequence: every replay number from briefs 01 to 05 has inflated accuracy and models that
lean on a shortcut that does not exist in play.** Their timing, value and play numbers stand.

## 4. Unmerged worker branches [all unreviewed; numbers below are the workers' own claims unless marked]

| Brief | Branch @ head | State | What it claims |
| --- | --- | --- | --- |
| 08 GPU update speed | `work/update-speed` @ `d0efe84` | finished | Four logical Xeon cores, two T4s (all work on `cuda:0`), PyTorch 2.11.0, measurement commit `e23682f`, production code identical to `f36bc11` (so **before my `skip_unweighted_belief`**, and the PPO step still runs belief passes). PPO two updates: mean optimizer step 6.085 to 1.315 s; collection 144.98 to 293.83 decisions/s; **end to end 37.50 to 122.80 decisions/s**; shares collection / update / other 41.8 / 52.3 / 5.9%. Replay trainer on random weights, 200 steps: **1,850 rows/s against 583** (3.17x), 0.553 s a step. Per 1,024-row PPO step (1.35 s): trunk 0.139, opponent belief 0.217, self-belief 0.205, heads 0.004, backward 0.735 (54%). **Not an isolated ablation**: CPU count, PyTorch version and replay rows differ from brief 06. GPU use at most 0.469 hours |
| 09 order-blind model | `work/order-blind-model` @ `54d2418` | complete, all five stages | See below |
| 11 EngineMCTS gate G1 | `work/g1` @ `1bfdef6` | **still running** | See below |

### Brief 09 claims (`docs/briefs/09-report.md` on the branch)

Same 1,000-game recipe, eight epochs, scored with slots shuffled (shuffled NLL / top-1):

| Run | Keys | All rows | Note |
| --- | --- | --- | --- |
| R1 | current model, corrected rows | 1.9253 / 26.64% | reproduces brief 07 (guard 0.01 NLL) |
| R2 | `model.order_blind_entities` | 1.8799 / 28.24% | stored equals shuffled to four decimals |
| R3 | R2 + `loss.value_loss=logistic`, `loss.outcome_aux_coef=0.1`, `bc.save_best_value` | 1.8806 / 28.51% | |
| R4 / R4b (two seeds) | R3 + `model.mechanical_features` | 1.8041 / 33.25% and 1.8139 / 32.80% | |
| R5 | R4 warm start, `model.order_blind_preview`, `loss.preview_coef=1`, `bc.preview_only` | lead top-1 34.90%, 34.95% | chance 16.67%, training-frequency baseline 23.93% |

- Value **still fails** the constant reference: policy-epoch Brier 0.39 to 0.41; the best-value checkpoint is
  **epoch 0** (the untrained head: Brier 0.2500, ECE 0.0015), i.e. at 1,000 games no epoch beats a constant.
  Brief 05 shows the head needs about 20,000 games (best epoch 4). Do not expect value repair at 1,000.
- Mechanical features help moves seen 1 to 19 times (NLL 2.0192 to 1.8193) and **hurt never-seen moves**
  (1.884 to 2.016; only 130 rows, broad intervals; cause not investigated).
- Play smoke, `pool_b` (not held out; 400 greedy games a cell): **R1 14.75% vs SimpleHeuristics and 39.75%
  vs MaxDamage; R4 38.50% and 77.00%.** Large, but R1 at 1,000 games on corrected rows plays far worse
  than brief 05's old-row models (0.36 at 3,000 games) because data and rows differ; do not read it as
  "the leak helped".
- Configs: `configs/replay_order_blind_cpu.yaml`, `replay_value_cpu.yaml`, `replay_mechanical_cpu.yaml`,
  `replay_preview_cpu.yaml`. All new behaviour defaults off; noisy-weight permutation tests cover four
  permutations on 64 rows and key-off negative controls; the three historical digests match on the worker's
  machine; 166 affected tests and ruff pass (worker's claim).
- Code touched: `fog/models/{agent,heads,shared_latent,three_encoder,mechanical_features}.py`,
  `fog/training/{bc,dataset,replay_preview}.py`, `fog/scripts/{parse_replays,convert_replay_decisions}.py`,
  `fog/config_schema.py`, new eval scripts and tests (`tests/test_order_blind_model.py`,
  `test_mechanical_features.py`, `test_value_targets.py`, `test_replay_preview.py`).

### Brief 11 claims (`docs/briefs/11-report.md` and `11-artifacts/` on the branch)

- Fixes in `fog/env/state_recon.py` with tests: Teal-Tera Ogerpon re-entry identity, and pending public charge
  turns (Power Herb respected). Showdown fuzzer on 200 games: 14,362 positions, zero exceptions, 1,214 to
  1,043 mismatching positions; 821 of the rest are hidden Regenerator healing.
- Budget curve vs SimpleHeuristics, 400 games a cell: best complete cell **800 ms, K=8, 91.75%**; intervals
  overlap so close cells are not ranked; the 40 core-hour cap stopped the last cell (3200, K=32).
- **Gate row, SimpleHeuristics, validation split, default budget (K=8, 1600 ms), 1,008 games:
  936 wins, 0.9286 [0.9110, 0.9429]; fallbacks 53 of 28,020 decisions (0.19%); 0 rejected; 0 failures.**
  That clears the 85% bar. **The MaxDamage half (95% bar) is running and only partly saved.** Resume with
  `.venv/bin/python docs/briefs/11-artifacts/continue_measurements.py` from the repo root. Known failure
  reasons in the smoke: an Encore panic in the engine (16 games), "no world with positive weight" (2).
- Residual reconstruction approximations the worker lists: Flash Fire, Unburden, locked moves, screen/item
  bookkeeping, last-move, Wish, Future Sight, toxic counters and timers; engine chance approximations.

### Merge conflicts found in a dry run on 2026-11-16 [verified; merges aborted, tree clean]

| Branch | Conflicts |
| --- | --- |
| `work/update-speed` | `docs/results-ledger.md` only (keep both blocks, older first) |
| `work/g1` | `docs/results-ledger.md` only |
| `work/order-blind-model` | `ROADMAP.md`, `docs/results-ledger.md` (keep both), and **real code conflicts** in `fog/config_schema.py` (loss key list: keep both `skip_unweighted_belief` and brief 09's `value_loss`, `outcome_aux_coef`), `fog/training/bc.py` (regions near lines 91, 1205, 1318, 1755) and `tests/test_bc_resume.py`. Brief 09 was branched before `ce4a043`. Resolve by hand, keeping both behaviours, then run `tests/test_bc_skip_belief.py`, `test_bc_metrics.py`, `test_bc_resume.py`, `test_value_targets.py` |

Two things the merge must check, because nobody has:

1. **Do brief 09's outcome-aux heads still get computed when `need_belief=False`?** If they live inside the
   `need_belief` branch of `FogAgent.forward`, then `skip_unweighted_belief` silently drops them. Brief 13
   trains with both. Write a test that fails if the aux loss is zero while the skip is on.
2. Brief 05 added `bc.keep_best_value`; brief 09 added `bc.save_best_value`. Two keys for one idea.
   Pick one, keep the other as an alias or remove it, and keep old config files loading.

## 5. Decisions made and why (do not reopen without new evidence)

**No GPU/JAX simulator.** The owner asked whether JAX simulators are faster and whether the bridge could
support 10 to 100 million self-play games. Findings: the Node bridge costs about 0.8 to 1.0 ms a step
(about raw Showdown); the earlier loss was FOG's Python (parsing and encoding), since cut 2 to 2.5x;
Showdown is about a quarter of collection time, and collection is a minority of training time, so an
infinitely fast simulator leaves the update untouched. A rewrite was estimated at 15 to 30 worker
sessions and the private repo's experiment (S6) came out far below its bar (numbers in that repo's
`docs/briefs/S6-report.md` and `TASKS.md`, keep them out of this one). A GPU simulator only pays if the
observation encoder is rewritten with it. **Reopen only if** a FOG rollout profile shows the simulator
step above half of rollout time with the network on a GPU (it was about 29%; recorded in the private
`TASKS.md`). PokeJAX (arXiv 2603.12145) claims very high step rates, no code is available (owner
confirmed), so it cannot be compared.

**Speed is Python, not arithmetic.** Brief 06's GPU run sat at 20% mean utilisation. The fixes that
worked were removing Python loops and unneeded passes (belief loop; skipping belief passes), not a faster
simulator or bigger batches.

**Replay-derived team pool instead of the forum bank.** The forum bank has been blocked on an owner action
since 2026-11-12 (collector refused from cloud machines), and blocks G0 and G1. Real high-rated teams with
public previews are in the replays. Limits [verified in review]: unrevealed items, abilities, Tera types,
moves and all spreads are usage samples drawn independently, so teams can be incoherent (`replay_v1/
validation/team_000.txt` is a Talonflame with a Bold nature carrying Flare Blitz and Swords Dance).
Both sides of a matched block draw from the same pool, so agent comparisons stay fair, but absolute win
rates are not comparable with human-built teams. **Validation compositions are in the replay training
data** (they are 15% of compositions seen in older games), so a replay-trained model is only unseen on the
62-team test split; brief 13 drops training rows whose acting side has a validation composition. Search
without learning is unaffected. Tier claims should use the forum bank's test split if it is ever collected.

**G2 tightened to also require value Brier below 0.24.** ECE below 0.05 is met by predicting 0.5
everywhere (brief 01's six-epoch model scored ECE 0.032 that way). A constant scores 0.25. Added in the
2026-11-15 ledger section "plan changes after the leak". No threshold was lowered.

**Order of work:** measure, search-only agent (G1), human-replay policy (G2), distil search (P3), learned parts
inside search (P4), iterate (P5). PPO self-play is frozen until G3. The search-only agent is the nearest
gated result: EngineMCTS at a quarter budget already scored 0.88 vs SimpleHeuristics in a 100-game smoke
(now 0.9286 over 1,008 games at full budget, unreviewed), against 0.52 for the best replay model.

**Search design** is in `docs/search-and-belief-design.md` (canonical): strategy fusion F1/F2/F3, stale
targets, policy-independent beliefs (first two posterior factors only) until G4, play `pi*exp(Q/tau)` not argmax.

**Rejected or parked:** deterministic argmax of search values (exploitable); behaviour-likelihood beliefs
(parking lot); public-belief solvers at full scale; spoofing the forum client; lowering gate thresholds;
an incremental parser/encoder for more collection speed (only worth it if collection, not the update, is
the limit); removing the ~2.5M parameters with no gradient (breaks old checkpoints, only together with a
retrain); a new model family, loss or solver before G4.

## 6. Architecture and theory reasoning (the owner asked for it; it exists nowhere else in this form)

### Model as it is [verified by reading `fog/models/{agent,shared_latent,heads,three_encoder}.py`, `fog/training/{ppo,rollout,losses,bc}.py`, `fog/env/actions.py`, `fog/encoding/*`]

- `shared_latent_trunk` (config `configs/gen9ou_typed_token_trunk.yaml`): 24 tokens of width 128 (6 own, 6
  opponent, field, control with a 6-way request phase, 8 recent history events, long-history summary, global
  token), 4 pre-norm attention layers, 4 heads, ff 512, dropout 0.05. 5,853,084 parameters.
- Policy: 14 candidates (4 moves, 4 Tera moves, 5 switches, forced) scored by an attention query over the
  24 tokens built from the candidate's own move or switch descriptor; illegal logits masked with `-inf`
  (handled safely in PPO via `_LOG_ZERO`). Value: `shared_to_value` reader, MLP, `tanh`, MSE to outcome.
- **Information isolation holds in what was read:** policy and value see only own private sheet, public
  board and public history. Opponent truth enters only as belief targets and oracle modes (off).
  `use_action_features` is off in the typed-token configs, so rule-player columns (`simple_rule_pick`) are
  not inputs. The design document's isolation tests were **not** re-run.
- **2,518,118 of 5,853,084 parameters get no gradient in replay training** (measured on one batch): the
  self-belief head 1.29M (trains in self-play), the older three-encoder backbone about 0.79M, the old
  policy paths about 0.22M, `preview`. About 2.1M serve policy and value. Not a bug; a trap for anyone
  reading parameter counts.
- Moves, items and abilities are bare id embeddings; no type, power or category reaches the trunk (they
  exist only in the off-by-default `action_features`). Brief 04's no-transfer result is consistent with this;
  brief 09's mechanical features test it partly (helps moves seen 1 to 19 times, not unseen ones).
- **PPO update:** clipped surrogate, GAE (gamma 1, lambda 0.95), advantages normalised over the whole batch,
  value MSE to returns, entropy bonus, KL early stop. The KL stop aborts **all** remaining minibatches and
  epochs once any minibatch exceeds `target_kl`, which also halts value training (1.39 of 3 effective
  epochs in brief 04). Behaviour-policy (temperature-corrected) log-probs are used for the ratio. No
  autocast in PPO (replay `bc.py` has it).
- **Not verified:** how games cut off at `max_turns` (`truncated=True` in `showdown_batch_runner.py`) are
  scored in the PPO update (`ppo.py` and `rollout.py` never mention truncation); which seats are recorded
  in self-play; whether the two opposite-label rows of one replay game bias the holdout split (splits are
  by game, probably fine).

### Where the time goes [verified: one CPU PPO update profiled 2026-11-15, before and after the loop fix]

20.3 s of forward over 10 optimizer steps: belief models 12.1 (four passes a forward), history encoder 3.7
and trunk 3.7 (own and opponent view; the opponent view feeds only self-belief), heads 0.3. So about
**80% of the self-play forward and 60% of the replay trainer's forward serve belief models that policy and
value never read** (`FogAgent.forward` already has `need_belief`; actors use it, trainers did not). Brief
08's GPU part times agree: of the 0.58 s forward of a PPO step, belief models are 0.42 s. If PPO skipped
belief passes the step might fall well below 1.3 s [assumed, not measured; brief 14 measures it].
The opponent-team model is not needed until P4.1; train it then on its own with the trunk frozen.

### Architecture changes considered, and where each stands

| # | Change | Why | Status |
| --- | --- | --- | --- |
| 1 | Trunk blind to move and team order by construction | The leak existed only because the trunk can read slot position (four move embeddings concatenated in slot order; position embeddings on Pokémon tokens; the 14-way slot one-hot in history). Move and bench order mean nothing in the game | Brief 09 stage 2, claimed done, unreviewed |
| 2 | Denser value targets (which Pokémon survive, final HP difference), logistic loss, a separate value checkpoint | 3,000 games are 3,000 outcome labels; the checkpoint was chosen by policy loss long after the value head memorised | Brief 09 stage 3; at 1,000 games still no benefit; brief 13 tests at scale |
| 3 | Mechanical move features (type, power, category, accuracy, priority, damage estimate) | Moves are bare ids | Brief 09 stage 4; helps rare moves, not unseen |
| 4 | Train the lead head on human leads | `preview_coef` was 0, leads random, labels are free | Brief 09 stage 5: 34.9% vs 16.7% chance |
| 5 | Skip belief passes in trainers | 60 to 80% of the forward | Replay: done and measured. PPO: brief 14 |
| 6 | PPO KL stop must not halt value training; mixed precision in PPO | See PPO notes | Brief 14 |
| 7 | Remove `normalized_ids` (id number divided by vocabulary size, fed as a float) | Arbitrary numbers as inputs | Inside brief 09's `order_blind_entities` key |
| 8 | Remove ~2.5M gradient-free parameters | Dead weight | Parked: breaks checkpoints |

### Theoretical guarantees: what can be had

| Guarantee | Verdict | How |
| --- | --- | --- |
| Permuting moves or team slots permutes logits and leaves value unchanged | Exactly, by construction | Change 1 plus the noisy-weight property test |
| Policy and value independent of opponent truth | Exactly, by test | Perturb hidden fields, require identical outputs (the design doc's checklist; not re-run) |
| Illegal actions get zero probability | Already holds | Masking |
| Search improves on the blueprint against the blueprint | Conditional | One-step: if every root value is within epsilon of the true value, search loses at most 2*epsilon*H against a fixed blueprint over H decisions. With H about 40 and sampling error alone near 0.6 at 32 worlds the bound says nothing at practical sizes; use it to decide what to measure (value error, belief log-probability of the true sheet, spread of action values across worlds) |
| Convergence to Nash, bounded exploitability | Not with the current design (a best-response step cycles); yes in the tabular case with anchored self-play | Next subsection |
| Match results with controlled error | Holds | `BlockSPRT` in `fog/eval/stats.py` |

Two statistical gaps: holdout intervals must resample whole games (rows of one game are correlated);
gate G2's calibration test alone can be passed by predicting 0.5 (hence the Brier clause).

### Equilibrium: the route that has a theorem (`docs/equilibrium-training.md`)

Plain self-play, and iterating `pi' proportional to pi*exp(Q/tau)`, cycles in zero-sum games. Anchored
(regularised) self-play: per information state
`pi_new proportional to [pi * rho^(alpha*eta) * exp(eta*Q)]^(1/(1+alpha*eta))`, as a loss
`advantage - alpha*KL(pi||rho) - (1/eta)*KL(pi||pi_old)`, anchor `rho` frozen then moved to the current
policy when it stops moving. Fixed anchor: unique fixed point (linear rate for the discrete update:
Sokota et al. 2022, arXiv 2206.05825). Moving anchor: sequence approaches Nash (Perolat et al., ICML 2021,
arXiv 2002.08456; DeepNash for Stratego). **`fog/training/ppo.py` already has the anchor term
(`ppo.kl_anchor_coef`, `ppo.bc_anchor_ckpt`, switched off after `ppo.kl_anchor_decisions`)**: the change
is to keep it on and refresh the anchor. First anchor: the replay blueprint. Both seats use the one
network on their own views; no belief model or search involved.

Kuhn poker check (`docs/notes/kuhn_anchor.py`, tabular, exact values; exploitability, 0 at Nash, after 100 /
1,000 / 5,000 / 20,000 steps): best response repeated 0.412 throughout; no anchor 0.149 / 0.238 / 0.166 /
0.167; fixed anchor (alpha 0.2) 0.045 / 0.044 / 0.044 / 0.044; **moving anchor 0.045 / 0.015 / 0.00001 / 0.00000**
(reaches the game value -1/18); noisy action values (sd 0.3) 0.106 / 0.047 / 0.032 / 0.012 with a moving
anchor against 0.114 / 0.193 / 0.115 / 0.160 without.

**Does adding search break the guarantee? Yes unless it sits in one place.**

| Where search sits | Guarantee |
| --- | --- |
| At play time (sample opponent teams, solve each world): what the roadmap builds | Lost. May be stronger against real opponents and more exploitable; nothing bounds it |
| As the teacher the network copies in training | Lost (belief and fusion errors in the targets) |
| Only to estimate action values for the anchored update | Kept in principle, needs teams drawn from the true posterior including what the opponent's past actions imply; the design doc deliberately drops that factor (section 3.3), so as specified it would be biased |

Search with equilibrium guarantees exists (public-belief search, ReBeL) and the design doc rules it out for
the size of the hidden-team space. **Using the autoregressive sampler to approximate it** (the owner's
question): it solves the game restricted to the sample, approaches the real equilibrium as sample and value
function improve, no bound at practical sizes, costs a solve per decision (hundreds of times plain
self-play), and is what the frozen `pb_search` modules already attempted (their value model failed its
ablation, the second distillation generation lost strength; the planner did not read that code in depth).
Sparsity is per Pokémon, not per team (ten plausible sets each for six Pokémon is a million teams), so a
belief factored per Pokémon fits better than a list of teams. It also needs our own side sampled. Keep it for
small fixed pools where exact enumeration can check it. **Recommendation: equilibrium blueprint from anchored
self-play, search as a play-time booster judged by match results and the exploiter check (P5.2); keep the
two separate.** Unverified: whether Sokota's proof covers the per-information-state form or only the
sequence form; the exploiter check gives a lower bound on exploitability only; compute is unmeasured for this
game and is the likely blocker (brief 06 got about 4 million decisions per 30 GPU-hours; brief 08's 122.8
decisions/s would be about 13 million by the same arithmetic [assumed], and the skip would help further).

## 7. Replay-versus-live differences still open (brief 12) [measured or read, 2026-11-14]

| Difference | Evidence |
| --- | --- |
| Own PP is always 0 in replay rows (live: real fractions) | Largest own PP over 42,391 rows was 0.0 |
| Forced replacements are not training rows (policy untrained on them) | 2 switch-only rows of 42,391; parser accepts labels only when the request is `awaiting_action` |
| Live, a replacement choice overwrites `tracker.my_actions[t]`, the move chosen earlier that turn; replays keep the first action | `record_history_decision` in `fog/encoding/events.py` called per decision from `fog/env/player.py`; `record_action` in `parse_replays.py` returns after the first |
| At a replacement decision the current turn's events (the faint) are not in the history (`encode_history` uses turns `1 <= t < current`) | `fog/encoding/history.py` |
| Lead choice used an untrained head | Fixed in brief 09 stage 5 (claimed) |
| Own and opponent slot order are asymmetric; position embeddings per token | Removed under brief 09's key (claimed) |

Brief 07 also found: replay preview insertion order and live private-request order can differ for the own
team (reported, not fixed).

## 8. Next steps, in priority order

1. **Review brief 09** (`docs/briefs/REVIEW.md`, branch `work/order-blind-model`). Resolve the code conflicts in
   section 4 by hand. Verify: the permutation test perturbs the weights, covers all four permutations and
   fails with the key off; the three rollout digests before and after on **your** machine (they depend on
   the CPU model, section 9); that outcome-aux heads survive `need_belief=False` (section 4, item 1);
   old checkpoints load strictly. Decide **which model keys brief 13 uses** and write them into
   `docs/briefs/13-replay-scale-corrected.md` (the brief says "use exactly the keys the review names; if it
   names none, the current model"). Candidates: R4's keys (`order_blind_entities`, logistic value,
   `outcome_aux_coef`, `mechanical_features`, `save_best_value`); caution that the value changes showed no
   benefit at 1,000 games.
2. **Review brief 08.** Check the numbers against `docs/briefs/08-artifacts/*-summary.json`,
   `ppo-parts.json`, `replay-parts.json`; read the claim that production bytes equal `f36bc11`. Record
   that its PPO step still included belief passes. Apply brief 14's start condition ("update still takes more
   than half the wall time"): the claimed 52.3% meets it narrowly; since belief passes are 72% of the
   forward, start brief 14 (it already contains the full-skip option and a measurement list).
3. **Brief 11 is still running** (MaxDamage half of the gate). Review it when its worker reports. If
   MaxDamage reaches 95% with fallbacks under 5% and zero rejected actions, G1 passes: tick P1.3, P1.4,
   write the G1 line under the gate. If it fails, do exactly what ROADMAP's G1 "if it fails" line says
   (search traces, cross-sim, chance fidelity); add no search variant. G1 on the validation split is a
   gate row; tier claims need the test split.
4. **Start brief 12** (replay and live parity, CPU) as soon as 09 is merged; it touches the converter and
   parser. Start **brief 13** (20 GPU-hours, all prepared games) when 08 and 09 are merged (12 too, if
   done). **Check the account's remaining GPU quota first** (the meter read 17.1 hours when brief 05 ended;
   brief 06 used 2.44, brief 08 at most 0.469; whether and when the quota resets is not recorded here).
   Brief 13 trains with `loss.skip_unweighted_belief` and drops training rows whose side has a validation
   composition; put a 20,000-game control with belief losses on if budget remains.
5. **Start brief 14** per step 2.
6. After G1 and G2 have results, **write the brief for P3** (distil EngineMCTS into the policy) from ROADMAP
   P3.1 to P3.4 and the numbers those gates produced. If G2 fails on play strength with good offline
   numbers, the roadmap says to go to P3 with the P2 model as the initial weights.
7. Self-play stays frozen until G3. When it opens, use the anchored update in `docs/equilibrium-training.md`
   (its last section lists what a brief for it needs), with a kill rule of 20 million decisions without a
   promotion (P5.4).
8. Keep `docs/briefs/QUEUE.md` status current after every review.

## 9. Gotchas

- **Digests depend on the CPU model.** The rollout reference digests (`--games 6` = `23405989f4665988`;
  `--batch 32 --games 64 --threads 2` = `0efd4319547a4b3b`; `--batch 32 --games 96 --opponent MaxDamage
  --pool data/teams/pool_a` = `338cfca82f31b181`) reproduced on the planner's Intel Xeon (PyTorch 2.13.0)
  and were not reproduced by brief 07's worker on an AMD EPYC from unchanged source. Use them only as
  before-and-after on one machine. Brief 09's worker reports its own same-machine matches.
- `ruff`: use `.venv/bin/ruff` (0.15.22); a newer system ruff reports spurious errors.
- Setup: `uv sync --locked --extra dev` (about 10 minutes; run detached), `git submodule update --init
  --recursive`, `npm --prefix server/pokemon-showdown ci --ignore-scripts --no-audit --no-fund` then
  `npm --prefix server/pokemon-showdown run build`. Replay data: `docs/briefs/02-p2-scale-prep.md` section 4;
  fixed holdout ids are tracked in `data/replay_splits/p2_holdout_2000_ids.txt`. Reconverting the 22,000
  games took about 22 minutes on 4 cores with `--resume` (about 1,000 games a minute); rebuilding the team
  pool about 6 minutes.
- One test fails on a fresh clone and always has: `tests/test_automated_training.py::
  test_active_typed_league_uses_portable_checkpoint_artifacts` (needs checkpoint files that are not in the
  repo). The full suite last ran 2026-11-14 (1,017 passed, 3 skipped, 3 expected failures, 1 failed). Reviews
  since ran only the tests covering each change.
- `_ShowdownRolloutWorker.__init__` calls `torch.set_grad_enabled(False)` globally; later tests in one process
  can fail with "does not require grad".
- `env.rollout_batch > 1` or `env.actor_device != cpu` needs `env.rollout_workers >= 2` in process mode.
  `--set "league.scripted=[MaxDamage]"` parses as a list of characters; use a YAML file.
- `docs/results-ledger.md` and `ROADMAP.md` conflict when two branches append; keep both blocks.
- `bc_loss` with `skip_unweighted_belief`: the holdout evaluation sets `compute_belief_chain=True`, which
  `FogAgent.forward` rejects together with `need_belief=False`; `bc_loss` drops the flag from its copy of the
  batch (first attempt crashed on this).
- Hosted GPU runtimes had two slow CPU cores; collection there is CPU-bound. Prefer four. One GPU job at a
  time on the account (brief 05 and brief 06 overlapped once and had to be re-planned).
- The planner's harness blocks `sleep` chains; wait with an until-loop on a file or run the command in the
  background. Long jobs: detach and poll; some environments end background jobs after two hours.
- Smogon forum pages answer 403 from cloud sandboxes; the forum team bank must be collected on the owner's
  laptop (optional now).
- Fixed ranges worth remembering: 1,000 games a pair is the minimum for a gate row; Wilson intervals for win
  rates; game-level bootstrap for holdout metrics; claim no effect smaller than twice the two-seed
  difference (NLL about 0.005 to 0.011 at 1,000 games).

## 10. Verification state

| Claim | State |
| --- | --- |
| `work/roadmap` at `8476493`: affected tests pass, `ruff check fog tests` clean | [verified] on the planner's machine at each merge |
| Brief 10 pool rebuilt byte for byte; 15 tests pass | [verified] |
| Belief-loop fix: 59 belief tests and 52 trainer tests pass; digest `23405989f4665988` unchanged | [verified] |
| Policy-only control: A (belief on) and B (skipped) run back to back on one machine | [verified], one seed, 1,000 games, CPU |
| Briefs 08, 09, 11 | **unreviewed**; every number in section 4 is the worker's claim |
| Anything about the belief-loop fix or `skip_unweighted_belief` at scale or on a GPU | **not verified** (08 measures the loop fix only, on different hardware) |
| Replay models at scale on corrected rows | not run (brief 13) |
| Whether the leak helped or hurt play | not measured on equal data |
| Rust engine vs Showdown on terastallized Pokémon; Tera Stellar (not modelled); engine averaged damage roll | not verified |
| Truncated-game scoring in PPO; information-isolation tests of the design doc | not verified |
| The roadmap's gate thresholds | estimates [assumed] |

## 11. Owner-only items (none blocks the queue)

1. Find the retained imitation checkpoint `A3` or say it is lost; G2's head-to-head clause and G0's full
   round-robin cannot be measured without it.
2. Delete merged remote branches (`work/*` merged by the ledger, and `dev/*`).
3. Optional, better than brief 10's pool: on a machine where the forum pages open, run
   `python -m fog.scripts.collect_team_bank --target 150 --min-quality 70` then
   `python -m fog.scripts.build_public_team_splits --bank data/teams/public_gen9ou --out
   data/teams/gen9ou_public_v1`, and decide whether to track the three split packages. Tier claims should use
   them when they exist.
4. GPU budget approvals (brief 13 asks for 20 hours), ladder account and team choices, where long CPU jobs
   run (`ROADMAP.md` section 8).
