# Handoff

Written 2026-11-16 (second version of that day, about 05:30 UTC) for a fresh coding session.
It replaces the earlier 2026-11-16 handoff (`824a5f5`); sections 5 to 7 and 9 are carried over
unchanged because they still hold. Facts are marked **[verified]** (measured, or read in the
code or an artifact, evidence named) or **[assumed]**. Unmeasured numbers are never filled in.

## November 17 update

The ledger entries dated November 17 supersede the earlier status below.
G0 and T1 passed on the replay-derived pools. A3 was recovered and evaluated;
see the ledger for its filename, SHA-256 and private recovery instructions.
Brief 12 is integrated. Brief 13 completed calibration at 2,948 training
rows/s and resumed G20 for the remaining 17 epochs, with a 2.25 h hard cap.
The last pre-submission account balance was 26.28 GPU-hours. Network 3g won
13.72% against full-budget search. The eight-epoch narrow policy won 74.80%
against SimpleHeuristics; its G3 teacher comparison is running on CPU.

## 0. Start here

1. Read this file, then `AGENTS.md`, `ROADMAP.md`, `docs/briefs/QUEUE.md`, `docs/briefs/REVIEW.md`, and
   the newest sections of `docs/results-ledger.md` (from "review of brief 08" down).
2. **What changed on 2026-11-16:** briefs 08, 09 and 11 are reviewed and merged; **gate G1
   passed**; the search-distillation loop runs and its networks are the strongest policies
   without search in the repository (section 4); hosted GPU training and hosted CPU
   collection work from the planner's session (section 9). Workers are on briefs 12 and 13
   (`work/replay-parity`, `work/p2-scale-corrected` each had a first commit at 13:07 UTC);
   their branches are the next reviews. Brief 16 had not started.
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
  assigns a `dev/...` branch in both repos; the owner's rule wins. No pull requests unless asked.
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
| `IsaacJYPan/fog` | public | `/home/user/fog` | `origin/work/roadmap` carries everything below; `git log` is the record |
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

| Brief 08: GPU update speed (merged 2026-11-16) | T4, four CPU cores: PPO optimizer step 6.085 to 1.315 s, 122.8 decisions/s end to end, update still 52.3% of wall time; replay trainer 1,850 rows/s. The measured step still ran all belief passes (73% of its forward). The 3.69 GPU-hours this review read as left was wrong: the meter's total is 45 hours, and 27.5 were left on 2026-11-16 |
| Brief 09: order-blind model (merged 2026-11-16) | Review confirmed: permutation test (all parameters perturbed, five permutations, key-off control), defaults bit-compatible (same state tensors and seeded values; three digests equal before and after on an Intel Xeon). R4 (`configs/replay_mechanical_cpu.yaml`) is the configuration brief 13 uses. Merge unified `bc.keep_best_value` and `bc.save_best_value`; a test shows the outcome auxiliary heads survive `loss.skip_unweighted_belief`. **Gap found in review:** the lead stage also trains weights the policy reads and nobody measured the effect (brief 13 does) |
| Brief 11: gate G1 (merged 2026-11-16, run completed in review) | **G1 passed.** EngineMCTS, K = 8, 1,600 ms, validation teams: 936 of 1,008 against SimpleHeuristics (0.9286 [0.9110, 0.9429]), 981 of 1,008 against MaxDamage (0.9732 [0.9613, 0.9815]); fallbacks 0.19% and 0.17%; no rejected action. The last 504 MaxDamage games ran on a different machine (490 and 491 wins in the two halves). Budget curve: K = 8 beats K = 1 by 7 to 13 points at every budget; 200, 800 and 3,200 ms at K = 8 score 91.0, 91.75 and 90.5% (400 games each), so search time is not the limit |
| `fog/training/search_distill.py` (planner's own work, P3.1) | The network plays on pinned Showdown, EngineMCTS labels every decision of a recorded side from that side's own view and plays its own move with probability `beta`; one file a game, resumable; soft-target training without belief passes. Tests `tests/test_search_distill.py`. Config `configs/search_distill_cpu.yaml` |
| Briefs 15 and 16 written; brief 13 split | 15: distillation pilot (section 4). 16: EngineMCTS once on the test teams for tier T1. 13: first split in two for lack of GPU time, then made whole again (20 GPU-hours) when the meter was read correctly |

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

## 4. Work in flight

### Worker briefs handed out on 2026-11-16 [the owner starts workers; whether each started is not known here]

| Brief | Branch | What it will hand back |
| --- | --- | --- |
| 12 replay and live parity (CPU) | `work/replay-parity` | A parity check between replay rows and live observations, then own PP, forced replacements and the history record fixed in the converter |
| 13 (GPU, 20 hours, no longer split: the meter had 27.5 hours left, not 3.7) | `work/p2-scale-corrected` | R4 keys on corrected games with the belief skip, value at both checkpoints, the lead-stage check, play rows on validation teams; each G2 clause |
| 16 EngineMCTS on the test teams (CPU, about 8 hours) | `work/t1-search` | One run, 1,008 games an opponent; tier T1 clauses met or not |

Review each with `docs/briefs/REVIEW.md` when its branch appears (`git ls-remote --heads origin`).

### The distillation loop (briefs 15 and 17, run by the planner) [verified; the ledger's "search distillation" sections of 2026-11-16]

Round 0 the teacher plays; in every later round the newest network plays with the teacher's
move one time in four; a network is trained from random weights on the rounds named in its
ledger section. Teacher: EngineMCTS, K = 8, 200 ms. Play rows: the network alone, greedy,
1,008 games an opponent, validation teams, one seed (15200) for all networks.

| Network | Training rows | SimpleHeuristics | MaxDamage |
| --- | --- | --- | --- |
| untrained (400 games, smoke) | 0 | 0.008 | 0.048 |
| 0 | 58,160 | 0.448 | 0.815 |
| 1 | 118,518 | 0.615 | 0.909 |
| 2 | 178,317 | 0.682 | 0.923 |
| 3, trained twice (CPU, GPU) | 233,895 | 0.685 and 0.701 | 0.925 and 0.932 |
| 4 (four epochs) | 455,501 | **0.708 [0.680, 0.736]** | **0.943 [0.927, 0.956]** |
| EngineMCTS, the teacher (gate G1) | | 0.929 | 0.973 |

**Network 2 against EngineMCTS: 0.128 [0.109, 0.150] over 1,008 games** (the opponent ran
below its budget on a hosted machine; not a gate row). Gate G3 asks for 0.30.

What is settled [measured]:

- **Strength is close to flat in data from 178,000 rows on.** 2.6 times the rows bought 2.6
  points. Two trainings of one recipe differ by 1.6 points, so smaller differences are not
  findings.
- **Sharper targets do not help**: `--target-power 2` is +1.1 and +1.7 points on two data
  sizes (inside the noise), power 8 is 1.6 lower. The switch stays off.
- **The value head needs the search value in its target**: with the result alone
  (`--value-mix 0`) it is worse than a constant on held-out games. With the half-and-half
  target its best epoch scores 0.78 to 0.87 where a constant scores 1.0 and the teacher's own
  root value 0.91; `value_best.pt` keeps that epoch. The teacher's value says almost nothing
  about who wins before the last quarter of a game.
- **Label quality passes G3's bar everywhere** (74% to 79% agreement between the 200 ms and
  1,600 ms teachers), on this machine and on hosted ones.
- Where the network and the teacher differ (ledger, "where network 2 and the teacher
  differ"): the teacher's visit distribution is nearly flat (top share 0.40), the network
  follows its switches only 40% of the time, and neither terastallizes much (the search's
  first choice is a Tera move on 0.4% of decisions).

What is running or next [state at 20:00 UTC on 2026-11-16; the ledger says what finished]:

1. `configs/search_distill_wide.yaml` (entity width 256) on network 4's rows and epochs, on
   the planner's machine: G3's "if it fails" line, third step. Output `runs/distill/net4w`.
2. The narrow model for eight epochs on about 500,000 rows, on a hosted CPU machine (`net4-e8`).
3. `net3g` against EngineMCTS, 1,008 games on two hosted machines with one evaluation
   process a core (`--only 0,1` and `--only 2,3` of one four-shard row, seed 15400).
4. If the wide model is flat too: the distilled policy alone will not reach G3 at this
   teacher and this architecture family. Record that under G3; the question for the owner is
   then whether P4 (the network inside the search) may start from G1 and a policy that is
   strong enough to be a prior but not to pass G3. Do not start P4 before that is decided.

**Data and checkpoints are not in the repository** (`runs/` is ignored and the container is
ephemeral). Every round (`<folder>.tar`) and the acting networks (`net2.pt`, `net3.pt`,
`net3g.pt`) are in the owner's private hosted dataset (`kaggle datasets list --mine`); a new
session downloads them instead of regenerating them. SHA-256 of each reported checkpoint is
in the ledger. `scripts/hosted.sh` (settings from the environment, see its header) runs
collection, training and evaluation on the hosted machines.

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

1. **Finish the distillation experiments in flight** (section 4, "What is running or next")
   and record each as a ledger row. More rows of the same kind are not the lever: the data
   curve is flat. The G3 row proper is 1,008 games against EngineMCTS at K = 8 and 1,600 ms
   with one evaluation process a real core.
2. **Review brief 13** when it reports. It decides G2's measurable clauses (65% against
   SimpleHeuristics; value ECE below 0.05 and Brier below 0.24) and measures whether the lead
   stage moves the policy. If brief 12 is merged first, its rows should be used.
3. **Review brief 12**, then brief 16. Brief 16's result goes under the tier table in
   `ROADMAP.md` section 1.
4. **Brief 14** (PPO update levers, GPU 2 hours) when the GPU is free; self-play stays frozen
   until G3, so it is the lowest priority.
5. After G3: P4 (learned parts inside search). The budget curve says the search is limited
   by what it assumes about the opponent's team, not by search time, which is what P4.1's
   learned world sampler addresses; read `docs/search-and-belief-design.md` first.
6. Self-play stays frozen until G3. When it opens, use the anchored update in
   `docs/equilibrium-training.md`, with a kill rule of 20 million decisions without a
   promotion (P5.4).
7. Keep `docs/briefs/QUEUE.md` status current after every review.

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
- **A cloud session's container is suspended when the session goes idle** unless a
  harness-tracked background command is running. A job detached with `setsid nohup` does not
  count: on 2026-11-16 the loop died minutes after the planner's turn ended. Files survive
  the suspension; processes do not. Run long jobs as tracked background commands (two-hour
  limit each) and make them resumable.
- **Hosted account from the planner's session.** The CLI lives in its own virtual environment
  (not in `.venv`); the token is an environment secret; the environment's network list needs
  `www.kaggle.com`, `api.kaggle.com`, `storage.googleapis.com` and
  `www.kaggleusercontent.com`. Read `kaggle quota --format json` before a GPU job (27.5 of 45
  hours were left on 2026-11-16 13:00 UTC; reset 2026-11-21). Never print or commit the
  account name or the dataset name.
  - GPU training: `docs/briefs/15-artifacts/gpu_train_job.py` (a list of runs in one job). A
    T4 trains 118,518 rows in 44 s an epoch against about 7 minutes on four cores, and
    reproduces the CPU run's held-out numbers. A GPU job can wait in the host's queue: one
    waited more than half an hour on 2026-11-16.
  - CPU collection or evaluation: `docs/briefs/15-artifacts/hosted_cpu_job.py` (builds
    Showdown and the engine, runs the listed commands, stores each output folder as a tar).
    Output of a job that overruns is lost, hence `collect --max-seconds`. Logs are not
    readable while a job runs.
  - `kaggle kernels output <kernel> -p <folder>` downloads results; the session's permission
    check refused it once and allowed a plainer form of the same command later.
- Smogon forum pages answer 403 from cloud sandboxes; the forum team bank must be collected on the owner's
  laptop (optional now).
- Fixed ranges worth remembering: 1,000 games a pair is the minimum for a gate row; Wilson intervals for win
  rates; game-level bootstrap for holdout metrics; claim no effect smaller than twice the two-seed
  difference (NLL about 0.005 to 0.011 at 1,000 games).

## 10. Verification state

| Claim | State |
| --- | --- |
| Briefs 08, 09, 11: numbers against artifacts, code diffs, covering tests, `ruff`, digests before and after each code merge | [verified] in review on 2026-11-16 (ledger sections "review of brief 08", "09", "11 and gate G1") |
| Brief 09's permutation test on replay holdout rows; its trainings and play smoke | not re-run (no replay data in the review container); the worker's claim |
| Gate G1 rows | [verified]: `docs/briefs/11-artifacts/gate/merged.json`; MaxDamage row played on two machines |
| `search_distill`: stored rows reproduce live log-probabilities; targets valid; resume | [verified] by `tests/test_search_distill.py` on pinned Showdown |
| Distillation rounds 0 to 2 and networks 0 to 2 | [verified]: 1,008 games a cell, one training seed each; play JSONs in `docs/briefs/15-artifacts/` |
| Hosted GPU training equals CPU training | [verified] once: network 1 retrained on a T4, held-out cross-entropy 1.8286 against 1.829, top-1 56.6% against 56.7% |
| Hosted CPU collection gives labels of the same quality | see the ledger; not verified if no row compares simulations per label and label-check agreement between machines |
| Full test suite | last run 2026-11-14 (1,017 passed, 3 skipped, 3 expected failures, 1 failed); reviews since ran covering files only |
| Belief-loop fix and `skip_unweighted_belief` at scale or on a GPU | not verified (brief 13 uses the skip on a GPU) |
| Replay models at scale on corrected rows | not run (brief 13) |
| Whether lead training damages the policy | not measured (brief 13) |
| Rust engine vs Showdown on terastallized Pokémon; Tera Stellar (not modelled); engine averaged damage roll; the Encore panic | not verified or not fixed; worlds that panic are dropped (0.17% of sampled worlds in the gate) |
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
4. GPU time: 27.5 of 45 hours were left on 2026-11-16 (the earlier 3.69 was a misreading), reset on
   2026-11-21; brief 13 may use 20. Ladder account and team choices; where long CPU jobs run
   (`ROADMAP.md` section 8).
