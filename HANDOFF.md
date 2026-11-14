# Handoff

Written 2026-11-14 for a fresh coding session. It replaces the 2026-11-11 handoff (still in git
history); everything in that file that is still true is carried over below. Read this, then
`AGENTS.md`, `ROADMAP.md`, and the newest sections of `docs/results-ledger.md`
(`## 2026-11-14: ...`). Facts are marked **[verified]** (measured, or read directly in the code,
with the evidence named) or **[assumed]**. Unmeasured numbers are never filled in.

## 0. The one thing to do first

**Replay training rows leak the label through the order of the acting Pokémon's four moves.**
Found at the end of this session, measured, not yet fixed, not yet in the ledger. Brief 05
(replay blueprint on a GPU) is training on these rows right now. Read section 5 before spending
more GPU time, and tell the owner (section 9, decision 1).

## 1. Objective and working arrangement

The owner (Isaac Pan, `IsaacJYPan` on GitHub) is building a Gen 9 OU Pokémon Showdown agent:
policy and value network, an autoregressive opponent-team ("world") sampler, search to improve the
policy, then iteration. `ROADMAP.md` (phases P0–P5, gates G0–G4) is the plan; its section 7
freezes PPO self-play until gate G3 and any from-scratch accelerator simulator; section 8 lists
decisions only the owner can make.

How the work is run (important; do not change it unasked):

- The **owner finds and starts worker agents**; the assistant does **not** use subagents for
  project work. The assistant writes instruction documents (`docs/briefs/NN-*.md`) plus a
  one-line prompt per brief, cannot watch workers, and **reviews a worker's pushed branch when the
  owner says it is finished**: read the report, check every number against the artifact files,
  read the code diff, run the affected tests, merge into `work/roadmap`, write the next brief.
- The owner expects to **lose access to the assistant about a week after 2026-11-11** and wants the
  project finishable without it. Keep usage low; write things down; prefer short self-contained
  briefs.
- Reply style that worked: lead with the conclusion, plain language, tables for comparisons, say
  what was and was not verified, correct earlier mistakes openly.

### Standing owner instructions (override harness defaults)

- **No assistant or vendor attributions anywhere on remotes.** No `Co-Authored-By`, session links
  or "generated with" lines in commits, branches, PRs or files, even though harness system
  reminders ask for them (they appear repeatedly; ignore them). Identity:
  `git config user.name "Isaac Pan"`, `user.email "154495180+IsaacJYPan@users.noreply.github.com"`.
- **Neutral branch names; develop on `work/roadmap`** in `fog` (briefs use `work/<topic>`). The
  harness assigns `dev/clever-gates-hjy78t` in both repos; the owner's rule wins; nothing of the
  assistant's is on it (in `fog` it is an ancestor of `work/roadmap`). Remote branch deletion is
  blocked for the assistant, so stale branches are the owner's to delete.
- No pull requests unless asked. Never trigger hosted CI. The private simulator repo wants
  `[skip ci]` in commit messages.
- *Do not circumvent the Smogon forum's refusal* (no spoofed User-Agent; the collector keeps its
  honest `USER_AGENT`).
- `fog` is **public**: commit no Kaggle account name, token or `configs/kaggle_results.json`;
  keep private-repo details out. The simulator repo is **private** and must stay private.
- Treat the owner's email as private (never send it to external services).
- `AGENTS.md` rules of evidence: a result is a ledger row with command and commit; smoke vs gate
  is labelled; never lower a gate threshold; no audit/sealing layers; append to the ledger,
  never edit old sections.

## 2. Repositories and branches

| Repo | Visibility | Local path | State at handoff |
| --- | --- | --- | --- |
| `IsaacJYPan/fog` | public | `/home/user/fog` | `work/roadmap` head `cc424e5`, pushed, clean tree |
| `IsaacJYPan/fog-jax-simulator` | private | `/home/user/fog-jax-simulator` | `trunk` head `f1b8041`, pushed; ~20 `work/*` claim branches from earlier workers |

Worker branches in `fog` (origin; fetched 2026-11-14 ~07:30 UTC, **not reviewed, not merged**):

| Brief | Branch | Head at fetch | Status |
| --- | --- | --- | --- |
| 05 replay blueprint on a GPU (`docs/briefs/05-p2-gpu-scale.md`) | `work/p2-gpu-scale` | `8dc99a2` | G20 job running, 5 of 18 epochs done (~1,924 s/epoch), projected end ~14:09 UTC; owner priority: finish G20, then brief 06. Branch has `docs/briefs/05-report.md` (draft), `05-artifacts/`, `tests/test_replay_gpu_job.py`, `--prepare-only` work |
| 06 collection on a GPU (`docs/briefs/06-collection-on-gpu.md`) | `work/collect-gpu` | `77f428a` | timing hooks and job script prepared; no GPU result yet |

One GPU job at a time is allowed on the owner's account. Prompt to start a brief (06 as example):
`In IsaacJYPan/fog, check out work/roadmap, read AGENTS.md, then carry out docs/briefs/06-collection-on-gpu.md.`

## 3. What is complete [verified unless marked]

Reviewed and merged into `work/roadmap` (ledger sections carry commands, commits, intervals):

| Item | Where | Result |
| --- | --- | --- |
| P0.1–P0.4, P0.6, P1.1, P1.2, P1.2a–d | earlier commits, ledger | see old handoff table in git history; P1.2c/d boxes are ticked in `ROADMAP.md` |
| Brief 01–03: replay learning curve, scale prep, CPU profiling and R/A/S/B runs | `docs/briefs/01..03-report.md`, ledger "review of brief 03" | policy limited by data; best epoch 8 of 18 then overfits; 3,000-game model NLL 1.579, top-1 0.441, play smoke 0.360 vs SimpleHeuristics and 0.7725 vs MaxDamage (400 games, `pool_b`); value head memorises: Brier 0.3445 (constant scores 0.25), ECE 0.299 |
| **P0.5 collection speed** (my own work, `work/fast-collect`, merged) | ledger "faster rollout collection on headless Showdown" | 2–2.5× on CPU with **bit-identical rollouts**: 451 learner decisions/s self-play, 647 vs MaxDamage, 4 cores, no GPU (179/324 before). `env.rollout_batch`, `env.actor_device` added. **GPU acting path never run on a GPU** (P0.5 box deliberately unticked) |
| Brief 04: limited learning-from-play check | `docs/briefs/04-report.md`, ledger "review of brief 04" | see section 4 |
| `fog/eval/replay_slot_order_check.py` + test | commit `cc424e5` | the leak diagnostic (section 5) |

Rollout reference digests (`python -m fog.eval.showdown_rollout_probe`): `--batch 32 --games 64
--threads 2` self-play on `pool_b` = `0efd4319547a4b3b`; `--games 6` sequential =
`23405989f4665988`; `--batch 32 --games 96 --opponent MaxDamage --pool data/teams/pool_a` =
`338cfca82f31b181`. Any encoder/collection change must reproduce them.

## 4. Brief 04 findings and what they mean [verified; smoke, one seed, budgets scaled to 0.557]

- Random-weight model on one team vs MaxDamage: 0.160 → 1.000 by 106,889 decisions; 0.920 vs
  SimpleHeuristics (never trained against). The model and the PPO loop learn.
- Blueprint (1,000-game replay model) on 8 `pool_a` teams: vs SimpleHeuristics 0.255 → 0.9125
  (peak 0.9675) on those teams; **no transfer** to 16 unseen `pool_b` teams (0.220 → 0.215; vs
  MaxDamage 0.565 → 0.560).
- Value head learns from play (explained variance 0.5–0.6). So the replay value problem is the
  data/labels, not the architecture.
- CPU PPO update takes 83–87% of wall time; 22–43 decisions/s end to end; KL early-stop gave 1.39
  effective epochs of 3.
- **Reading:** team variety, not game count, decides whether self-play is worth scaling. The wide
  pool is the public team bank (P0.7), an owner action. Brief 04's runs were not re-run by the
  assistant; win rates were cross-checked against its evaluation JSONs.

## 5. Replay-row leak and other train/live mismatches (this session's last, unfinished work)

### 5.1 Move-order leak [measured]

`complete_private_team` (`fog/training/replay_policy_data.py`) fills the acting player's moveset
with the moves revealed during the whole game **in order of first use**, then pads with sampled
moves. `apply_private_set` installs them in that order, so slot position encodes the label.

- 150 local replays (`data/metamon_gen9ou_r1500`, 8,485 kept rows): labelled-move slot share
  45.8 / 31.3 / 17.1 / 5.8%; on decisions where the acting Pokémon had not moved yet (23.5%) the
  label is slot 0 in 99.8%; 99.85% of labels lie in revealed (non-sampled) slots.
- 900-game conversion (1 imputation, 42,391 train / 6,673 holdout rows) trained 5 epochs on CPU
  (`replay_blueprint_cpu.yaml`, batch 256): holdout NLL 2.266 → 1.920, top-1 0.101 → 0.290.
  `replay_slot_order_check` on its best checkpoint, rows scored as stored vs with the acting
  Pokémon's four slots shuffled (ids, PP, legality, label, provenance together):

  | Rows | top-1 stored → shuffled | NLL stored → shuffled |
  | --- | --- | --- |
  | all 6,668 | 0.290 → 0.221 | 1.920 → 2.041 |
  | move rows 4,783 | 0.357 → 0.259 | 1.702 → 1.871 |
  | not-yet-moved 1,123 (**clean test**) | 0.573 → 0.303 | 1.404 → 1.831 |

  After epoch 1 there was no dependence (0.2029 vs 0.2002), so the model learns the shortcut once
  it can. Caveats: tiny 900-game model, one seed; on rows where the Pokémon already moved, the
  shuffle also breaks the history's past-action slot one-hots, so only the not-yet-moved row is a
  clean comparison. Numbers are in the scratchpad (`.../scratchpad/arch/slot_e4.json`, which
  will not survive); re-run to regenerate.
- The untrained model is exactly shuffle-invariant (zero-initialised output layers), so the
  invariance is not architectural.
- **Why it matters.** Live play lists moves in team-sheet order, so a policy that leans on slot 0
  is wrong live; holdout accuracy is inflated by it. This may also feed the overfit-after-epoch-8
  and value-memorisation stories and brief 03's play numbers (the effect on play is **unknown**).
  Every point of brief 05's learning curve shares the leak, so the curve's *shape* may still
  mean something but its absolute numbers are inflated.
- **Fix (not started):** permute the four move columns per Pokémon (stable per-replay seed)
  inside `complete_private_team` before `apply_private_set`; downstream slot indices
  (`_PerspectiveState.record_action`, tracker `my_actions`) then stay consistent because they read
  the installed order. Consider also random slot permutation as training augmentation, because
  `ARCHITECTURE.md`'s claim "permuting candidates permutes their logits" holds only for the
  candidate scorer, not for the trunk (the entity encoder concatenates the four move embeddings in
  order [assumed, consistent with the result above]). Verify with the check: label share by slot
  ≈ 25% each, `not_yet_moved_label_is_slot0` ≈ 0.25, shuffled ≈ stored.
- Related, smaller: sampled padding moves are never labels and the model can partly tell them
  apart by being "typical for the species"; unused real moves are replaced by usage samples
  (selection effect) [assumed, unmeasured].

### 5.2 Other replay-versus-live differences

| Difference | Evidence |
| --- | --- |
| Own PP is always 0 in replay rows (live: real fractions) | largest own PP over 42,391 rows = 0.0 [measured] |
| Forced-replacement decisions are absent from replay training (policy untrained there) | 2 switch-only rows of 42,391 [measured] |
| Live: a replacement choice in turn *t* overwrites `tracker.my_actions[t]` (the move chosen earlier that turn); replays keep only the first action | `record_history_decision` in `fog/encoding/events.py` called per decision in `fog/env/player.py`; replay `record_action` returns after the first [read in code] |
| At a replacement decision the current turn's events (including the faint) are not in the history (`encode_history` uses turns `1 <= t < current`); the board shows the result | `fog/encoding/history.py` [read in code] |
| Lead choice uses `TeamPreviewHead`, which never trains in replay (zero gradient) and has `preview_coef: 0.0` in the typed-token and self-play configs, so leads are essentially random | gradient test; `grep preview_coef configs/` [measured] |
| Own and opponent slot order are asymmetric; three position embeddings are added per token (`token_position`, `token_kind`, `token_side`, `party_position`) and no training step shuffles team order | `fog/models/shared_latent.py` [read] |

### 5.3 Architecture review: what was checked and what holds

Verified by reading `fog/models/{agent,shared_latent,heads,three_encoder}.py`,
`fog/training/{ppo,rollout,losses,bc}.py`, `fog/env/actions.py`, `fog/encoding/*`:

- **Model (current):** `shared_latent_trunk` with 24 tokens × 128 (12 Pokémon, field, control with
  6-way request phase, 8 recent events, long-history summary, global), 4 pre-norm attention layers,
  4 heads, ff 512, dropout 0.05. Policy = 14 candidates (4 moves, 4 Tera moves, 5 switches,
  forced) scored by an attention query over the 24 tokens with the candidate's own move/switch
  descriptor; masked illegal logits (`-inf` handled safely in PPO via `_LOG_ZERO`). Value =
  `shared_to_value` reader + MLP + `tanh`, MSE to outcome. 5,853,084 parameters.
- **Information isolation holds** in what I read: policy and value see only own private sheet,
  public board and public history; opponent truth enters only as belief targets and oracle modes
  (off). `use_action_features` is off in the typed-token configs, so the rule-decision columns
  (incl. `simple_rule_pick`) are not model inputs. I did **not** re-run the design doc's
  information-isolation tests.
- **Parameters with no gradient in replay training: 2,518,118 of 5,853,084** (measured on one
  batch): self-belief head 1.29M (it does train in self-play), the older three-encoder backbone
  (`current_state_encoder`, `private_state_encoder`, `public_fusion`, `preview_context`,
  `private_to_belief` ≈ 0.79M, dead in the typed-token config), `policy_value.policy` and
  `conditional_action_scorer` and `shared_to_policy` (~0.22M, unused because the action-query
  path replaces them), `preview`. Not a bug; dead weight and a trap for anyone reading parameter
  counts. About 2.1M of 5.85M parameters serve policy and value.
- Moves, items and abilities are bare ID embeddings; no type/power/category features enter the
  trunk (they exist only in `action_features`, which are off). The type chart must be learned per
  move, and moves outside the training teams learn nothing. Consistent with brief 04's
  no-transfer result but **not tested as the cause**.
- **PPO update:** clipped surrogate, GAE (γ=1, λ=0.95), advantages normalised over the whole
  batch, value MSE to returns, entropy bonus, KL early stop. **The KL stop aborts all remaining
  minibatches and epochs once any minibatch exceeds `target_kl`**, which also halts value
  training (1.39 of 3 effective epochs in brief 04). Behaviour-policy (temperature-corrected)
  log-probs are used for the ratio, correct.
- **Not verified:** how truncated games (`max_turns`, `truncated=True` in
  `showdown_batch_runner.py`) are scored in PPO (bootstrapping vs terminal zero); which seats are
  recorded in self-play; whether value targets from both perspectives of one replay game (opposite
  labels, correlated rows) bias the holdout split (splits are by game, so probably fine).

## 6. Decisions made and why (do not reopen without new evidence)

**Optimise official Showdown instead of building a GPU simulator.** The owner asked whether JAX
simulators are faster and whether the bridge could support 10–100M self-play games. Findings:
the Node bridge costs ~0.8–1.0 ms/step (≈ raw Showdown); the old loss was FOG's Python (parsing
and encoding), now cut 2–2.5×; the simulator experiment (S6, private repo) came out **below the
bar by a wide margin** and a rewrite was estimated at 15–30 worker sessions. Recommendation
(owner agreed in effect): **stop the simulator.** Reopening condition recorded in the private
repo's `TASKS.md`: a FOG rollout profile showing the simulator step above half of rollout time
after the network runs on a GPU. Numbers are in the private repo (`docs/briefs/S6-report.md`,
`TASKS.md`); keep them out of this public repo. PokeJAX (arXiv 2603.12145) claims very high step
rates, its code is not available (owner confirmed), so it cannot be compared.

**Honest corrections already given to the owner:** "four cores of Showdown match a T4" compared
raw engines only; FOG's wrapper delivered ~300 learner decisions/s with a free forward.

**Order of work (unchanged):** measure → search-only agent → human-replay policy → distil search →
learned parts inside search → iterate; PPO self-play last behind G3. Search design (strategy
fusion F1/F2/F3, stale search targets, policy-independent beliefs until G4, play `π·exp(Q/τ)`)
is in `docs/search-and-belief-design.md`; it is canonical.

**End-to-end speed estimates given to the owner** [assumed, 40 learner decisions/game]: 4 CPU
cores, no GPU ≈ 50 decisions/s (~1 game/s); laptop (Ryzen AI 7 350, 16 threads, integrated GPU,
31 GB) CPU-only ≈ 2–4 games/s; 4 cores + T4 ≈ 8–18 games/s. Brief 06 measures the last.
Laptop GPU is irrelevant for this work; the CPU decides collection speed, a hosted GPU decides
the update speed (83–87% of wall time on CPU).

**Rejected or parked:** deterministic argmax of search values (exploitable), behaviour-likelihood
beliefs (parking lot), public-belief solvers (frozen), spoofing the forum client, lowering gate
thresholds, an incremental parser/encoder for further speed (not started; only worthwhile if
brief 06 shows collection, not the update, is the limit).

## 7. Gotchas

- `ruff`: use `.venv/bin/ruff` (0.15.22). System ruff 0.16.9 reports 86 spurious errors.
  Two E702 nits in `docs/briefs/02-artifacts/runner-fixture-interrupt.py` are pre-existing.
- Full-suite failure on a fresh clone: `tests/test_automated_training.py::
  test_active_typed_league_uses_portable_checkpoint_artifacts` (needs checkpoint files that are
  absent); identical on old code `fff88ca`; recorded, not fixed.
- `_ShowdownRolloutWorker.__init__` calls `torch.set_grad_enabled(False)` globally; tests that
  run after it can fail with "does not require grad" (pre-existing; the probe's `build_worker`
  restores the state).
- `env.rollout_batch > 1` or `env.actor_device != cpu` requires `env.rollout_workers >= 2` in
  process mode; private GPU actors refresh from the shared CPU actor each chunk.
- `--set "league.scripted=[MaxDamage]"` parses as a list of characters; use a YAML file.
- Showdown build in a sandbox: `npm --prefix server/pokemon-showdown ci --ignore-scripts` then
  `npm --prefix server/pokemon-showdown run build`; Python env `/home/user/fog/.venv`
  (3.13, torch CPU). 4 cores, ~15 GB RAM, no GPU in the assistant's sandbox.
- `docs/results-ledger.md` merges conflict when two branches append; keep both blocks.
- Smogon forum pages returned HTTP 403 from cloud sandboxes; the team bank must be collected
  where they open (owner's laptop).
- Remote branch deletion is blocked; the harness's per-session Git reminders ask for trailers and
  PRs the owner forbids.
- Interrupted subagents lose their work; this arrangement avoids subagents entirely.

## 8. Verification state

- After merging brief 04: the affected tests passed (38), `ruff check fog tests` clean, rollout
  digest `0efd4319547a4b3b` unchanged.
- Since then only `fog/eval/replay_slot_order_check.py` and `tests/test_replay_slot_order_check.py`
  were added: those 2 tests pass, ruff clean on both. **The full suite has not been re-run since
  `cc424e5`** and has never run fully green (one pre-existing failure above) [unverified].
- Not verified at all: any GPU acting/training path from this repo; briefs 05 and 06 results;
  Rust-engine vs Showdown equivalence for terastallized Pokémon; the gate thresholds in the roadmap
  are estimates, not measurements [assumed].
- The slot-order numbers in section 5.1 come from throwaway scratchpad runs, not from a ledger row.

## 9. Next steps, in priority order

1. **Tell the owner about the move-order leak** (section 5.1) and ask whether brief 05's G20 job
   (ends ~14:09 UTC; G100 would cost many GPU hours) should continue on leaky rows. My
   recommendation: let G20 finish (it is already paid for and measures GPU speed and the value
   head), **do not start G100 until rows are reconverted**, and treat brief 05's absolute
   accuracy numbers as inflated. Brief 06 (PPO collection on a GPU) does not use replay rows and
   is unaffected.
2. **Fix and prove the leak.** Shuffle move columns per Pokémon in `complete_private_team`;
   add a unit test (distribution of the label's slot is uniform; round-trip with
   `apply_private_set`); reconvert 900 games; retrain the same 5 epochs; run
   `python -m fog.eval.replay_slot_order_check --config configs/replay_blueprint_cpu.yaml
   --manifest <holdout_manifest.parquet> --checkpoint <bc_best.pt> --out <file>` before and after;
   write one ledger section with both (the commands are in section 5.1; the conversion command is
   in `docs/metamon-gen9ou.md`). Decide on slot-permutation augmentation.
3. **Record this session's architecture review in the ledger** (section 5 content, labelled
   as diagnostics) and add the open mismatches to the roadmap's parking lot or as P2 sub-tasks:
   PP = 0 in replays, replacement decisions, history overwrite, current-turn events at
   replacements, untrained preview head. Fix cheap ones in the converter/player only with an
   equivalence or distribution test (AGENTS: behaviour-preserving, say how checked).
4. **Review brief 06** when the owner says it is finished: `docs/briefs/06-report.md`,
   `06-artifacts/`; check the `cuda` probe rows ran, GPU memory, stage 2 learning curve vs brief
   04's CPU numbers (0.160 → 1.000 vs MaxDamage by 106,889 decisions), then merge and, if the GPU
   forward is shown to work, tick P0.5.
5. **Review brief 05** when finished: `docs/briefs/05-report.md`; apply the section-6 value
   reading rule from the brief (Brier < 0.24 at 100k and late-third ≥ 0.02 better than early),
   P2.4's decision rule, check GPU-hours ≤ 20, no account name or token in the repo, and read all
   of it in light of the leak.
6. **Owner action: collect the public team bank** (P0.7) on a machine where the forum pages open:
   `python -m fog.scripts.collect_team_bank --target 150 --min-quality 70` then
   `python -m fog.scripts.build_public_team_splits --bank data/teams/public_gen9ou --out
   data/teams/gen9ou_public_v1` (from `KAGGLE_TRAINING.md`; split output is git-ignored; ask the
   owner whether the three split files should be tracked so cloud workers can use them). Not run
   by the assistant. Needed for held-out evaluation, for G0, and for self-play to generalise.
7. Then the roadmap in order: finish P0.7/G0, P1.3 per-game fidelity counters, P1.4 budget curve,
   G1 (EngineMCTS ≥ 85% vs SimpleHeuristics and ≥ 95% vs MaxDamage, ≥ 1,000 matched games,
   fallbacks < 5%); P2 gate G2 (> 55% vs A3, ≥ 65% vs SimpleHeuristics, value ECE < 0.05 — the
   current value head fails this by a wide margin, ECE 0.299 on 3,000 games); P3 distillation;
   P4 learned components in search (not before G1 and G2); P5. PPO fine-tuning arm only after G3.
8. Housekeeping the owner must do: delete stale remote branches (`dev/clever-gates-hjy78t` in both
   repos, finished `work/*` branches); decide whether to commit this file (it is already tracked
   in `fog`; it contains no private simulator numbers).

## 10. Open questions and owner decisions

1. Continue brief 05 on leaky rows? (section 9.1.)
2. Collect the team bank on the owner's laptop? Track the split files in git?
3. Ladder account and team choices (roadmap section 8); where long CPU jobs run; whether to
   approve further hosted-GPU budget after briefs 05/06.
4. Unmeasured: does the leak depress or inflate live play? Do type/power features in the trunk fix
   unseen-team transfer? Is F2 (fusion below the root) costing strength (oracle-gap diagnostic)?
   Does the learned belief sampler beat the usage sampler on exact-truth logs (P4.1)? Does the Rust
   engine's averaged damage roll cost strength vs Showdown (`fog/eval/chance_fidelity.py`)?
5. Whether GPU acting (`env.actor_device: cuda`) reaches the ~900–1,200 learner decisions/s
   estimated for 4 cores + GPU (estimate, not measured).
