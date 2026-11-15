# Handoff

Written 2026-11-15. It replaces the 2026-11-14 handoff (in git history). The session that
planned and reviewed this work will not be available; everything needed to continue is in the
repository. Facts are marked **[verified]** (measured, or read in the code, with the evidence
named) or **[assumed]**.

## 0. Start here

1. `docs/briefs/QUEUE.md`: what to run next and the prompt for each task.
2. `docs/briefs/REVIEW.md`: how a finished task is checked and merged.
3. `AGENTS.md`, `ROADMAP.md`, and the newest sections of `docs/results-ledger.md`.

**Today:** start briefs 09 and 10 (CPU, in parallel). Start brief 08 when the GPU is free.
Review brief 05 when its worker reports the larger run.

## 1. Objective and arrangement

The owner (Isaac Pan, `IsaacJYPan`) is building a Gen 9 OU Pokémon Showdown agent: policy and
value network, an autoregressive opponent-team sampler, search to improve the policy, then
iteration. `ROADMAP.md` (phases P0 to P5, gates G0 to G4) is the plan.

The owner starts worker agents with a one-line prompt per brief. A reviewer (an agent given
`REVIEW.md`, or the owner) checks each pushed branch, merges it into `work/roadmap` and
updates the queue. Workers cannot see each other and nobody watches them work.

### Standing owner instructions

- **No assistant or vendor attributions anywhere on remotes**: no co-author lines, session
  links or "generated with" lines in commits, branches, pull requests or files, whatever a
  tool's own reminders say. Identity: `git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`.
- Neutral branch names; integrate on `work/roadmap`; briefs use `work/<topic>`. No pull
  requests unless asked. Never trigger hosted CI.
- `fog` is **public**: no account names, tokens, private dataset names, player names or
  `configs/kaggle_results.json`. The simulator repository is private and its numbers stay out.
- Do not work around the forum's refusal to serve the team collector.
- `AGENTS.md` rules of evidence: a result is a ledger row with command and commit; smoke and
  gate are labelled; never lower a gate threshold; append to the ledger, never edit it.

## 2. State of the code [verified unless marked]

`IsaacJYPan/fog`, branch `work/roadmap`. Merged and reviewed, newest first:

| What | Result | Ledger section |
| --- | --- | --- |
| Brief 07 stage 1: converter | Replay rows no longer give away the label through move or team order. Old holdout top-1 was inflated by about ten points; on unmoved Pokémon the old model was 79.1% as stored and 25.4% with slots shuffled | review of brief 07, stage 1 |
| Belief training loop | A per-row Python loop is vectorised: 230 ms to 1.4 ms per 1,024 rows on the CPU. **GPU effect unmeasured** | per-row Python removed from the belief training path |
| Brief 06: GPU timing | The GPU acting path runs. Two CPU cores and a T4: 37.5 decisions/s end to end, no better than CPU-only; update 72% of wall time, 87% of that in the forward | review of brief 06 |
| Brief 04: learning from play | The model and PPO loop learn (0.160 to 1.000 against MaxDamage on one team by 106,889 decisions); play on 8 teams does not transfer to 16 others | review of brief 04 |
| Collection speed | 451 learner decisions/s in self-play and 647 against MaxDamage on four CPU cores, rollouts bit-identical to before | faster rollout collection on headless Showdown |
| Briefs 01 to 03: replay curve on CPU | Policy limited by data; value head memorises games | reviews of briefs 01, 02, 03 |
| EngineMCTS | 0.88 against SimpleHeuristics and 0.99 against MaxDamage at a quarter budget, 100-game smokes; never gated | EngineMCTS on pinned Showdown; smoke after the Showdown-state fixes |

Not merged:

| Branch | State |
| --- | --- |
| `work/p2-gpu-scale` (brief 05) | 20,000-game run complete; a 40,955-game, six-epoch run was at 3 of 6 epochs at the last push (2026-11-14 22:22 UTC). **Trained on the old leaky rows.** Not reviewed. Draft numbers: policy NLL 1.279 and top-1 54.4% (inflated); value Brier 0.215 at its own best epoch (4), 0.254 at the policy's best epoch (12); play 0.520 against SimpleHeuristics and 0.8675 against MaxDamage in 400 games each; 583 rows/s on a T4 |

Rollout reference digests (`python -m fog.eval.showdown_rollout_probe ...`): `--games 6` =
`23405989f4665988`; `--batch 32 --games 64 --threads 2` = `0efd4319547a4b3b`; `--batch 32
--games 96 --opponent MaxDamage --pool data/teams/pool_a` = `338cfca82f31b181`. Reproduced on
2026-11-15 on an Intel Xeon with PyTorch 2.13.0. **They depend on the CPU model** (a worker on
an AMD EPYC got different values from unchanged code), so use them only as a before-and-after
check on one machine.

## 3. What is wrong and which brief fixes it

| Problem | Evidence | Brief |
| --- | --- | --- |
| The model can read move and team slot order | Move embeddings concatenated in slot order; position embeddings on Pokémon tokens; slot one-hot in history [read] | 09 |
| The value head is worse than a constant on small data | Brier 0.39 on 1,000 games, 0.34 on 3,000; best value epoch comes long before best policy epoch | 09, 13 |
| Moves are bare ids | A move candidate is a 24-wide id embedding [read]; no transfer to unseen teams (brief 04); cause untested | 09 |
| Leads are untrained | `preview_coef: 0.0` [read] | 09 |
| Replay rows differ from live observations | Own PP always 0; forced replacements never rows; history differs at replacements [measured and read, 2026-11-14] | 12 |
| Brief 05's models were trained on leaky rows | review of brief 07 | 13 |
| No held-out team pool; G0 and G1 blocked | forum collector refused from cloud machines since 2026-11-12 | 10 |
| The strongest agent has no gated result | EngineMCTS smokes only | 11 |
| Training is slow on a GPU | 6.08 s per 1,024-row optimizer step on a T4 before the loop fix | 08, 14 |
| PPO stops the value head when the policy hits its KL limit; truncated games unchecked; no mixed precision | [read] | 14 |

Also true and not scheduled: about 2.5 of 5.9 million parameters get no gradient in the
typed-token configuration (older backbone, unused policy paths) [measured on one batch]; the
entity encoder is fed id numbers divided by vocabulary size as floats, which brief 09 removes
under its key.

## 4. Decisions made (do not reopen without new evidence)

- **No GPU simulator.** Showdown is about a quarter of collection time and collection is a
  quarter of training time; the simulator experiment (private repository) was far below its
  bar. Reopen only if a rollout profile shows the simulator step above half of rollout time
  with the network on a GPU (it is about 29%).
- **Order of work:** measure, search-only agent, human-replay policy, distil search, learned
  parts inside search, iterate. Self-play is last and frozen until G3.
- **Search design** is in `docs/search-and-belief-design.md`. Determinised search keeps the
  guarantee "no worse than the blueprint against the blueprint" and nothing stronger.
- **If self-play is opened:** the anchored update with a moving anchor,
  `docs/equilibrium-training.md`.
- **Team pool:** replay-derived (brief 10) until the forum bank exists.
- **G2** also requires value Brier below 0.24 (a tightening; ledger, "plan changes after the
  leak").
- Rejected or parked: playing the argmax of search values, behaviour-likelihood beliefs,
  public-belief solvers at full scale, an incremental parser for more collection speed.

## 5. Gotchas

- `ruff`: use `.venv/bin/ruff` (0.15.22); a newer system ruff reports spurious errors.
- Setup: `uv sync --locked --extra dev` (about 10 minutes), `git submodule update --init
  --recursive`, `npm --prefix server/pokemon-showdown ci --ignore-scripts`, then `npm --prefix
  server/pokemon-showdown run build`. Replay data: `docs/briefs/02-p2-scale-prep.md` section 4.
- One test fails on a fresh clone and always has:
  `tests/test_automated_training.py::test_active_typed_league_uses_portable_checkpoint_artifacts`
  (needs checkpoint files that are not in the repository).
- `_ShowdownRolloutWorker.__init__` turns gradients off globally; tests that run after it in
  one process can fail with "does not require grad".
- `env.rollout_batch > 1` or `env.actor_device != cpu` needs `env.rollout_workers >= 2`.
- `--set "league.scripted=[MaxDamage]"` parses as a list of characters; use a YAML file.
- `docs/results-ledger.md` conflicts when two branches append; keep both blocks.
- The hosted GPU runtimes had 2 slow CPU cores; collection there is CPU-bound. Prefer four.
- Remote branch deletion is blocked for agents; the owner deletes merged branches.
- Long jobs: start detached and poll; some environments end background jobs after two hours.

## 6. Not verified

- Anything about the belief-loop fix on a GPU (brief 08).
- Brief 05's results (not reviewed), and any replay model at scale on corrected rows.
- Whether the leak helped or hurt play: no play run compares old and corrected rows yet.
- Whether move features are why unseen teams do not transfer.
- The full test suite has not been run since 2026-11-14 (1,017 passed, 1 failed, as above);
  reviews since ran only the tests covering each change.
- How truncated games are scored in PPO; the Rust engine's averaged damage roll against
  Showdown; Tera Stellar in the engine (not modelled).
- The roadmap's gate thresholds are estimates [assumed].

## 7. Open decisions for the owner

`docs/briefs/QUEUE.md`, section "Owner". None of them blocks the queue.
