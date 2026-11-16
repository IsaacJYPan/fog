# Brief 15: teach the network to play like the search (P3, generation 0, pilot)

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

The search-only agent, EngineMCTS, is by far the strongest thing in the repository: 92.9%
against SimpleHeuristics over 1,008 games on held-out teams. The best network trained on human
replays scores about 52% against the same opponent, and the corrected 1,000-game model 38.5%
(ledger: "review of brief 05", "review of brief 09", and the review of brief 11 for the gate).
The roadmap's next phase, P3, closes that gap the standard way: let the network play, ask the
search what it would have done in every position the network reaches, train the network on
those answers, repeat. It needs no GPU, and the search is cheap enough: at 200 ms and eight
sampled worlds it scored 91.0% over 400 games at 0.26 s a decision, against 91.75% at 800 ms
(brief 11's budget curve; intervals overlap).

This brief is a **pilot**: three rounds, about 180,000 labelled decisions, random initial
weights. It answers one question: does a network trained this way, playing without search,
beat the scripted bots on teams it never trained on, and how fast does it improve with data?
The full run (one million decisions, starting from the replay blueprint) is written after
this and brief 13 report.

**Start only if `docs/briefs/QUEUE.md` says gate G1 passed.**

## 2. Rules

- Branch **`work/distil-g0`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md`, then `docs/search-and-belief-design.md` (section 7, information
  isolation). CPU only. Setup is in `HANDOFF.md` section 9; no replay data is needed.
- **Teams.** Generate only on `data/teams/replay_v1/train`. Evaluate on
  `data/teams/replay_v1/validation`. Never read the test split.
- **Budget: 60 core-hours** in total, counted as cores times wall time while jobs run. Keep a
  running total in the artifacts. Every long job writes its output in shards and resumes.
- Do not change search code (`fog/search/`, `fog/env/state_recon.py`) or the model
  (`fog/models/`). New behaviour in shared training code sits behind a key that defaults to
  off. The three rollout digests (`HANDOFF.md` section 9) are equal before and after on your
  machine.
- The teacher sees what a player sees. It runs `ShowdownEngineMCTS.choose` on the acting
  side's own battle object and tracker, exactly as it does when it plays. No true opponent
  team, and nothing derived from one, is stored as a model input.
- Every number is smoke unless it has at least 1,000 matched games.

## 3. Stage 1: collection (P3.1)

Write `fog/training/search_distill.py` with a `collect` command. Reuse what exists rather
than copying it: `fog/training/dagger_heuristic_distill.py::collect_rows` already plays the
network on pinned Showdown in worker processes and asks a teacher for a label at every
decision through `FogPlayer.supervised_label_provider`; `FogPlayer.action_override` can replace
the executed action; `ShowdownEngineMCTS` leaves its answer in `last_policy` (visit share per
action, 14 numbers) and `last_value` (0 to 1).

At every decision of the recorded side, store:

- the observation exactly as the network received it (the same arrays the rollout buffer
  keeps), the legality mask, and the action features if the model configuration reads them;
- the teacher's visit distribution over the 14 actions (zero on illegal actions, sums to
  one), its value mapped to -1..1, and a flag when the teacher had one legal option or fell
  back (those rows carry no policy target);
- the action executed, whether the teacher or the network chose it, the network's own
  log-probabilities, the game id, team indices, opponent kind, turn, and at the end of the
  game the outcome.

Acting rule: with probability `beta` the teacher's best action is executed, otherwise the
network's (sampled at temperature 1 from the masked policy, so play is varied). The teacher
labels every decision either way.

Opponents: half the games against a copy of the same network under the same acting rule
(not recorded), a quarter against SimpleHeuristics, a quarter against MaxDamage. Teams: both
sides drawn from the train split with a seeded schedule; sides alternate.

Teacher: `k=8`, `budget_ms=200`, `threads=1`. One collection process per core, and nothing
else running, because the budget is wall-clock.

Tests, each small enough for the normal suite:

1. A stored row replayed through the network reproduces the log-probabilities recorded live
   (to 1e-5) on at least 200 decisions from real games.
2. The stored target sums to one, is zero on illegal actions, and its largest entry is the
   action the teacher would have played.
3. With `beta=1` the executed action always equals the teacher's; with `beta=0` the teacher
   never changes what the network plays (same seed, same game with and without the teacher).
4. Collection resumes after an interruption without duplicating or losing a game.

**Before collecting at scale, check the labels.** On 1,000 stored decisions from at least 50
games, re-run the teacher at the default budget (`k=8`, `budget_ms=1600`) and at 200 ms with a
second seed. Report top-1 agreement of 200 ms with 1,600 ms, and of 200 ms with itself. The
roadmap asks for more than 70% (gate G3, "if it fails"). If it is below 70%, try 400 ms once;
if still below, stop and report.

## 4. Stage 2: training (P3.3)

A `train` command in the same file, or a default-off switch in `fog/training/bc.py` if the
rows are written in the layout that trainer reads; whichever is less code. Either way:

- Policy loss: cross-entropy between the network's masked policy and the teacher's visit
  distribution, on rows that have a target.
- Value loss: regression to `0.5 * outcome + 0.5 * teacher value` (outcome alone where the
  teacher has no value), on every row.
- No belief losses, and the model called with `need_belief=False`.
- Model: the typed-token model with `model.order_blind_entities` and
  `model.mechanical_features` on (the configuration of `configs/replay_mechanical_cpu.yaml`
  without the replay-only outcome targets). Random initial weights, seeded. The roadmap says
  to start from the replay blueprint or anchor `A3`; neither exists yet, which is why this is
  a pilot.
- Batch 256, learning rate 3e-4 with cosine decay, 8 epochs, 5% of games held out by game id,
  best epoch chosen by held-out policy cross-entropy. Resumable.
- Report held-out cross-entropy, top-1 agreement with the teacher, and value error per epoch.

Test: on a tiny hand-made batch the gradient of the policy loss equals `softmax - target` on
legal actions, and rows without a target contribute nothing to it.

## 5. Stage 3: three rounds (P3.2)

| Round | Who acts | New labelled decisions | Train on |
| --- | --- | --- | --- |
| 0 | the teacher (`beta = 1`; no network needed) | 60,000 | round 0 |
| 1 | network from round 0, `beta = 0.25` | 60,000 | rounds 0 and 1 |
| 2 | network from round 1, `beta = 0.25` | 60,000 | rounds 0, 1 and 2 |

Each round's network is trained from random weights on all data so far (so the three are a
learning curve at 60,000, 120,000 and 180,000 decisions). At 0.26 s a label on four cores a
round is about 70 minutes of collection. If the budget runs short, drop round 2, not the
evaluation.

## 6. Stage 4: play (P3.4)

All play is the network alone, greedy, no search, on the validation teams, with
`fog.eval.showdown_direct_selection` in matched blocks. The lead is whatever the untrained
lead head picks; say so, and use the same rule in every row.

1. Each round's network: 400 games against SimpleHeuristics and 400 against MaxDamage (smoke;
   minutes each).
2. The best network: at least 1,000 matched games against each of the two (these are rows
   the roadmap accepts), and 400 games against EngineMCTS at the default budget (smoke; about
   an hour and a half on four cores with `fog.eval.run_shards`).
3. The same 400-game cells for the untrained network, as the floor.

For reference: gate G3 asks the distilled policy to score at least 30% against EngineMCTS at
the default budget and to beat its initial weights; tier T1 asks 90% against SimpleHeuristics
and 97% against MaxDamage. State where the pilot stands against each; claim neither on fewer
games than they require.

## 7. What to hand back

1. Code and tests; `.venv/bin/ruff check fog tests` clean; the digests before and after.
2. One ledger section (`## <date>: search distillation pilot (P3, generation 0)`) with the
   printed rows, commands and commits.
3. `ROADMAP.md`: dated lines under P3.1 to P3.4. Tick P3.1 if its acceptance is met. Do not
   tick P3.2 (it asks for one million decisions) or write a G3 result from a 400-game cell.
4. `docs/briefs/15-artifacts/` (label check, per-round metrics, play JSONs, budget record;
   no checkpoints or data shards) and where checkpoints are kept, with their SHA-256.
5. `docs/briefs/15-report.md`, at most one page: **Answer** (teacher agreement; the three
   networks' scores against both bots; the 1,000-game rows; the score against EngineMCTS;
   whether strength is still rising with data), then what was not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 8. How the reviewer will check this

- Generation used train teams only and evaluation validation teams only (the commands show
  it).
- The replay test of section 3 passes on the reviewer's machine; digests are unchanged.
- Play rows match their JSON files; nothing below 1,000 games is called more than smoke.
- No search or model code changed; the new trainer switch, if any, defaults to off.
