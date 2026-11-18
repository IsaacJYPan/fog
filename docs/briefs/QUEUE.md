# Work queue

What to run next, in what order, and the prompt for each. Written 2026-11-15 so the project
can continue without the person who planned it. The roadmap says why; this file says what to
start. The reviewer keeps the status column current (`docs/briefs/REVIEW.md`).

Every worker prompt has this form (replace the file name):

> In IsaacJYPan/fog, check out work/roadmap, read AGENTS.md, then carry out
> docs/briefs/`<file>`.

Every review prompt:

> In IsaacJYPan/fog, check out work/roadmap, read AGENTS.md and docs/briefs/REVIEW.md, then
> review branch `<branch>` for brief `<NN>`.

One GPU job at a time on the account. Two CPU workers at a time is fine when their briefs are
in different rows of the "touches" column.

## Queue

| # | Brief file | Needs | Start when | Touches | Branch | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 05 | `05-p2-gpu-scale.md` | GPU | done | training only | `work/p2-gpu-scale` | **merged 2026-11-15** (old rows; see its review) |
| 10 | `10-replay-team-pool.md` | CPU | done | team data, evaluation | `work/replay-team-pool` | **merged 2026-11-15** |
| 08 | `08-update-speed-gpu.md` | GPU, 1 h | done | nothing (measures) | `work/update-speed` | **merged 2026-11-16** |
| 09 | `09-order-blind-model.md` | CPU | done | model, converter | `work/order-blind-model` | **merged 2026-11-16** |
| 11 | `11-engine-mcts-gate.md` | CPU | done | search reconstruction | `work/g1` | **merged 2026-11-16; gate G1 passed** |
| 12 | `12-replay-live-parity.md` | CPU | **now** | converter, parser | `work/replay-parity` | completed and reviewed; corrected replay conversion and paired smoke results integrated |
| 13 | `13-replay-scale-corrected.md` | GPU, 20 h | **now** | training only | `work/p2-scale-corrected` | G20 trained for 18 epochs; lead comparison and validation play running on CPU |
| 15 | `15-distil-engine-mcts.md` | CPU, about a day on 4 cores | see the note under this table | new training script | `work/distil-g0` | code merged; rounds 0 to 3 done by the planner (ledger) |
| 16 | `16-engine-mcts-test-split.md` | CPU, about 8 h on 4 cores | **now** | nothing (measures) | `work/t1-search` | **reviewed; T1 passed on the replay-derived test teams** (first completed run, 1,008 games per opponent) |
| 17 | `17-distil-at-scale.md` | CPU about 150 core-hours, GPU 6 h | the pilot's last round is in the ledger | trainer switches | `work/distil-scale` | eight-epoch model selected; value calibration complete, final head-to-head evaluations in progress |
| 14 | `14-ppo-update-levers.md` | GPU, 2 h | when brief 13 is not using the GPU | PPO update | `work/ppo-update` | turn-limit test and opt-in controls on the assigned branch; fixed-team CPU learning comparison running |

**Gate G1 (brief 11): passed** on 2026-11-16 (ledger, "review of brief 11"): 92.9% against
SimpleHeuristics and 97.3% against MaxDamage over 1,008 games each.

**Briefs 15 and 17:** the planner's session runs the distillation rounds itself, because the
collected rows, the networks and the hosted account are in that session: collection on this
machine and on hosted CPU machines, training on the hosted GPU (or here when the GPU is not
to be had), play here. The ledger's "search distillation" sections say how far it got; rows
and weights are in the owner's private hosted dataset. Give brief 17 to a worker only when
that session has ended; the worker downloads the dataset and continues from the first round
the ledger does not report, with `scripts/hosted.sh`.

GPU: one job at a time. G20 used 1.97 h through epoch 18. The account had
24.47 h of 45 h available afterward. The next replay training is the larger
corrected set after G20 play and CPU conversion. Matched distillation trials
can use the intervening window. Read the live quota before every accelerator job.

## What each one buys

- **09, 12, 13** are one line of work: a replay policy that is trained on honest rows, cannot
  read slot order, has a value head that says something, sees what it will see in play, and
  is trained on all the data the budget allows. It ends at gate G2.
- **10, 11** are the other line: a held-out team pool, then the search-only agent taken to
  gate G1. EngineMCTS is already the strongest thing in the repository; it has never been
  gated only because there was no pool.
- **08, 14** are about speed. They decide how much training a week of GPU buys.
- **15** is the first step of P3: the network learns to imitate the search on positions the
  network itself reaches. It is the shortest path to a learned policy that plays well without
  search, and it needs no GPU.
- **16** measures the search-only agent once on the held-out test teams (tier T1).
- **17** takes the distillation loop from the pilot's 120,000 rows towards one million and
  measures gate G3.

## After the queue

After G3: P4 (learned parts inside search), starting with the learned world sampler, which
is what the budget curve points at (ledger, "review of brief 11 and gate G1").

Self-play stays frozen until gate G3. When it opens, use the anchored form in
`docs/equilibrium-training.md`, not plain PPO.

## Owner

Things only the owner can do, most useful first:

1. **Nothing is required to keep the queue moving.** Brief 10 removes the dependency on the
   forum team bank.
2. Optional, better than brief 10's pool: on a machine where the forum pages open, run
   `python -m fog.scripts.collect_team_bank --target 150 --min-quality 70` and
   `python -m fog.scripts.build_public_team_splits --bank data/teams/public_gen9ou --out data/teams/gen9ou_public_v1`,
   then commit the three split packages. Tier claims should use these when they exist.
3. A3 was recovered on November 17. Its hash, recovery path and three validation matches are in the ledger.
4. Delete finished remote branches (workers cannot): every `work/*` branch that the ledger
   says is merged, and the `dev/*` branches.
5. Decisions in `ROADMAP.md` section 8 (ladder account, teams, compute).
