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
| 05 | `05-p2-gpu-scale.md` | GPU | running | training only | `work/p2-gpu-scale` | larger run finishing; then review |
| 09 | `09-order-blind-model.md` | CPU | now | model, converter | `work/order-blind-model` | not started |
| 10 | `10-replay-team-pool.md` | CPU | now | team data, evaluation | `work/replay-team-pool` | not started |
| 08 | `08-update-speed-gpu.md` | GPU, 1 h | brief 05 has released the GPU | nothing (measures) | `work/update-speed` | not started |
| 11 | `11-engine-mcts-gate.md` | CPU, about 3 days of 4 cores | 10 merged | search reconstruction | `work/g1` | waiting |
| 12 | `12-replay-live-parity.md` | CPU | 09 merged | converter, parser | `work/replay-parity` | waiting |
| 13 | `13-replay-scale-corrected.md` | GPU, 20 h | 08 and 09 merged (12 too if it is done) | training only | `work/p2-scale-corrected` | waiting |
| 14 | `14-ppo-update-levers.md` | GPU, 2 h | review of 08 says the update is still over half the wall time | PPO update | `work/ppo-update` | conditional |

Start **09 and 10 today**, in parallel. Start 08 as soon as the GPU is free.

## What each one buys

- **09, 12, 13** are one line of work: a replay policy that is trained on honest rows, cannot
  read slot order, has a value head that says something, sees what it will see in play, and
  is trained on all the data the budget allows. It ends at gate G2.
- **10, 11** are the other line: a held-out team pool, then the search-only agent taken to
  gate G1. EngineMCTS is already the strongest thing in the repository; it has never been
  gated only because there was no pool.
- **08, 14** are about speed. They decide how much training a week of GPU buys.

## After the queue

When G1 and G2 have results, the roadmap continues at P3 (distil EngineMCTS into the policy).
P3 needs a brief written from `ROADMAP.md` P3.1 to P3.4 as they stand; write it then, with the
numbers G1 and G2 produced. If G2 fails on play strength with good offline numbers, the
roadmap already says to go to P3 with the P2 model as initial weights.

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
3. Find the retained imitation checkpoint `A3`, or say it is lost. Gate G2's head-to-head
   clause cannot be measured without it.
4. Delete finished remote branches (workers cannot): every `work/*` branch that the ledger
   says is merged, and the `dev/*` branches.
5. Decisions in `ROADMAP.md` section 8 (ladder account, teams, compute).
