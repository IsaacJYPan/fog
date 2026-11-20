# Brief 16: EngineMCTS on the held-out test teams

**Answer:** EngineMCTS meets T1 on the replay-derived test pool: 927/1,008 wins
against SimpleHeuristics, **91.96% [95% Wilson 90.12%, 93.49%]**, and 980/1,008
against MaxDamage, **97.22% [96.01%, 98.07%]**. Fallback rates are **0.36%**
(102/27,956 decisions) and **0.26%** (63/24,039), respectively. Both rows have
**zero rejected actions, failed games and truncations**.

| T1 clause | Result |
| --- | --- |
| At least 90% against SimpleHeuristics | Met: 91.96% |
| At least 97% against MaxDamage | Met: 97.22% |
| Held-out teams | Met: 62 teams from `data/teams/replay_v1/test` |
| Pinned Showdown | Met: `e440c4a18385274f10c405d0b158b6a962ce6d94` |
| At least 1,000 matched games each | Met: 1,008 each, four-game blocks crossing teams and sides |

Clause decisions use the measured rate, as the brief requires.

| Opponent | Test, this brief: win rate [95% Wilson] | Validation, brief 11 |
| --- | --- | --- |
| SimpleHeuristics | 91.96% [90.12%, 93.49%] | 92.86% [91.10%, 94.29%] |
| MaxDamage | 97.22% [96.01%, 98.07%] | 97.32% [96.13%, 98.15%] |

Validation fallbacks were 0.19% and 0.17%, with zero rejected actions, failures or
truncations. No new strength comparison is inferred from the small rate differences.

One attempt, unchanged source `79283c9`, CPU only on AMD EPYC 9V74 with a four-core
quota. Default K=8, 1,600 ms, one search thread, four shards, seeds 16000–16003,
no sequential stop; measured 1.812 and 1.849 seconds per decision. The exact command,
environment, four shard JSONs, copied logs and `test/merged.json` are in
`16-artifacts/`. The existing merger was called after completion because tool-session
reconnections left the matches running but no pooled report; no match was replayed.
The pooled rows reproduce exactly from the shards. The 15 preflight tests passed;
application code did not change.

These are **replay-derived teams with usage-sampled sets** (ledger, "review of
brief 10"). The claim must be repeated on the forum bank's test split if it is ever
collected. No code or parameter was tuned on these games.
