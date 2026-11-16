# Brief 16: EngineMCTS on the held-out test teams

**Answer:** SimpleHeuristics: 91.96% [95% Wilson 90.12%, 93.49%] over 1,008 games;
fallbacks 102/27956 (0.36%), rejected actions 0,
failed games 0, truncations 0. MaxDamage is still running, so T1 as a whole is not yet decided.

SimpleHeuristics's ≥90% clause is met. The held-out-team, pinned-Showdown and
≥1,000-game clauses are met for this opponent; MaxDamage's clauses await its completed row.

Validation reference (brief 11): SimpleHeuristics 92.86% [91.10%, 94.29%],
MaxDamage 97.32% [96.13%, 98.15%], 1,008 games each; fallbacks 0.19% and 0.17%,
zero rejected actions, failures or truncations.

No code changed from source `79283c9`. CPU only, AMD EPYC 9V74, four-core quota;
SimpleHeuristics measured 1.812 s/decision. The command, environment,
shard checkpoints, copied logs and completed opponent summary are in `16-artifacts/`.
This is the prescribed one attempt, with no change to seed, budget or game count.
The replay-derived teams use usage-sampled sets (ledger, "review of brief 10");
any tier claim must be repeated on the forum bank's test split if it is ever collected.
