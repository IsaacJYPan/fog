# Replay blueprint at scale on corrected rows — 2026-11-16

**Answer (partial: CPU stage complete, GPU submission blocked).** The first
20,000-game training set has **989,494 rows**; the fixed **2,000-game holdout has
103,944 rows**. No new NLL, top-1, value or play result exists. P2.4 cannot be
applied. Every G2 clause (A3 head-to-head, SimpleHeuristics strength, value ECE
and Brier) is **not measurable** for this run. GPU usage by this brief is **0 h**.

Fresh conversion at application source `79283c9` (unchanged in `7d17e1a`) accepted
all **24,000 games / 1,229,877 rows**, with no quarantines and dense terminal
targets. No brief 05 shards were reused. The fixed holdout excludes 203 newer
non-holdout games, leaving 21,797 training candidates / 1,115,358 rows. Acting
validation compositions remove **36,563 rows in 1,120 games**, including **28
whole games**, leaving 21,769 games / 1,078,795 rows before the nested 20k
selection. Matching preview sides are masked from lead training. Larger-set
preparation must use `--prefix-manifest` to retain these actual G20 game IDs.

Holdout move-slot label shares are **25.09 / 24.83 / 24.82 / 25.25%**; the unmoved
slot-zero share is **24.70%**. There are 103,933 policy-eligible rows. All 103,944
outcome signs agree with protocol winners. The sign diagnostic initially assumed
six-member teams; its retry supports absent-slot padding and passes, including
two five-member-team games. The failed attempt remains in the artifacts.

R4, batch 1,024, zero belief coefficients and `skip_unweighted_belief=true` are
prepared. The first job stops after one epoch of a resumable 18-epoch schedule;
its estimated session is 0.35 h, with a 1 h hard cap. This estimate uses brief
08's different-recipe smoke rate, not a new measurement. The 20 h budget and 2 h
reserve remain; the larger schedule awaits the calibration measurement.

**Blocker:** the account check found `fog-distill-train-gpu` **QUEUED**, with GPU
enabled. `fog-distill-collect-a` is RUNNING on CPU. Brief 13 allows one GPU job at
a time. Neither job was changed and no additional GPU job was submitted. Quota
is **27.52 h remaining of 45 h**, reset 2026-11-21. The CPU archive is retained
privately off the runtime; a separate download verified its SHA-256. Recover it
with the configured official client: `python docs/briefs/13-artifacts/download_inputs.py --out runs/brief13/recovered`.
[`13-artifacts/`](13-artifacts/) holds commands, counts, budget and job scripts.
No checkpoints exist for this brief yet. Model, loss and trainer code are unchanged.

**Not verified:** GPU training/throughput, best-policy/value metrics and intervals,
lead-stage retention, checkpoint persistence, larger data, learning curve or play.
Checks: 24 source tests passed, four missing-fixture skips; three split tests
passed; Ruff passed. No P2 box or gate is claimed.
