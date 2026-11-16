# Replay blueprint at scale on corrected rows — 2026-11-16

**Answer (in progress).** No new NLL, top-1, value or play result yet. P2.4 cannot
be applied and every G2 clause is not measurable for this run. Briefs 08 and 09
are merged. The fresh quota reading is 27.52 GPU-hours remaining of 45.00, reset
2026-11-21 00:00 UTC; this brief retains its 20-hour limit and two-hour reserve.
No GPU job has been submitted or charged to this brief.

The first CPU import retained 24,000 games, using the fixed source revision,
rating floor 1500 and start date 2024-02-01. Fresh, resumable conversion uses the
merged converter, including dense terminal targets. The fixed 2,000-game holdout
will remain unchanged by the composition filter. Training shards exclude rows
whose acting submitted team matches a validation composition. Preview records
retain two slots per game, with matching sides masked from lead training.

The submitted recipe will be `configs/replay_mechanical_cpu.yaml`, batch 1,024,
with zero belief coefficients and `loss.skip_unweighted_belief=true`. The
resumable job template reuses brief 05's completed-epoch state and export logic;
the first submission stops after one trained epoch of an 18-epoch schedule.
The larger-size forecast awaits that measurement. Checkpoint scoring and play
run on CPU. Model, loss and trainer code are unchanged.

Checks: 24 affected source tests passed, four skipped for missing replay data;
two composition-filter tests passed; Ruff passed. Evidence is in
[`13-artifacts/`](13-artifacts/).

**Not verified:** completed conversion and split counts, holdout slot check,
GPU throughput, schedule/budget forecast, training, lead-stage policy retention,
checkpoint persistence, learning-curve decision and every play/gate clause.
