# Replay blueprint GPU scale — 2026-11-14 (smoke, G100 pending)

**Answer.** Full G20 trains at **583.24 rows/s** on a T4, with **161–166 s** validation
and **2.44 GiB** peak allocation. At 20,000 games, policy NLL is **1.278605**, versus B's
**1.579018**. Its own-best value Brier is **0.215320**, with late-third Brier **0.065683**
below early; that checkpoint is **eight epochs before** the best policy. The exact
20k-to-100k P2.4 decision and strict 100k value reading remain **unverified**. The budget
selects a **40,955-game, six-epoch** fallback next; CPU play scores **0.520 vs SimpleHeuristics / 0.8675 vs MaxDamage**. **No gate is claimed.**

| Run (smoke) | Games; epochs; batch | Policy epoch | Top-1 | NLL | Value Brier | Value ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B | 3,000; 18; 256 | 8 | 0.440802 | 1.579018 | 0.344501 | 0.298756 |
| G3 | 3,000; **2/18**; 1024 | 2 | 0.300391 | 1.895671 | 0.244778 | 0.033124 |
| G20 | 20,000; 18; 1024 | 12 | 0.544369 | 1.278605 | 0.254356 | 0.153633 |
| G100, planned fallback | 40,955; 6; 1024 | pending | — | — | — | — |

| Checkpoint (smoke) | Epoch | Train MSE | Holdout MSE / Brier / ECE | Early / middle / late Brier | Early / middle / late ECE |
| --- | ---: | ---: | --- | --- | --- |
| B policy | 8 | 0.272015 | 1.378004 / 0.344501 / 0.298756 | 0.343345 / 0.363723 / 0.327324 | 0.278211 / 0.328123 / 0.291107 |
| B value | 4 | 0.746103 | 0.956552 / unavailable / unavailable | unavailable | unavailable |
| G3 policy/value (partial) | 2 | 0.988902 | 0.979114 / 0.244778 / 0.033124 | 0.248064 / 0.244514 / 0.241777 | 0.020288 / 0.027854 / 0.050854 |
| G20 policy | 12 | 0.543421 | 1.017423 / 0.254356 / 0.153633 | 0.270891 / 0.277673 / 0.215759 | 0.129318 / 0.194757 / 0.138509 |
| G20 value | 4 | 0.832760 | 0.861281 / 0.215320 / 0.016159 | 0.244754 / 0.222779 / 0.179070 | 0.021834 / 0.032837 / 0.027886 |

Every available phase MSE/Brier/ECE, extreme-bin share and observed win rate is in
[the full value tables](05-artifacts/curve-value-tables.json); all ten policy-checkpoint
reliability bins are in [reliability.csv](05-artifacts/reliability.csv). B's own-best value
weights are absent: only its recorded MSE is available. G20's policy checkpoint puts
29.12% of rows below 0.1 or above 0.9; its own-best value checkpoint puts 3.71% there.
For the latter, low/high shares are 2.22%/1.49%, with observed win rates 0.078754/0.946521.
At the policy checkpoint its value Brier has worsened to 0.254356; preserve both selections.

**Data and budget.** CPU preparation produced **20,000 / 1,023,525 rows** and
**101,928 / 5,217,022 rows**, with the fixed **2,000 / 104,037** holdout byte-identical.
All 112,000 games converted; 8,072 newer non-holdout games were excluded. The rating floor
stays 1500. Both inputs are private. Import, conversion and splitting used CPU.

Calibration used **0.879015 Kaggle quota-hours**, retaining two G3 epochs and proving
cross-job continuation: CSV `[0,1]` is an exact byte prefix of `[0,1,2]`. Full 18-epoch G3
projects beyond two hours, so the one-hour calibration cap prevents completing it and
the batch-256 comparison. No batch/device attribution is claimed. G20 completed all
18 epochs at `7a23497`, best policy/value epochs **12/4**. Its own session upper bound is
**9.840590 h**; calibration plus G20 is at most **10.719605 quota-hours**. The account counter
is **11.118819 h**, including another brief's earlier work. G100 conservatively charges
that whole counter: its largest nested prefix projects to **6.881063 h**, leaving two hours
inside 20. All prepared games for six epochs project to **15.865326 h** and do not fit.
These are projections, not completed G100 results; see [G100-plan.json](05-artifacts/G100-plan.json).
The initial shared-account overlap and corrected account-wide submission guard remain in
the ledger. The independent brief-06 timing is finished; G20 was not interrupted for it.

**Play and checks.** G20's 400 greedy games per opponent on `pool_b` score **0.520
[0.471, 0.569]** vs SimpleHeuristics and **0.8675 [0.831, 0.897]** vs MaxDamage, with
zero failures/truncations. B scored **0.360 [0.314, 0.408] / 0.7725 [0.729, 0.811]**
respectively. G100 awaits training. All play rates are smoke, not strength gates.
No protocol/manifest outcome mismatch appeared in 103,928 games. In 128 sampled games,
6,283 serialized labels agree per side and 3,150 earlier observations survive future deletion.
G20's 256-row label/oracle perturbation leaves policy/value tensors identical in evaluation
and matched-RNG training modes. These are bounded input checks; the named leakage module
supplies readability metrics rather than a privileged-input checker. No model, loss,
schedule or filters changed. Both added options default off; **71 affected tests**, **6 job
tests**, and Ruff passed. P2.1–P2.5 stay unchecked.

G3 (partial) and G20's four selected files are in the **one private checkpoint dataset**,
version 2; the download below was executed and every size/SHA-256 matched
[checkpoints.json](05-artifacts/checkpoints.json). G100 files will be added there too.

```sh
uv run --no-project --with kaggle==2.2.5 --with six python -c 'from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.dataset_download_files(a.config_values["username"]+"/fog-p2-gpu-checkpoints-20261114", path="checkpoints", unzip=True, quiet=True)'
```

Not verified: full G3/batch calibration, G100 training/play, either exact 100k reading,
B's missing own-best value phases/ECE, exhaustive input isolation, or any strength gate.
