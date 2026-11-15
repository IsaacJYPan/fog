# Replay blueprint GPU scale — 2026-11-15 (smoke)

**Answer.** T4 training runs at **583 / 644 rows/s** for G20 / the larger fallback.
Policy NLL is **1.579018 (B) → 1.278605 (20k) → 1.263437 (40,955 games)**.
The larger point completed six epochs; its own-best value Brier is **0.208909**, with late
Brier **0.075929** below early, one epoch before its best policy. Those numerical value
thresholds are met at 40,955 games; the strict **100k** reading and exact P2.4 branch remain
**unverified**. CPU play intervals overlap G20 in this smoke. Conservative
account usage is **17.078547 / 20 Kaggle quota-hours**. **No gate is claimed.**

| Run (smoke) | Games; epochs; batch | Policy epoch | Top-1 | NLL | Value Brier | Value ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B, historical CPU | 3,000; 18; 256 | 8 | 0.440802 | 1.579018 | 0.344501 | 0.298756 |
| G3, partial | 3,000; **2/18**; 1024 | 2 | 0.300391 | 1.895671 | 0.244778 | 0.033124 |
| G20 | 20,000; 18; 1024 | 12 | 0.544369 | 1.278605 | 0.254356 | 0.153633 |
| G100, budget fallback | 40,955; 6; 1024 | 6 | 0.545475 | 1.263437 | 0.210989 | 0.024287 |

| Checkpoint (smoke) | Epoch | Train MSE | Holdout MSE / Brier / ECE | Early / middle / late Brier | Early / middle / late ECE |
| --- | ---: | ---: | --- | --- | --- |
| B policy | 8 | 0.272015 | 1.378004 / 0.344501 / 0.298756 | 0.343345 / 0.363723 / 0.327324 | 0.278211 / 0.328123 / 0.291107 |
| B value | 4 | 0.746103 | 0.956552 / unavailable / unavailable | unavailable | unavailable |
| G3 policy/value (partial) | 2 | 0.988902 | 0.979114 / 0.244778 / 0.033124 | 0.248064 / 0.244514 / 0.241777 | 0.020288 / 0.027854 / 0.050854 |
| G20 policy | 12 | 0.543421 | 1.017423 / 0.254356 / 0.153633 | 0.270891 / 0.277673 / 0.215759 | 0.129318 / 0.194757 / 0.138509 |
| G20 value | 4 | 0.832760 | 0.861281 / 0.215320 / 0.016159 | 0.244754 / 0.222779 / 0.179070 | 0.021834 / 0.032837 / 0.027886 |
| G100 policy | 6 | 0.794077 | 0.843957 / 0.210989 / 0.024287 | 0.245429 / 0.218727 / 0.169517 | 0.032741 / 0.036234 / 0.008662 |
| G100 value | 5 | 0.800861 | 0.835635 / 0.208909 / 0.009065 | 0.243596 / 0.216147 / 0.167667 | 0.021998 / 0.018296 / 0.014701 |

All phase MSE/Brier/ECE, training MSE, extreme shares/win rates and source reports are in
[the full value tables](05-artifacts/curve-value-tables.json); all ten policy-checkpoint
reliability bins are in [reliability.csv](05-artifacts/reliability.csv). B's own-best value
weights are absent, leaving only its recorded MSE. For G100's own-best value checkpoint,
low/high shares are **2.67% / 3.63%**, with observed win rates **0.078198 / 0.930103**.
G20's value optimum is eight epochs before its policy optimum; G100's is one before.

**Reading.** The observed 20k→40,955 NLL improvement is **0.015168**; larger-point train NLL
is **1.225838** versus holdout **1.263437**. Neither prescribed 100k branch or strict value reading is established.

**CPU play**, greedy `pool_b`, **400 games per opponent per model**: B scored **0.360
[0.314, 0.408] / 0.7725 [0.729, 0.811]** against SimpleHeuristics / MaxDamage;
G20 **0.520 [0.471, 0.569] / 0.8675 [0.831, 0.897]**;
G100 **0.5225 [0.474, 0.571] / 0.8575 [0.820, 0.888]**. Both new models have zero
failures/truncations. Their intervals overlap. Raw play files are in `05-artifacts/`. The narrow pool supplies no strength gate.

**Data, time and limits.** CPU preparation produced **20,000 / 1,023,525 rows** and
**101,928 / 5,217,022 rows**, with the fixed **2,000 / 104,037** holdout byte-identical.
All 112,000 converted; 8,072 newer non-holdout games were excluded, rating floor 1500.
Full prepared data for six epochs projected to **15.865326 h** and did not fit. The largest
seeded nested prefix fitting the measured forecast plus a two-hour reserve was
**40,955 / 2,090,705 rows**, six epochs. No import, conversion or splitting used GPU time.

G20 completed 18 epochs at `7a23497`; G100 six at `46e9c0d`. Per-epoch GPU train rates are
580–586 / 642–646 rows/s; validation is 161–166 / 144–147 s, peak allocation about
2.44 GiB each. Both used device 0 of two visible T4s; full environment/timers are retained. G100 exported at **00:33 UTC**, all follow-up work finished by **00:41 UTC** on
15 November. Its charge bound is **5.959728 h**; the long uncharged queue is excluded.
Calibration used **0.879015 h**, G20's own-session bound **9.840590 h**. The **17.078547 h account meter** includes earlier work from another brief. Per-job scopes
and the initial account-overlap correction remain in `gpu-hours.json` and the ledger.

Full G3 and the batch-256 comparison did not fit the one-hour calibration cap; G3 remains
**2/18**, with cross-job CSV `[0,1]` preserved byte-for-byte inside `[0,1,2]`. No batch/device
quality attribution is made. Checks: 103,928 manifest outcomes match protocol winners; 128 sampled games verify
6,283 per-side labels and 3,150 prefix observations. G100's 256-row label/oracle
perturbation preserves policy/value tensors in both modes. The named leakage module supplies readability metrics; these bounded checks are not
exhaustive isolation proof. Model, losses, schedule and filters are unchanged; both added
options default off. **71 affected tests**, **6 job tests**, Ruff passed.
P2.1–P2.5 remain unchecked.

All **six selected files**, including partial G3, are in **one private dataset, version 3**.
The following owner command was executed; every size/SHA-256 matched
[checkpoints.json](05-artifacts/checkpoints.json). Older versions are retained.

```sh
uv run --no-project --with kaggle==2.2.5 --with six python -c 'from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.dataset_download_files(a.config_values["username"]+"/fog-p2-gpu-checkpoints-20261114", path="checkpoints", unzip=True, quiet=True)'
```
