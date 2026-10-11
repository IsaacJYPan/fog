# Automatic GPU-to-CPU evaluation

Training and evaluation use separate private Kaggle jobs. Training requests a GPU; development evaluation requests no accelerator and runs policy inference on CPU. The laptop can be closed after the queue is armed and training is submitted. Results still need review before a new recipe or model promotion.

Kaggle supports [output-update scheduling](https://www.kaggle.com/product-feedback/284587): attach the training notebooks' outputs and select **Data updates (once daily)** in the evaluation notebook's Settings tab. Scheduling is configured through the website; the current CLI does not expose it. Disable an obsolete schedule before arming a new experiment if the account's private scheduling limit is reached.

## Prepare the chain

1. Freeze both GPU packages with identical data, recipes and update budgets.
2. Run `python -m fog.scripts.pretrained_experiment register RUN_ID` for each arm. Each registration produces a tiny CPU-only output, without executing training or reserving GPU hours. Kaggle requires an existing output before a downstream notebook can attach it. The frozen training code and inputs stay intact; the actual GPU submission creates a subsequent version.
3. Prepare an `evaluate` package whose configuration includes `training_sources` instead of `conditions` and `matched`. Each `ordinary`/`joint` source specifies its local frozen `experiment.json`, that file's `experiment_sha256`, and its `owner/notebook-run` kernel. Keep the normal development game protocol and optional fixed belief cohort. Candidate hashes are resolved from completed exports. An optional `archive_sha256` pins an already completed source.
4. Run `python -m fog.scripts.pretrained_experiment submit EVALUATION_ID`. This performs an initial CPU run. With registered training placeholders, it only exports `waiting`, without installing environments or playing games. Enable its Data updates schedule and verify that it is private with Accelerator None. Record `SCHEDULE.json` in its experiment folder with `enabled: true`, `trigger: "data_updates"`, `kernel`, and the ordered `sources` from its frozen notebook metadata. This receipt records a verified website setting; writing it does not create a schedule.
5. Submit both GPU packages. Newly prepared GPU packages require exactly one armed, hash-matched CPU evaluation queue before allocating GPU time.
6. Collect the evaluation after completion. Collection preserves scheduled runs separately, including an earlier waiting result and a later completed comparison. Review the reports locally before promotion or another training recipe.

Use a new experiment identity when a recipe changes. Do not repoint an existing evaluation at different training notebooks or overwrite sealed files. Keep Kaggle credentials in the authenticated local client, outside Git; the scheduled jobs need no account token in their source or datasets.

## Verification and limitations

The CPU queue verifies every training export's file hash, original recipe, completed update count, CPU/CUDA probability checks, finite optimization metrics, and matched data schedules. It retains the candidate exports and probability references without copying optimizer checkpoints into its working directory. Full training-state tensor audits remain part of result review. If an output is absent, it exports a `waiting` status before installing environments or playing games. Failed or mismatched training outputs stop evaluation. Each scheduled run has a frozen whole-job deadline and exports partial results on interruption. Evaluation never promotes a model automatically.

Update triggers run at most once per day per scheduled notebook; subsequent updates can be coalesced until the next day. If one arm finishes much earlier, the first trigger may only record waiting. This workflow does not promise an immediate handoff. See [Kaggle's scheduling documentation](https://www.kaggle.com/docs/notebooks).

Scheduling a Quick Save produced a conversion-only run in a control probe; its Complete status did not mean its cells executed. Queued evaluations therefore require an initial CPU submission before enabling updates. They use a thin, valid `.ipynb` that launches the sealed worker with a clean argument list. A separately frozen queue pilot may declare additional `trigger_sources` to test dependency resolution using existing training exports; ordinary evaluations cannot add such triggers.

CPU development comparisons cover at least 256 crossed games per arm against unchanged Kakuna, preceded by eight smoke games and numerical checks. Optional belief diagnostics use a separately frozen exact-team development cohort. Final confirmation cohorts and public rated play remain subject to their own registered gates. Running successfully does not establish better playing strength.
