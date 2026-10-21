# Kaggle workflow

Kaggle runs a frozen copy of the local code, recipe and inputs. The official client
handles uploads, submission, status and downloads.

Before preparing a run, copy and fill in `configs/kaggle_results.example.json` as
`configs/kaggle_results.json`, then run the setup/login commands in
[KAGGLE_TRAINING.md](../KAGGLE_TRAINING.md). The local account settings are not committed.

## Commands

From the repository root in PowerShell:

```powershell
.\fog-experiment.cmd prepare --label candidates --trial
.\fog-experiment.cmd submit
.\fog-experiment.cmd status
.\fog-experiment.cmd collect
```

`prepare` prints a unique experiment ID. Subsequent commands accept that ID after the
command; omitting it selects `runs/experiments/latest.json`. Use explicit IDs when
working on several experiments. If the dataset is still processing, `submit` asks you
to run it again. Once submission starts, it refuses a second launch, including after
an ambiguous network failure. Inspect `status` and `state.json` in that case.

For the full recipe, omit `--trial`:

```powershell
.\fog-experiment.cmd prepare --label candidates --job configs/automated_gen9ou_candidates_kaggle.yaml
.\fog-experiment.cmd submit
```

The trial targets 64 decisions; the default full recipe targets 24,000. Rollout
chunks can overshoot the target. Both refresh direct and replay audits and validate
replay shards before training. Each submission requests a T4 with a two-hour maximum
runtime by default. Set `kaggle.timeout_seconds` in a new recipe to change the cap
(600 through 43,200 seconds). The dataset and script are private.

## What is saved

Each `runs/experiments/<id>/` contains:

- `job.yaml`, `experiment.json`, `seal.json`: frozen recipe, individual input hashes,
  parent checkpoint, source revision, and hashes of the exact uploaded files. The
  snapshot includes uncommitted code, so the Git commit alone does not identify it.
- `dataset/bundle.zip` and `notebook/`: the private upload and minimal execution script.
- `state.json`, `HANDOFF.md`: launch state and recovery instructions.
- After collection, `output.zip` and `results/`: checkpoints, optimizer state,
  training metrics, audit reports, resolved configuration, runtime versions and logs.

Each experiment has its own dataset and script. The worker checks the input hashes;
collection checks the returned descriptor and records Kaggle's run ID. Account
settings in `configs/kaggle_results.json` are excluded from archives, and prepared
packages marked `superseded` cannot be submitted. Imports preserve earlier results.

The worker writes checkpoints and logs to `/kaggle/working/training`; the final ZIP
contains that directory. If platform termination prevents export, look for completed
intermediate checkpoints there. Partially written files may be incomplete. Long jobs
can disable numbered-checkpoint pruning to retain learning curves.

The worker reserves five minutes of the frozen runtime cap for export, including setup
time in the budget. At the internal deadline it stops the training process group and
exports completed checkpoints and logs. Such a return has execution code `124`,
`timed_out: true`, and pipeline status `interrupted`; reaching a deadline does not count
as finishing the decision budget. Abrupt platform termination can still prevent export.

Once Kaggle accepts the job, the laptop can sleep. Run status checks and collection
when it is available again; there is no local polling service.

## Actor/learner likelihood check

Recipes can require `safety.policy_contract` with a frozen real-rollout `batch` path
and log-probability `tolerance` (default .001). Include the batch in `bundle_inputs`.
It must use the same initial policy, behavior temperature and bonus as the job.

Before training, the worker compares the learner on its training device against the
CPU actor's recorded probabilities, with and without explicit chain supervision.
A failed comparison stops the job. `policy_contract.json` records the result; this
checks likelihood consistency, not observation quality or learning progress.

## Heuristic cloning

`configs/gen9ou_typed_token_heuristic_bc.yaml` selects a cloning stage in the online
runner. The learner plays the games and receives public SimpleHeuristics labels on
the states it visits. `ppo.policy_objective: heuristic_bc` uses teacher cross entropy
as the policy objective, with no PPO surrogate or PPO KL stop gate. Exact submitted
teams provide marginal and autoregressive belief targets; those labels do not enter
the acting policy. The value head fits completed outcomes of the learner's games.
The default opponent pool for this stage contains SimpleHeuristics and MaxDamage.

Keep `ppo.heuristic_coef` and its final coefficient zero in this mode: the teacher
loss is already the primary objective. An optional
`ppo.heuristic_voluntary_switch_weight` emphasizes teacher switches when moves were
also legal, without upweighting forced switches. Its default is one. Metrics retain
unweighted teacher NLL and report move, forced-switch and voluntary-switch agreement
separately. `loss.require_belief_labels` refuses updates missing either team's
supervision or either belief decoder output. Configuration digests distinguish this
recipe from PPO when resuming optimizer state.

Evaluate cloning against SimpleHeuristics on held-out teams before starting
self-play. During self-play, keep exact-team belief supervision active and check
held-out belief quality alongside win rates.

## Evaluation and continuation

```powershell
.\fog-experiment.cmd evaluate <experiment-id> --pool data/teams/public_regeneration_splits/validation.json --games 64
```

Supply your evaluation pool; the example filename must match your split files.
Evaluation uses local CPU, matched games against the parent,
SimpleHeuristics and MaxDamage, with recorded hashes and a fixed seed. It rejects team
composition overlap with the frozen live-training pool, including renamed/reordered
teams. This check does not prove independence from every historical checkpoint or
replay donor: review split provenance too. A tiny trial is only an execution check.

Review `evaluation-*/summary.json` and the full report before changing any champion.
Evaluation runs with current local source; keep model/encoder compatibility with the
snapshot and record code changes when interpreting results. No command promotes a model.
To continue a selected result, pass `prepare --parent <downloaded-checkpoint-path>`
(a file inside this repository), or create a new YAML recipe pointing `initial_checkpoint`
to that checkpoint, then prepare a new experiment. This starts a new PPO
block from weights. Use `ppo.resume_ckpt` for optimizer/RNG continuation; the optimizer
sidecar is retained for that workflow.

For recovery, check `runs/experiments/latest.json` and the selected experiment's
`HANDOFF.md` and `state.json`. Credentials stay in Kaggle's user credential store.
If authentication expires, run `fog-kaggle.cmd login` again.

Earlier saved notebook exports can be fetched with `fog-kaggle.cmd pull`; see
[kaggle-results.md](kaggle-results.md). It is separate from these immutable experiments.
