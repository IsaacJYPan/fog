# Training on Kaggle

Develop and evaluate locally, then send a frozen training package to Kaggle. Each
experiment records its code, recipe, data and parent checkpoint.

## Setup

Install the local environment described in [README.md](README.md), then create the account-local
settings file:

```powershell
Copy-Item configs/kaggle_results.example.json configs/kaggle_results.json
```

Set `kernel` to `your-account/your-notebook`, `version` to its saved version, and `run_id` to the
`scriptVersionId` in that version's URL. These settings are ignored by Git. New immutable
experiments use the owner portion of `kernel`; the remaining fields identify a legacy notebook
for `fog-kaggle pull`. The sample numeric values are placeholders, not a downloadable run.

```powershell
.\fog-kaggle.cmd setup
.\fog-kaggle.cmd login
```

The official Kaggle client installs into `.tools/kaggle`, separate from the training environment.
Authentication stays in the user-level credential store. On other platforms, the equivalent modules are
`python -m fog.scripts.kaggle_results` and `python -m fog.scripts.kaggle_experiment`.

## Prepare the data

Recipes require their referenced team pools, replay manifests, and checkpoint files. These
generated inputs are not distributed in Git. See the
[candidate training guide](docs/kaggle-candidate-training.md) for reconstruction and split commands,
and the [Metamon guide](docs/metamon-gen9ou.md) for pinned public-log imports.

Collect and split published teams before selecting the training pool:

```powershell
python -m fog.scripts.collect_team_bank --target 150 --min-quality 70
python -m fog.scripts.build_public_team_splits --bank data/teams/public_gen9ou --out data/teams/gen9ou_public_v1
```

Check the collection errors, legality report and deduplicated splits, then update the
recipe paths. Recollecting public teams will produce a different corpus. The split
does not establish independence from older checkpoints.

## Freeze, train, and collect

```powershell
.\fog-experiment.cmd prepare --label candidate-trial --job configs/automated_gen9ou_candidates_kaggle.yaml --trial
.\fog-experiment.cmd submit <experiment-id>
.\fog-experiment.cmd status <experiment-id>
.\fog-experiment.cmd collect <experiment-id>
```

`prepare` prints the experiment ID. Each experiment creates a private dataset and
saved script; once Kaggle accepts the submission, the local machine can sleep.

The trial targets 64 decisions; full-game collection can overshoot it. Omit `--trial`
for the recipe's full 24,000-decision budget. This is an execution baseline; evaluate
the result before spending more training time.

For an exact-team curriculum, the direct simulator can combine outcome PPO with
public-information heuristic labels on states visited by the learner. In the job's
`overrides`, for example:

```yaml
env.belief_supervision: full_truth
model.aux_into_trunk: true
loss.value_coef: 0.25
loss.belief_coef: 0.05
loss.belief_chain_coef: 0.05
ppo.heuristic_teacher: simpleheuristics
ppo.heuristic_coef: 0.3
ppo.heuristic_coef_final: 0.0
ppo.heuristic_decay_decisions: 250000
```

These settings require `showdown_direct` and submitted team sheets. Hidden sheets
supply auxiliary targets only. The teacher sees the acting player's censored battle
view and uses an independent random stream; the policy still chooses the actions.
The teacher coefficient reaches zero at the stated decision count. Compare against
an equal-budget control to find out whether the guidance helps.

The worker checks the input hashes, preserves Kaggle's CUDA PyTorch, installs other
dependencies, validates replay shards and refreshes direct/replay audits. It saves
checkpoints, optimizer sidecars, metrics and logs under `/kaggle/working/training`.
Five minutes are reserved for export; a timed-out run is marked `interrupted`.
Platform termination can still prevent the final archive, so inspect intermediate
outputs if needed.

`collect` checks the returned descriptor and saves the archive and imported results
under `runs/experiments/<id>/`. Check the result status: failed runs can also export
archives. See [the workflow reference](docs/kaggle-workflow.md) for recovery details.

## Evaluate locally

```powershell
.\fog-experiment.cmd evaluate <experiment-id> --pool data/teams/gen9ou_public_v1/validation.json --games 256
```

Use your held-out validation pool. The command rejects composition overlap with the
frozen live-training pool and runs matched games against the parent and scripted
opponents. Evaluation uses current local source, so check compatibility with the
training snapshot. Reserve a separate test set for promotion; a smoke run is too
small for that decision.

`prepare --parent <checkpoint>` starts a new block from weights. Exact PPO state
restoration uses `ppo.resume_ckpt` instead.

## Manual notebooks and saved runs

`notebooks/kaggle_candidates.ipynb`, `kaggle_training.ipynb`, and `kaggle_trial.ipynb` support manual
bundle workflows. Their code cells have no saved outputs and require referenced inputs.
Use the script workflow above for new experiments. Older exports can be retrieved with
`fog-kaggle.cmd pull`; see [saved-run retrieval](docs/kaggle-results.md).
