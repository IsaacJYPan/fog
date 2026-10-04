# Training on Kaggle

Develop and evaluate locally; use Kaggle for a frozen training package. Every changed recipe gets
a new experiment identity. Outputs do not replace the protected model automatically.

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
Authentication stays in the user-level credential store. The launcher does not store credentials
in experiment bundles. On other platforms, the equivalent modules are
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

Inspect source errors, legality reports, semantic deduplication, and split provenance. Adjust
recipe paths to the generated inputs. A fresh bank is not a byte-for-byte recovery of an older
training corpus, and its disjoint splits do not prove independence from historical checkpoints.

## Freeze, train, and collect

```powershell
.\fog-experiment.cmd prepare --label candidate-trial --job configs/automated_gen9ou_candidates_kaggle.yaml --trial
.\fog-experiment.cmd submit <experiment-id>
.\fog-experiment.cmd status <experiment-id>
.\fog-experiment.cmd collect <experiment-id>
```

`prepare` prints the experiment ID and records hashes of the code, configuration, data, and parent
checkpoint. Use that ID explicitly when several runs exist. Each experiment creates a private
dataset and a private saved script. After acceptance, training continues without the local
machine. No local background polling service is installed.

The trial targets 64 decisions; full-game collection can overshoot it. The example full recipe
targets 24,000 decisions and is retained as an execution baseline. Choose budgets through a new
recipe with a fixed evaluation protocol and stopping criteria. A successful execution alone
does not justify a longer training run.

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

These illustrative settings require the `showdown_direct` backend and submitted team
sheets. Hidden sheets supply auxiliary targets only. The teacher uses the acting player's
censored battle view and an independent random stream; its labels never override the
policy's actions or enter policy inputs. Guidance reaches zero at the stated decision
count, leaving subsequent updates to PPO and the remaining auxiliary objectives. Use
an equal-budget control to measure benefit: teacher agreement, nonzero shared gradients,
and auxiliary accuracy do not by themselves establish stronger play.

The worker verifies the frozen inventory, preserves Kaggle's CUDA PyTorch, installs other
dependencies, validates replay shards, and refreshes both direct and replay audits before
training. It saves intermediate checkpoints, optimizer sidecars, metrics, configuration, runtime
versions, and logs under `/kaggle/working/training`. A hard platform termination can prevent final
archive export; inspect retained intermediate outputs if that occurs.
The worker reserves five minutes for export before the frozen session deadline. An
internally timed-out run retains its completed checkpoints and is marked `interrupted`.

Collection verifies the returned experiment descriptor and preserves the archive and imported
results under `runs/experiments/<id>/`. Read the result status and audits: a downloaded archive
can contain a failed run. The detailed command and recovery contracts are in
[docs/kaggle-workflow.md](docs/kaggle-workflow.md).

## Evaluate locally

```powershell
.\fog-experiment.cmd evaluate <experiment-id> --pool data/teams/gen9ou_public_v1/validation.json --games 256
```

Select an existing evaluation pool deliberately. The command rejects semantic composition
overlap with the frozen live-training pool and evaluates matched games against the parent and
scripted anchors. It uses current local source; record compatibility and source hashes before
comparing results. Validation selects experiments; a separate test protocol is required for
promotion. Small smoke evaluations establish execution only.

`prepare --parent <checkpoint>` starts a new block from weights. Exact PPO state restoration is
a separate `ppo.resume_ckpt` path. Keeping optimizer sidecars does not turn weights-only block
initialization into exact continuation.

## Manual notebooks and saved runs

`notebooks/kaggle_candidates.ipynb`, `kaggle_training.ipynb`, and `kaggle_trial.ipynb` support manual
bundle workflows. Their code cells have no saved outputs and require referenced inputs.
The immutable script workflow above is preferred for new experiments because it records input
identity and avoids editing a running recipe. Older exports can be retrieved with
`fog-kaggle.cmd pull`; see [saved-run retrieval](docs/kaggle-results.md).
