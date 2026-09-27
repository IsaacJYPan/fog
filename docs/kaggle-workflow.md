# Local development and private Kaggle training

Use the local repository as the source of code and experiment recipes. Kaggle runs a
frozen snapshot on its free GPU quota. The official authenticated client handles uploads,
submission, status and downloads; ordinary runs require no browser interaction.

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

For a full bounded continuation, omit `--trial`:

```powershell
.\fog-experiment.cmd prepare --label candidates --job configs/automated_gen9ou_candidates_kaggle.yaml
.\fog-experiment.cmd submit
```

The trial targets 64 decisions; the default full recipe targets 24,000. Rollout
chunks can overshoot the target. Both refresh direct and replay audits and validate
replay shards before training. Each submission requests a T4 with a two-hour maximum
runtime by default and uses existing free quota. Set `kaggle.timeout_seconds` in a new
recipe for a different frozen cap (600 through 43,200 seconds). It creates a private dataset and private script;
there is no purchasing, billing setup, or public publishing step.

## What is saved

Each `runs/experiments/<id>/` contains:

- `job.yaml`, `experiment.json`, `seal.json`: frozen recipe, individual input hashes,
  parent checkpoint, source revision, and hashes of the exact uploaded files. The
  snapshot includes local uncommitted code, so the Git commit alone is not its identity.
- `dataset/bundle.zip` and `notebook/`: the private upload and minimal execution script.
- `state.json`, `HANDOFF.md`: launch state and recovery instructions.
- After collection, `output.zip` and `results/`: checkpoints, optimizer state,
  training metrics, audit reports, resolved configuration, runtime versions and logs.

Every experiment gets its own dataset and script, avoiding accidental updates to a
running experiment. The worker verifies the complete input hash inventory. Collection
checks the returned experiment descriptor before importing and records Kaggle's run ID.
Recipes can also require `safety.policy_contract` with a frozen real-rollout `batch` path
and a log-probability `tolerance` (default .001). Include that batch in `bundle_inputs`.
Before training, the worker checks the initial learner against the CPU actor's recorded
probabilities on the actual training device, with and without explicit chain supervision.
Any failed comparison stops the job; `policy_contract.json` records successful or numerical
failure results. The batch must come from the same initial policy and behavior temperature
and bonus. This is a likelihood check, not proof of observation quality or useful learning.
Archive validation prevents unsafe extraction; imports never replace earlier results.
The worker writes checkpoints and logs to `/kaggle/working/training` throughout the
run, with a link from the frozen recipe's output path. The final ZIP contains that
directory. If platform termination prevents ZIP export, inspect the saved training
directory for earlier completed checkpoints; final or partially written files may
be incomplete. Long jobs can disable numbered-checkpoint pruning to retain learning curves.

Once Kaggle accepts the job, the laptop can sleep or close. Training continues remotely.
Status checks, downloads and local evaluation resume when the laptop is available;
these commands do not install a background polling service. Hard timeouts or platform
termination can interrupt final export, so keep jobs bounded and collect completed runs.

## Evaluate before choosing the next parent

```powershell
.\fog-experiment.cmd evaluate <experiment-id> --pool data/teams/public_regeneration_splits/validation.json --games 64
```

Supply an existing, deliberately selected evaluation pool; the example filename must
match your split files. Evaluation uses local CPU, matched games against the parent,
SimpleHeuristics and MaxDamage, with recorded hashes and a fixed seed. It rejects team
composition overlap with the frozen live-training pool, including renamed/reordered
teams. This check does not prove independence from every historical checkpoint or
replay donor: review split provenance too. A tiny trial establishes execution, not quality.

Review `evaluation-*/summary.json` and the full report before changing any champion.
Evaluation runs with current local source; keep model/encoder compatibility with the
snapshot and record code changes when interpreting results. No command promotes a model.
To continue a selected result, pass `prepare --parent <downloaded-checkpoint-path>`
(a file inside this repository), or create a new YAML recipe pointing `initial_checkpoint`
to that checkpoint, then prepare a new experiment. This starts a new PPO
block from weights; it does not claim an exact optimizer/RNG continuation. The optimizer
sidecar is retained for explicit resume workflows.

To resume work, start with this guide, `runs/experiments/latest.json`, and the
selected experiment's `HANDOFF.md` and `state.json`. Credentials stay in Kaggle's user
credential store, outside the repository and bundles. If authentication expires, use
the existing `fog-kaggle.cmd login` flow once; normal commands reuse the connection.

Earlier saved notebook exports can be fetched with `fog-kaggle.cmd pull`; see
[kaggle-results.md](kaggle-results.md). It is separate from these immutable experiments.
