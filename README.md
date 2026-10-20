# FOG

FOG is a research agent for Pokémon Showdown Gen 9 OU singles. It learns a battle policy,
estimates hidden team information, and studies whether search can improve play under partial
observability. Battles use a pinned local Showdown simulator.

The repository supports typed-token models and an optional
[native Kakuna experiment](docs/native-kakuna.md) that preserves Metamon's pretrained policy while
testing Gen9OU fine-tuning with shared belief and outcome supervision. The earlier three-encoder
model and experimental belief-space search are retained for comparison. See
[ARCHITECTURE.md](ARCHITECTURE.md) for FOG's model contracts and implementation limits, and the
[native guide](docs/native-kakuna.md) for Kakuna's observation and training contracts.

Neither the typed-token PPO pilots nor native Kakuna fine-tuning have established a reliable
playing-strength gain over their respective starting checkpoints. The latest native runs and
next diagnostic questions are summarized in the [training review](docs/training-review.md).

The current [prior-conditioned AR pilot](docs/kakuna-prior-chain.md) targets useful hidden-team
sampling before further policy adaptation. It keeps Kakuna's ordinary policy frozen, compares
matched fact-only and native-context belief heads, and reserves fresh development rosters for
reconstruction evaluation. A better belief model alone is not a demonstrated playing-strength gain.

## Local setup

Install Python 3.11+, uv, Node.js with npm, and a Rust toolchain. On Windows, building the
Rust dependency also requires the MSVC C++ build tools. Run these commands from the repository root:

```sh
git submodule update --init --recursive
uv sync --locked --extra dev
npm --prefix server/pokemon-showdown ci
npm --prefix server/pokemon-showdown run build
```

In PowerShell, use `npm.cmd` if execution policy blocks `npm.ps1`.

`uv.lock` pins the Python dependencies. The install compiles poke-engine with its Gen 9 feature;
do not substitute a build using its default generation. The locked PyTorch install supports CPU
checks. For GPU training, install a PyTorch build matching your CUDA environment and record its
version in the run manifest.

The default `showdown_direct` backend starts a persistent Node worker through local pipes. It needs
the simulator build, but no running server or account. Node must be on `PATH`; set `FOG_NODE_BINARY`
to an executable path if needed.

Check the installation:

```sh
uv run --no-sync python -m fog.train --help
uv run --no-sync python -m ruff check fog tests
uv run --no-sync python -m pytest -q
```

The suite includes local Showdown battles, exact chance enumeration, model checks, and Rust engine
comparisons. Some tests require optional data or an external server. Do not run the suite alongside
training on a memory-constrained machine.

## Training

See the [Metamon Gen9OU guide](docs/metamon-gen9ou.md)
for bounded, revision-pinned replay imports and their reconstruction limits.

Use `fog.train` for one PPO or behavior-cloning experiment:

```sh
uv run --no-sync python -m fog.train --config configs/set_ar_train.yaml
uv run --no-sync python -m fog.train --config configs/bc_ou.yaml
uv run --no-sync python -m fog.train --config configs/set_ar_train.yaml --set ppo.lr=0.00003
```

These recipes need their referenced data, checkpoints, and safety reports. They are not installation
smoke tests. In particular, the set-AR recipes require current mechanical and replay audits before
training. Read the recipe's `preflight` settings and the diagnostic error for missing reports;
source changes invalidate existing certificates.

`fog.train.TrainingSession` provides the same entry point for Python scripts and notebooks.
For unattended training blocks, evaluation, and checkpoint retention, see
[KAGGLE_TRAINING.md](KAGGLE_TRAINING.md). Windows launchers use the repository's `.venv`:

```powershell
.\fog-train.cmd --help
.\fog-train.cmd doctor --job configs/automated_gen9ou_typed_league.yaml
```

The doctor checks GPU availability, referenced files, the simulator build, and safety reports.
A CPU-only development installation can run tests and display help while failing this GPU-job check.
The `.ps1` launcher is also available where PowerShell scripts are permitted.

Native Kakuna uses separately pinned Metamon and opponent runtimes in isolated environments.
The base installation above does not install those runtimes or download pretrained weights;
follow the [native experiment guide](docs/native-kakuna.md) before preparing a native run.

Historical milestone recipes are in `configs/archive/`. The old `fog.scripts.pipeline` entry point
is no longer supported.

## Repository layout

| Path | Contents |
| --- | --- |
| `fog/env/` | Direct Showdown worker, battle views, Rust backend, team loading |
| `fog/encoding/` | Observations, reveal tracking, public history, mirrored views |
| `fog/models/` | Policy, value, belief models, and encoders |
| `fog/pretrained/` | Native Kakuna adapters, shared supervision, evaluation, and experiment contracts |
| `fog/training/` | PPO, supervised training, distillation, and checkpoint state |
| `fog/search/` | Determinized and experimental public-belief search |
| `fog/eval/` | Battle evaluation and model diagnostics |
| `fog/scripts/` | Data preparation, audits, and training automation |
| `configs/` | Experiment recipes and pinned simulator/data versions |
| `docs/` | Data, native integration, Kaggle workflow, and training-review guides |
| `integrations/` | Foul Play process adapters and upstream attribution/licenses |
| `notebooks/` | Manual Kaggle training and trial notebooks |
| `tests/` | Unit tests and simulator integration tests |
| `server/pokemon-showdown/` | Pinned upstream Git submodule |
| `data/teams/` | Team pastes and packaged pools |
| `runs/experiments/` | Local frozen packages, handoffs, and collected results (ignored by Git) |

Generated lookup tables in `fog/constants/` and reusable team pools are tracked. Raw replays,
downloaded usage data, checkpoints, experiment reports, and training outputs are local inputs
and outputs excluded from version control.

## Data and checkpoints

Public replays reveal only part of a team. Optional reconstruction samples missing own-team fields
for replay policy training and marks them as imputed. Unrevealed opponent fields remain unknown
in chronological policy inputs; belief labels use observed facts, submitted sheets, or simulator
truth with explicit provenance. Ordinary policy inputs must never include the opponent's hidden state.

The team-bank collector downloads published Pokepaste imports, deduplicates them, and validates
them against the pinned Gen 9 OU rules:

```sh
uv run --no-sync python -m fog.scripts.collect_team_bank --target 150 --min-quality 70
```

Sources are configured in `configs/team_sources_gen9ou.yaml`. Inspect the resulting manifest before
training: tournament/resource teams and screened RMT candidates have different provenance, and
claimed ladder ratings are not independently verified. `configs/train_ou_public.yaml` requires at
least 100 teams. `fog.scripts.build_public_team_splits` creates source-stratified train, validation,
and test packages with no exact roster shared across splits.

Pretrained checkpoints are not distributed with the source. Recipes that reference
`artifacts/checkpoints/` require separately supplied compatible models; a fresh clone cannot run
those recipes until their inputs are available. Set the checkpoint and data paths in your recipe,
retain hashes and provenance with each run, and run the preflight checks before training.

## Research boundaries

Showdown is the mechanics authority and the backend for final evaluation. The Rust engine is a
faster approximation used for diagnostics and historical training; engine runs need cross-simulator
checks. A lower training loss is not evidence of stronger battle play.

The [replay-candidate imputer](docs/replay-candidate-imputation.md) uses joint public set evidence
from 12,547 games. Its known-sheet benchmark improves hidden-move reconstruction but regresses on
items; a matched policy pilot remains tied. It is available as an explicit conversion option.

The public-belief solver is experimental. Its direct-Showdown bridge currently uses one private
world per side; multi-candidate hidden-team substitution is not a production inference path.
The learned public-belief value model failed its public-only ablation, and the second search
distillation generation regressed in direct play.

The matched native Kakuna runs completed 1,250 optimizer updates per condition. On 256 crossed
development games per condition against unchanged Kakuna, ordinary fine-tuning scored 48.44%
(exploratory 95% interval 43.75–53.13%) and joint training scored 50.39% (44.92–55.47%). These
results establish neither improvement nor regression. The repeated cloud evaluation used the
same games and does not add an independent cohort. See the
[post-training review](docs/training-review.md) for training exposure and belief-learning evidence.

Changes to observations, belief, search, or training semantics should update the relevant contracts
in [ARCHITECTURE.md](ARCHITECTURE.md) and their acceptance tests.

## Private Kaggle workflows

Both paths freeze code, data, configuration, and checkpoint hashes into private experiment
packages. Changed recipes get new experiment identities. Collected results remain available for
review; successful execution does not promote a model.

### FOG model experiments

Follow [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md) for account setup and data preparation, then use
[the experiment workflow](docs/kaggle-workflow.md) for training, collection, and local evaluation:

```powershell
.\fog-experiment.cmd prepare --label candidate-trial --job configs/automated_gen9ou_candidates_kaggle.yaml --trial
.\fog-experiment.cmd submit RUN_ID
.\fog-experiment.cmd status RUN_ID
.\fog-experiment.cmd collect RUN_ID
```

Replace `RUN_ID` with the ID printed by `prepare`. The trial checks execution with a small decision
budget and still requires the recipe's data and starting checkpoint. The launcher uses `.venv`
and the `fog.scripts.kaggle_experiment` module. Evaluate the returned candidate on a deliberately
selected held-out pool before choosing the next parent checkpoint.

### Native Kakuna experiments

Use `fog.scripts.pretrained_experiment` for native collection, training, and evaluation packages:

```sh
uv run --no-sync python -m fog.scripts.pretrained_experiment prepare --help
uv run --no-sync python -m fog.scripts.pretrained_experiment status RUN_ID
uv run --no-sync python -m fog.scripts.pretrained_experiment collect RUN_ID
```

Preparation requires a stage (`collect`, `train`, or `evaluate`), the checkpoint, an isolated native
Python executable, and the stage's inputs. The [native guide](docs/native-kakuna.md) describes
composition-disjoint team pools, screened published games, exact-team collection, and probability
and data-isolation checks. A separate `prepare --cpu-preflight` recipe exercises the training and
export pipeline before allocating GPU time. Upstream component and checkpoint licenses are listed
in [integrations/THIRD_PARTY.md](integrations/THIRD_PARTY.md).

New GPU training packages require an armed, matched private CPU evaluation queue. Register both
training outputs, prepare and submit the initial CPU evaluation, then enable its data-update
schedule in Kaggle before submitting training. Follow the
[GPU-to-CPU handoff guide](docs/kaggle-evaluation-queue.md) to verify and record the schedule.
Once armed and submitted, the chain can run while the local machine is offline; scheduled updates
can take until the next daily trigger.

Native comparisons preserve teams, seats, seeds, inference settings, and data schedules across
unchanged Kakuna and both trained conditions. Development checkpoint retention recommends keeping
a valid candidate unless its exploratory 95% interval against unchanged Kakuna has an upper bound
below 50%; failed or incomplete evaluations require diagnosis. This is separate from the final
strength gate, which requires at least 1,024 crossed confirmation games, a matched strong-opponent
regression comparison, and the finalist's probability checks. Preserve prior checkpoints and
explicitly select the checkpoint or complete training state for a continuation.

## Evaluation and learning diagnostics

Review every completed training result before choosing another large run. The
[training-review guide](docs/training-review.md) covers data exposure, held-out progress,
uncertainty, optimization, and testable explanations for the next experiment.

For native models, `fog.scripts.pretrained_compare` checks matched runs and recommends development
retention. Frozen CPU evaluation recipes can also benchmark unchanged Kakuna and both trained
conditions against `simpleheuristics` and `maxdamage`. These basic-opponent diagnostics do not
replace final confirmation. `fog.scripts.pretrained_belief_benchmark` measures sampled hidden sets;
`fog.scripts.pretrained_belief_diagnose` compares public marginals and conditional predictions given
verified correct prefixes across saved checkpoints on fixed training and development cohorts.
Conditional accuracy, ancestral sampling quality, and battle strength are separate measurements.
Learned native belief sampling is not connected to search pending supporting benchmarks.
