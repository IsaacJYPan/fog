# FOG

FOG is a research agent for Pokémon Showdown Gen 9 OU singles. It learns a battle policy,
estimates hidden team information, and studies whether search can improve play under partial
observability. Battles use a pinned local Showdown simulator.

The active experiments use a typed-token model. The earlier three-encoder model and experimental
belief-space search are retained for comparison. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
model contracts and implementation limits.

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

Historical milestone recipes are in `configs/archive/`. The old `fog.scripts.pipeline` entry point
is no longer supported.

## Repository layout

| Path | Contents |
| --- | --- |
| `fog/env/` | Direct Showdown worker, battle views, Rust backend, team loading |
| `fog/encoding/` | Observations, reveal tracking, public history, mirrored views |
| `fog/models/` | Policy, value, belief models, and encoders |
| `fog/training/` | PPO, supervised training, distillation, and checkpoint state |
| `fog/search/` | Determinized and experimental public-belief search |
| `fog/eval/` | Battle evaluation and model diagnostics |
| `fog/scripts/` | Data preparation, audits, and training automation |
| `configs/` | Experiment recipes and pinned simulator/data versions |
| `tests/` | Unit tests and simulator integration tests |
| `server/pokemon-showdown/` | Pinned upstream Git submodule |
| `data/teams/` | Team pastes and packaged pools |

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
distillation generation regressed in direct play. The typed-token PPO pilots have not established
a reliable strength gain over their starting checkpoint.

Changes to observations, belief, search, or training semantics should update the relevant contracts
in [ARCHITECTURE.md](ARCHITECTURE.md) and their acceptance tests.

## Kaggle experiment workflow

See [docs/kaggle-workflow.md](docs/kaggle-workflow.md) for private API-launched GPU training,
result collection, and local evaluation through `fog-experiment.cmd`.
