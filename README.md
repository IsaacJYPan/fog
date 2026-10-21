# FOG

A Pokémon Showdown Gen 9 OU research agent. The experiments combine a learned battle policy,
hidden-team prediction, and search under partial information. Battles run against a pinned local
copy of Showdown.

The current work uses a typed-token model. PPO pilots have not yet shown a reliable improvement
over the starting imitation checkpoint, and search is still experimental. There are no pretrained
checkpoints in this repository. [ARCHITECTURE.md](ARCHITECTURE.md) covers the model and the remaining
search limitations.

## Setup

You need Python 3.11+, uv, Node.js with npm, and Rust. Windows also needs the MSVC C++ build tools
for poke-engine. From the repository root:

```sh
git submodule update --init --recursive
uv sync --locked --extra dev
npm --prefix server/pokemon-showdown ci
npm --prefix server/pokemon-showdown run build
```

Use `npm.cmd` in PowerShell if execution policy blocks `npm.ps1`.

The lockfile builds poke-engine with its Gen 9 feature; its default generation is different.
The locked PyTorch build supports CPU development. For GPU training, install the appropriate CUDA
build and record the version with the run.

The default `showdown_direct` backend talks to a persistent local Node worker, so no Showdown server
or account is needed. Node must be on `PATH`, or set `FOG_NODE_BINARY` to its executable.

```sh
uv run --no-sync python -m fog.train --help
uv run --no-sync python -m ruff check fog tests
uv run --no-sync python -m pytest -q
```

The tests include simulator battles and Rust engine comparisons. Some need optional data or an
external server, and the full suite can use a lot of memory.

## Running an experiment

`fog.train` runs PPO or behavior cloning from a YAML recipe:

```sh
uv run --no-sync python -m fog.train --config configs/bc_ou.yaml
uv run --no-sync python -m fog.train --config configs/set_ar_train.yaml
uv run --no-sync python -m fog.train --config configs/set_ar_train.yaml --set ppo.lr=0.00003
```

These need the data and checkpoints named in the recipe. The set-AR recipes also require fresh
mechanical and replay audit reports; check their `preflight` settings. Reports are tied to the
source revision, so code changes can invalidate them.

For replay imports, start with the [Metamon Gen9OU guide](docs/metamon-gen9ou.md). For repeated
training blocks, evaluation, and checkpoint retention, see [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md).
The Windows launcher uses the repository's `.venv`:

```powershell
.\fog-train.cmd --help
.\fog-train.cmd doctor --job configs/automated_gen9ou_typed_league.yaml
```

The doctor checks the job's GPU, input files, simulator build, and audit reports. It will fail a
GPU job on a CPU-only installation even when the development setup works.

[docs/kaggle-workflow.md](docs/kaggle-workflow.md) covers API-launched Kaggle runs and local result
evaluation through `fog-experiment.cmd`. Older milestone recipes live in `configs/archive/`.

## Teams and replay data

The team collector downloads published Pokepastes, deduplicates them, and validates them against
the pinned Gen 9 OU rules:

```sh
uv run --no-sync python -m fog.scripts.collect_team_bank --target 150 --min-quality 70
```

Sources are listed in `configs/team_sources_gen9ou.yaml`. Check the output manifest: tournament
teams and screened RMT posts have different provenance, and posted ladder ratings are unverified.
`configs/train_ou_public.yaml` needs at least 100 teams. `fog.scripts.build_public_team_splits`
creates source-stratified splits without sharing exact rosters.

Replays only reveal part of a team. Policy training can fill missing own-team fields using the
[candidate imputer](docs/replay-candidate-imputation.md), but those fields stay marked as inferred.
Opponent observations stay chronological; hidden opponent truth is reserved for supervised labels,
simulator search worlds, and oracle evaluations.

Generated lookup tables and reusable team pools are tracked. Downloaded replays, usage data,
checkpoints, and run outputs are excluded from Git. Recipes pointing at `artifacts/checkpoints/`
need separately supplied models, with compatible layouts and recorded hashes.

## Finding the code

- `fog/env/`: Showdown worker, Rust backend, battle views, team loading
- `fog/encoding/` and `fog/models/`: observations, history, encoders, policy/value/belief heads
- `fog/training/`: PPO, supervised training, distillation, checkpoint state
- `fog/search/` and `fog/eval/`: search experiments and battle diagnostics
- `fog/scripts/`: data preparation, audits, training automation
- `configs/` and `tests/`: experiment recipes and tests

Showdown is the mechanics reference and final evaluation backend. The faster Rust engine is useful
for diagnostics, but needs cross-simulator checks. Judge changes by held-out battles as well as
training metrics.
