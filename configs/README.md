# Experiment recipes

Start with the workflow you need:

| Recipe | Use |
| --- | --- |
| `gen9ou_typed_token_heuristic_bc.yaml` | Online imitation of SimpleHeuristics with the typed-token model |
| `gen9ou_typed_token_selfplay.yaml` | PPO from a retained typed-token checkpoint |
| `automated_gen9ou_typed_league.yaml` | Local training blocks with evaluation and checkpoint retention |
| `automated_gen9ou_candidates_kaggle.yaml` | Kaggle PPO with candidate-imputed replay regularization |
| `bc_ou.yaml` | Offline behavior cloning from parsed replay shards |

These are starting points, not pretrained agents. Supply the checkpoint, team pool,
and data paths they name. Local `automated_*` jobs use
`python -m fog.scripts.automated_training run --job <path>`; see the
[Kaggle guide](../KAGGLE_TRAINING.md) for remote jobs. Other training recipes use
`python -m fog.train --config <path>` when they define a PPO or behavior-cloning
section. Data preparation commands live under `fog/scripts/`; reconstruction trainers
live under `fog/training/`.

## Other families

- `default.yaml` supplies shared defaults. `gen9ou_typed_token_trunk.yaml` is the
  base typed-token layout; neither is a complete training job on its own.
- `gen9ou_typed_token_*` variations are research recipes: belief/value heads,
  gradient balancing, priors, continuation runs, and policy controls. Versioned
  recipes preserve the settings used for those comparisons. `*_smoke*`,
  `*_diagnostic*`, and `*_overfit*` are short diagnostic runs.
- `gen9ou_contextual_*` and `gen9ou_flat_autoencoder_trunk.yaml` belong to earlier
  reconstruction models. Keep them for checkpoint comparison and reproduction.
- `set_ar_*` recipes exercise replay belief training and scaling. `g0_*` recipes
  reproduce the older finite-team-pool search baseline; `g0_smoke.yaml` is its
  short plumbing check.
- `train_ou*.yaml` and `eval_ou.yaml` use the earlier three-encoder setup.
- `team_sources_gen9ou*.yaml` are collector inputs, not training configs.
- [`archive/`](archive/README.md) holds superseded milestone recipes.

Paths stay stable because recipes inherit from one another and older run records
refer to them. Inheritance follows each file's `extends` key. Use a new recipe for
changed experimental settings so a recorded run can still be reconstructed.
