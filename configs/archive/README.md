# Archived experiment recipes

These files reproduce superseded M0–M4 experiments. They are intentionally outside the active
`configs/` directory so a historical ablation is not mistaken for a supported current recipe.
Relative `extends` links inside this directory are preserved. Current runs should start from the
documented configs in the parent directory and `ARCHITECTURE.md`.

- `m0_plumbing.yaml`: obsolete network-runner throughput gate.
- `m1_poc.yaml`, `m1_5_belief.yaml`: confined-pool architecture experiments.
- `m2_oracle.yaml`: perfect-information ceiling experiment.
- `m3_leakage*.yaml`: leakage grids on historical backends/pools.
- `m4_ou_selfplay*.yaml`: early OU self-play recipes.
