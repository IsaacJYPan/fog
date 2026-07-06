# FOG

*Playing the fog of war in both directions.*

A small (≤5M param) battle agent for Pokémon Showdown **Gen 9 OU singles** that

1. conditions its play on a learned, battle-conditioned **autoregressive belief model** over the
   opponent's hidden team details (seeing *through* the fog), and
2. tests a novel research idea — an explicit **information-leakage penalty** that trains the agent to
   be hard to read, measured against oracle upper bounds (maintaining its *own* fog).

The name refers to the game's core condition: hidden-information play, where an edge comes from
reading what the opponent conceals while controlling what you reveal of your own team.

**Design constraints:** no LLM inference during battles; no search inside the training loop (search is an
optional inference-time add-on); every phase produces a usable artifact even if the project stops there.

> **Status (2026-07-05): code complete and the leakage experiment is VALID** — a review pass found
> and fixed the defect that would have voided M3 (the self-belief measurer was never trained in
> self-play; Φ̂ froze random weights), plus the full list of execution gaps (frozen probe = backbone
> + head pair, final-step ΔI, live watchdogs with control-run baselines, readability/L-vs-N/Pareto
> eval drivers, live team preview, per-battle state, encoder fidelity + golden-battle test, pools at
> 8/16, poke-engine adapter). CUDA torch is installed (note: the GPU is a GTX 1080, not the 3070
> older notes claimed). The remaining work is compute — run the milestones below in order. See
> for the remaining (non-blocking) approximations.

## Layout

```
fog/                 # repo root
  server/            # pinned pokemon-showdown checkout (git submodule) — NOT vendored here
  data/              # replays / parsed parquet / usage stats / team pools (gitignored, see .gitkeep)
  fog/               # the Python package
    constants/       # generated lookup tables (§1)
    env/             # poke-env wrappers, POC formats, vectorized battle runner
    encoding/        # state -> tensors (§2)
    models/          # encoder, belief, policy/value, leakage predictor (§3)
    training/        # BC (§5), self-play PPO (§6), leakage objective (§7)
    search/          # optional root-level determinized evaluation (§8)
    eval/            # gauntlet, Elo, oracle gaps, belief metrics, leakage metrics (§9)
    scripts/         # scrapers, parsers, cluster builders
  tests/             # unit + integration tests
  configs/           # yaml per experiment; every run reproducible from config + seed
```

## Stack

- Python 3.11, PyTorch (CUDA)
- [poke-env](https://github.com/hsahovic/poke-env) for the environment interface
- a local `pokemon-showdown` server **pinned to a specific commit** (recorded in configs)
- polars / parquet for data, tensorboard for logging

## Getting started

One-time setup (constants + usage-derived spreads are already generated and committed under
`fog/constants/generated/`; regenerate only if you re-pin the simulator or usage month):

```bash
pip install -e ".[dev]"                              # torch, poke-env, polars, tensorboard, ...

# The local simulator must be running for anything live (leave it in its own terminal):
cd server/pokemon-showdown && node pokemon-showdown start --no-security   # port 8000

# (only if re-pinning) regenerate constants + usage tables:
python -m fog.scripts.build_constants --showdown server/pokemon-showdown
python -m fog.scripts.download_usage  --month 2026-06 --rating 1695
python -m fog.scripts.build_team_pools --pool all    # validates POOL-A/B vs the pinned server
```

## Run the milestones (spec §10)

Everything below is a single command; each writes checkpoints + a `metrics.csv`/tensorboard dashboard
under `runs/`. Use `--set key=value` for quick overrides.

```bash
# M0  plumbing / throughput gate (≥3 battles/sec; measured 13.3 here)
python -m fog.env.runner    --config configs/m0_plumbing.yaml

# M1  confined-space PoC (shared-core POOL-A), policy/value only
python -m fog.training.ppo  --config configs/m1_poc.yaml
# M1.5 belief-net conditioning on the hidden-pool
python -m fog.training.ppo  --config configs/m1_5_belief.yaml
# M2  oracle ceilings (set / action)
python -m fog.training.ppo  --config configs/m2_oracle.yaml
python -m fog.training.ppo  --config configs/m2_oracle.yaml --set model.oracle=action

# M3  leakage grid — β=0 control FIRST (it provides watchdog baselines), then one run per β
python -m fog.training.ppo  --config configs/m3_leakage.yaml --set leakage.beta=0.0  log.dir=runs/m3_beta0
python -m fog.training.ppo  --config configs/m3_leakage.yaml \
    --set leakage.beta=0.03 leakage.baseline_run=runs/m3_beta0 log.dir=runs/m3_beta003
# M3 headline measurements (§7a/b/c) — one command each:
python -m fog.eval.run_eval --config configs/eval_ou.yaml --ckpt L.pt --lvn N.pt      # L vs N
python -m fog.eval.run_eval --config configs/eval_ou.yaml --ckpt L.pt --readability   # fresh-probe curve
python -m fog.eval.run_eval --config configs/eval_ou.yaml --ckpt any --pareto         # β-grid frontier

# M4  open Gen 9 OU: scrape → parse → BC → self-play fine-tune
python -m fog.scripts.scrape_replays --target-games 150000 --min-rating 1500
python -m fog.scripts.parse_replays  --raw data/raw_replays --out data/parsed
python -m fog.training.bc    --config configs/bc_ou.yaml
python -m fog.training.ppo   --config configs/m4_ou_selfplay.yaml

# straight-to-OU self-play (no BC) on the POOL-B bank
python -m fog.training.ppo   --config configs/train_ou.yaml

# M5  evaluation: gauntlet (Wilson CIs, incl. a set-oracle READING opponent) + oracle-gap closure
python -m fog.eval.run_eval  --config configs/eval_ou.yaml \
    --ckpt runs/m4_ou_selfplay/checkpoints/ppo_final.pt --bc-ckpt runs/bc_ou/checkpoints/bc_final.pt

# poke-engine (Rust) self-play track: verify before training on it (plan §7)
python -m fog.scripts.fuzz_sim --games 200 --pool data/teams/pool_b
```

| Milestone | Gate |
|---|---|
| **M0** Plumbing | ≥3 battles/sec aggregate (measured **13.3** at 8 workers) |
| **M1** PoC (POOL-A), policy/value only | beats MaxDamage ≥75%, SimpleHeuristics ≥60% |
| **M1.5** Belief-net conditioning | ≥80% top-1 team id by turn 4; `b_vec` conditioning ≥3% win-rate gain |
| **M2** Set-/action-oracle ceilings | oracle ≥ N-agent; record gaps |
| **M3** Leakage experiment (β grid) | full §7 grid; readability + win-rate per cell |
| **M4** Open Gen 9 OU (BC → self-play) | BC top-1 ≥45%; post-self-play ≥80% vs SimpleHeuristics |
| **M5** Search add-on + final eval | §8 determinized search; §9 gauntlet/Elo/closure |

## Reproducibility

Every run gets a **config hash + seed**. The Showdown commit and OU banlist/usage-month snapshot are
pinned in config. One fixed golden-battle integration test must produce byte-identical tensors across
refactors. Any metric regression >2σ vs the previous checkpoint halts the run for inspection.
