# FOG

*Playing the fog of war in both directions.*

A small (≤5M param) battle agent for Pokémon Showdown **Gen 9 OU singles** built around
hidden-information play: an edge comes from reading what the opponent conceals while controlling
what your own play reveals. Two research arms share one agent and one evaluation harness:

1. **Belief-space search + distillation (headline track).** The agent carries a learned
   **autoregressive belief model** over the opponent's hidden team and a **self-belief model**
   (the opponent's view of *us*). At decision time, **joint counterfactual search** optimizes the
   K private states the opponent thinks we could be in *together*, coupled through the belief
   update our action would induce — so bluffing and information cost are priced situationally.
   Search play is then **distilled** back into the fast policy (sampled/particle ReBeL for a
   simultaneous-move game whose private state space is too large for exact public belief states).
2. **Explicit information-leakage penalty (baseline arm, M3).** A train-time scalar penalty
   β·ΔI on measured information leakage. Nobody has the comparison "explicit information
   regularization vs implicit concealment from belief-consistent search distillation" — M3 exists
   to make that comparison.

**Design constraints:** no LLM inference during battles; search never runs inside the PPO gradient
loop (search enters training through logged data and distillation, never through the on-policy
rollout); every phase produces a usable artifact even if the project stops there.

## Two simulators, one protocol

- **poke-env + a pinned local Pokémon Showdown server** — ground truth. All headline/final
  numbers are measured here.
- **poke-engine (Rust, in-process)** — the volume backend (`env.backend: engine`): synchronous,
  so N battles step in lockstep and every decision wave is ONE batched forward. Measured
  ~335–350 decisions/sec vs ~25 on the Showdown path (13–14×). Training on it keeps
  `cross_sim.every_decisions > 0`: the same agent is periodically evaluated on BOTH backends and
  training HALTS if the win-rate gap exceeds 3% (an agent that farms engine-only mechanics has
  learned the engine, not Pokémon). Search (joint + determinized) is engine-native and also runs
  on Showdown battles through engine-position reconstruction.

## Layout

```
  server/            # pinned pokemon-showdown checkout — NOT vendored here
  data/              # team pools (pastes tracked: experiment definitions); replays/usage gitignored
  fog/
    constants/       # generated lookup tables pinned to the simulator commit + usage month
    env/             # poke-env wrappers, EngineBattle/EngineRunner (Rust backend), teams
    encoding/        # censored battle state -> tensors (reveal ledger, tracker, mirror)
    models/          # backbone, belief + self-belief, policy/value, oracles
    training/        # BC, self-play PPO, leakage objective, value fine-tune, distill, belief refresh
    search/          # determinized + joint counterfactual search, own-state sources, posterior log
    eval/            # gauntlets (incl. search arms), Elo, oracle gaps, belief/leakage metrics
    scripts/         # pipeline one-liners, scrapers, parsers, pool builders
  tests/             # unit + integration (engine-marked tests skip without poke-engine)
  configs/           # yaml per experiment; every run reproducible from config + seed
```

Experiment notes, issue notes, and implementation plans are maintained locally.
This README is the only tracked project document.

## Stack

Python 3.14 · PyTorch (CUDA) · [poke-env](https://github.com/hsahovic/poke-env) ·
poke-engine (Rust) · a pokemon-showdown server pinned to a specific commit · tensorboard.

## Getting started

```bash
pip install -e ".[dev]"

# Local simulator (needed for Showdown-backend runs, cross-sim checks, and headline evals):
cd server/pokemon-showdown && node pokemon-showdown start --no-security   # port 8000

# (only if re-pinning) regenerate constants + usage tables:
python -m fog.scripts.build_constants --showdown server/pokemon-showdown
python -m fog.scripts.download_usage  --month 2026-06 --rating 1695
python -m fog.scripts.build_team_pools --pool all
python -m fog.scripts.build_generative_pool --n 128 --seed 0
```

## Running things

`fog.scripts.pipeline` is the driver — one line per phase (add `--set key=value` overrides):

```bash
python -m fog.scripts.pipeline m15               # Phase 0: M1.5 belief-supervised PPO (Showdown)
python -m fog.scripts.pipeline gate-bvec         # THE go/no-go: M1.5-vs-M1 paired head-to-head
python -m fog.scripts.pipeline gate-belief       # M1.5 belief gate: pool-id top-1 by turn 4
python -m fog.scripts.pipeline m3-control        # M3 beta=0 control (engine backend)
python -m fog.scripts.pipeline m3-grid           # M3 beta grid x seeds (control must exist)
python -m fog.scripts.pipeline smoke-joint       # live joint-search smoke (no training)
python -m fog.scripts.pipeline log-posteriors    # search-play decision log (SEARCH_PLAN 4.2 / Phase-4 data)
python -m fog.scripts.pipeline finetune-value    # posterior-aware value fine-tune + gate
python -m fog.scripts.pipeline gauntlet-search   # Phase 2: policy/determinized/joint arm gauntlet
python -m fog.scripts.pipeline agreement         # Phase 3: exact-vs-sampled decision agreement
python -m fog.scripts.pipeline distill           # Phase 4: distill search play into the policy
python -m fog.scripts.pipeline refresh-beliefs   # Phase 4: belief consistency retrain
```

Milestone training runs (M-track) and evaluation:

```bash
python -m fog.training.ppo  --config configs/m1_poc.yaml          # M1 baseline (no belief vec)
python -m fog.training.ppo  --config configs/m2_oracle.yaml       # M2 set-oracle ceiling
python -m fog.training.ppo  --config configs/m3_leakage_engine.yaml --set leakage.beta=0.03 ...
python -m fog.training.bc   --config configs/bc_ou.yaml           # M4 behavior cloning
python -m fog.eval.run_eval --config configs/eval_ou.yaml --ckpt CKPT [--lvn N.pt|--readability|--pareto]
```

## Gates

| Milestone / phase | Gate |
|---|---|
| **M0** plumbing | ≥3 battles/sec (measured 13.3) |
| **M1/M1.5** PoC + belief conditioning | vs MaxDamage ≥75%, vs SimpleHeuristics ≥60%; pool-id ≥80% top-1 by turn 4; **b_vec ≥3% win-rate gain over M1 — the go/no-go for both research arms** |
| **M2** set-oracle ceiling | oracle ≥ N-agent; record the gap |
| **M3** leakage grid | β grid on POOL-A + pool_gen, ≥3 seeds/cell; readability + win-rate per cell |
| **SEARCH 4.2** posterior-aware value | aware < blind held-out MSE (**passed 2026-07-11**: 0.4730 vs 0.4901) |
| **SEARCH Phase 2** joint-search validation | joint > determinized AND joint > policy vs a reading opponent / head-to-head, Wilson CIs |
| **SEARCH Phase 3** sampled scalability | exact-vs-sampled decision agreement + gauntlet delta at K ∈ {8..64} |
| **SEARCH Phase 4** distillation | held-out KL(search‖π) shrinking across generations; readability vs the M3 β frontier at matched strength |
| **M4** open OU (BC → self-play) | BC top-1 ≥45%; post-self-play ≥80% vs SimpleHeuristics |
| **M5** final eval | full gauntlet/Elo/closure on Showdown |

## Reproducibility

Every run gets a config hash + seed. The Showdown commit and usage-month snapshot are pinned in
config. A golden-battle integration test must produce byte-identical tensors across refactors.
Engine-backend runs keep the cross-sim divergence control on; headline numbers are Showdown-only.
