# FOG architecture

The current experiments use a shared typed-token model. The earlier three-encoder model and
several search variants remain in the code for checkpoint compatibility and comparison.

## What the agent sees

An observation has the acting player's private team, the public board, and public turn history.
Private state includes the submitted sets and private mechanics such as PP. The board contains
revealed team information, HP, status, boosts, and field conditions. History records completed
turns in order, including actions, damage, reveals, switches, and speed order.

The board has a short, fixed-width encoding; history is capped and sequence-dependent. Keeping
them separate avoids making the current board depend on how much history fits in the cap.
Opponent truth can be a supervised target, a simulator search world, or an oracle evaluation input.
It must not enter the ordinary policy observation.

## Models

### Typed-token model

[`configs/gen9ou_typed_token_trunk.yaml`](configs/gen9ou_typed_token_trunk.yaml) configures
`SharedBattleLatent` in `fog/models/shared_latent.py`. Its 24 tokens, each 128 wide, represent:

- six own-private and six opponent-public Pokémon;
- the field and decision request;
- eight recent history records and one long-history summary;
- one learned global token.

Four attention layers mix the tokens. Each fact passes through a learned encoder. Own and opponent
Pokémon share an encoder with different type and side embeddings; unknown opponent fields keep
their sentinel values and visibility flags. History stops at the decision request.

Belief, value, and policy read the contextual sequence. The policy scores each candidate action
with a shared query using its move or switch destination. It has no absolute candidate-index
embedding, so permuting candidates permutes their logits. Masked visible-state prediction is an
auxiliary loss.

### Earlier models

`fog.models.three_encoder.Backbone` encodes the public board, ordered history, and own private team
separately. Policy and value use the private vector plus fused public context. Opponent belief
also uses a projection of our private vector to interpret damage and speed evidence. Self-belief
runs from the opponent's censored perspective, without access to our hidden tensor.

The flat 2,048-wide autoencoder and sixteen-token contextual model are older reconstruction
experiments. The sixteen-token runs plateaued around 34–36% reconstruction. The flat experiment
uses a 95% per-field reconstruction criterion on a fixed batch, then untouched team and battle
splits. That criterion belongs to those experiments; it is not the typed-token training criterion.

### Team belief

`fog.models.belief.BeliefModel` generates complete opponent teams autoregressively. It clamps known
facts and adds generated fields back into the context while enforcing mechanical legality.
Moves and team slots are unordered for this purpose. Search needs complete, correlated team
candidates, so candidate scores must account for the generation routes that can produce a team.

Native set truth is an integer `[6, 15]` grid:

```text
species, item, ability, move0..move3,
nature, ev_hp, ev_atk, ev_def, ev_spa, ev_spd, ev_spe, tera
```

Older typed checkpoints lack the EV-stat residual heads and identity embedding. Loading that
complete older layout initializes the additions to zero, preserving EV predictions. Partial or
unrelated missing weights still fail strict loading.

Public replays leave nature, EVs, and many set fields unknown. Those remain censored belief
targets; complete-team supervision comes from team pastes or simulator self-play. The optional
replay-policy imputer fills missing own-team fields from a compatible donor, with usage-based
fallbacks. Inferred fields are not exact belief labels. Donor selection excludes the target game,
duplicates, and later-dated games; leave held-out games out when building the bank. See the
[imputation benchmark](docs/replay-candidate-imputation.md) for the split rules and results: it
improved hidden-move reconstruction, regressed on items, and tied the matched policy pilot.

## Search

The intended search solves a two-sided Bayesian game over candidate private teams. It evaluates
joint actions and chance outcomes, then executes the mixed-strategy row for the agent's actual
private team. Solving types together lets them share opponent strategies and public belief
updates, avoiding strategy fusion when the candidate support is adequate.

There are three paths in the repository:

| Path | What it currently does |
| --- | --- |
| `pb_search` / `two_sided` | Experimental candidate grids and regret matching on the Rust engine |
| `direct_two_sided` | Showdown legal actions and exact chance branches, with one private world per side |
| Legacy determinized search | One-ply sampled worlds with a fixed reply policy |

The direct bridge solves simultaneous actions with complete information. It does not yet expand
multiple hidden-team candidates in a mid-battle Showdown state. Transcript reconstruction is
useful for debugging that work, but candidates can consume different irrelevant random calls;
a reconstruction mismatch alone must not discard a plausible team.

The learned public-belief value model failed its public-only ablation, and the second search
distillation generation regressed in direct play. Search remains experimental. Before using it
as a training expert, check information isolation, exact-versus-sampled agreement, and held-out
search-on/off strength.

Leaf values depend on what the opponent can infer about us. The board encoding
can be cached, but posterior-dependent heads need rescoring as beliefs change.
Each simulated player receives its own censored perspective. After a policy
changes substantially, refresh belief-training data and compare sampled search
against exact enumeration on held-out pools.

## Training

Battle reward is terminal and sparse. Typed-token experiments start from a retained imitation/DAgger
checkpoint and test bounded PPO runs with auxiliary belief supervision and optional replay
retention. They have not established a reliable strength gain over that checkpoint.

Belief cross-entropy is a dense auxiliary loss. Because ordinary policy inference does not read
decoded beliefs, its policy benefit needs a retraining ablation. Exact self-play labels, censored
replay facts, and imputed own sheets stay separate in the data.

Reconstruction, value pretraining, imitation, and search distillation are also available. Before
scaling a run, inspect value calibration, action-family suppression, and gradient interactions
under a fixed opponent mix. Compare optional reward shaping with an unshaped terminal-win-rate
control.

The CLI and Python API both use `fog.train.TrainingSession`:

```python
from fog.train import TrainingSession

session = TrainingSession.from_file("configs/bc_ou.yaml")
session.config["bc"]["lr"] = 2e-4
metrics = session.run()
```

The CLI equivalent is `python -m fog.train --config ...`. `fog.scripts.automated_training` runs
blocks of PPO with evaluation and checkpoint retention around them.

## Simulators

Pinned Showdown is the mechanics reference, observation source, and final evaluation backend.
The direct worker runs locally without a server. Poke-engine provides faster approximation
experiments and diagnostics; compare it against Showdown before relying on its results.
