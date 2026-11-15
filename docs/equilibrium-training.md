# Self-play that converges: anchored policy updates

Written 2026-11-15. This is a design note for the self-play arm, which is frozen until gate
G3 (ROADMAP section 7 and P5.4). Nothing here is scheduled. It records what to build when the
gate opens, and why, so that the question is not worked out again.

## The problem

Plain self-play, and any loop that repeatedly improves against the latest policy, has no
convergence guarantee in a zero-sum game: the policies can chase each other in cycles. The
play rule in `docs/search-and-belief-design.md` section 3.4, `π' ∝ π · exp(Q/τ)`, iterated,
is exactly such a loop.

## The method

Add a pull toward a frozen copy of the policy, the anchor `ρ`, and move the anchor each time
the policy settles. Per information state the update is

```text
π_new  ∝  [ π · ρ^(αη) · exp(η · Q) ]^(1 / (1 + αη))
```

and as a loss it is `advantage − α · KL(π ‖ ρ) − (1/η) · KL(π ‖ π_old)`. With a fixed anchor
the regularised game has one fixed point and the update converges to it. Setting the anchor
to that fixed point and repeating gives a sequence that approaches a Nash equilibrium of the
original game (Perolat et al., "From Poincaré Recurrence to Convergence in Imperfect
Information Games", ICML 2021; used for Stratego in DeepNash. Sokota et al., "A Unified
Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum
Games", 2022, give the discrete update above and a linear rate for a fixed anchor).

In this repository: `fog/training/ppo.py` already has the anchor term
(`ppo.kl_anchor_coef`, `ppo.bc_anchor_ckpt`), switched off after `ppo.kl_anchor_decisions`.
The change is to keep it on and replace the anchor's weights with the current policy's when
the policy has stopped moving (for example when the KL between checkpoints 200,000 decisions
apart falls below a set value). The first anchor is the replay blueprint. Both seats use the
one network, each on its own view. No averaged network, belief model or search is involved.

## Why self-play and not search

The guarantee needs each action's value averaged over the true posterior of the opponent's
hidden team given what we have seen, under the current policies. Real self-play games are
samples from exactly that, so belief error and strategy fusion never enter. Determinised
search used as the player or as the teacher gives up the guarantee; it keeps the weaker one
in the design document's section 2 (no worse than the blueprint against the blueprint).

## A check on a game that can be solved

`docs/notes/kuhn_anchor.py` (numpy only) runs the update on Kuhn poker with exact action
values. Exploitability, which is 0 at equilibrium, after 100 / 1,000 / 5,000 / 20,000 steps
(run on 2026-11-14, `python docs/notes/kuhn_anchor.py 20000`):

| Rule | 100 | 1,000 | 5,000 | 20,000 |
| --- | ---: | ---: | ---: | ---: |
| Best response to the last policy, repeated | 0.412 | 0.412 | 0.412 | 0.412 |
| No anchor | 0.149 | 0.238 | 0.166 | 0.167 |
| Fixed uniform anchor, α = 0.2 | 0.045 | 0.044 | 0.044 | 0.044 |
| Moving anchor | 0.045 | 0.015 | 0.00001 | 0.00000 |
| Moving anchor, action values with noise of sd 0.3 | 0.106 | 0.047 | 0.032 | 0.012 |
| No anchor, same noise | 0.114 | 0.193 | 0.115 | 0.160 |

The moving anchor reaches the game's value, −1/18. This is a table of twelve information
states, not a network.

## What the guarantee does and does not cover

- It is relative to a team pool. Teams must be drawn from a fixed distribution for the game
  to be defined. The result is "no opponent wins more than about half on average over that
  pool". It says nothing about teams outside it, and it is not the best win rate against weak
  players.
- It is exact only with a table and exact values. With a network and sampled games there is
  no theorem; the evidence is that it worked at Stratego's scale.
- The policy must see its whole information state. The history cap and the summary token are
  approximations.
- Exploitability cannot be computed for this game. The exploiter check (ROADMAP P5.2) gives a
  lower bound only.
- Compute is unmeasured. Brief 06 measured 37.5 decisions per second end to end on a T4; a run
  of this kind needs orders of magnitude more than that buys in a week.
- Whether the per-information-state form used here is covered by the published proof, or
  only the sequence form, was not confirmed. The Kuhn run says it works there.

## Search with equilibrium guarantees

Public-belief search (ReBeL) has them and is ruled out in the design document, section 6,
for the size of the hidden-team space. Approximating the belief with sampled teams from the
autoregressive sampler is what the frozen `pb_search` modules attempt. It solves the game
restricted to the sample: it approaches the real equilibrium as the sample and the value
function improve, with no bound at any practical size, and it costs a solve per decision.
The hidden information is sparse per Pokémon, not per team (ten plausible sets for each of
six Pokémon is a million teams), so a belief factored per Pokémon fits better than a list of
teams. It remains a study for small fixed pools, where exact enumeration can check it.

## If this is built

A brief for it should contain: the anchor refresh behind a key; the Kuhn script turned into a
unit test of the update rule; a run on a small fixed pool from the blueprint; the exploiter
check before and after; and a kill rule (ROADMAP P5.4's: no promotion after 20 million
decisions).
