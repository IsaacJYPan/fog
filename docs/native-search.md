# Native search engineering reference

The native reference keeps Kakuna checkpoint 34's neural weights and architecture
unchanged. It adds proposal/commit operations and isolated counterfactual histories
to test a search bridge before introducing inferred opponent teams.

`NativePlayer.update(op="propose")` encodes the current decision and returns the
ordinary policy proposal without advancing its live trajectory cache. AMAGO
writes cache tensors in-place, so proposal inference uses a copied cache.
`commit` records the action actually executed, including a search override.
`fork` copies parser, observation-space, cache, action/reward history and random
state, then commits a hypothetical action in that branch. An idle opponent can
be forked without inventing a decision. Branches are closed after evaluation.

The critic readout evaluates all 13 one-hot actions and averages the critic
ensemble at the deployed gamma, then takes the legal-policy expectation. The
training objective multiplies environment rewards by `reward_multiplier`;
the readout divides by that factor so immediate rewards and continuation values
share units. Published Kakuna uses `AggressiveShapedReward`, with +200 for a win,
zero terminal loss bonus, a multiplier of 10, and shaping for HP/faints. Its
critic values are neither bounded outcomes nor calibrated win probabilities.
Ensemble dispersion is not a confidence guarantee. The bootstrap reflects the
native stochastic continuation policy used by the critic; an argmax deployment
screen does not establish its calibration on search-selected states.

`fog.search.native_reference.known_team_action` samples authoritative Showdown
chance branches and evaluates a small set of policy-ranked own alternatives.
The opponent's frozen native reply is selected before the own alternative and
uses its own private request and POV history. Replies and initial chance seeds
are shared across alternatives. Opponent-only replacements resolve before the
acting player's next decision. A terminal branch uses the terminal shaped
reward; other branches use reward plus discounted continuation value.

This coordinator has complete simulator teams. It is an explicitly privileged
engineering diagnostic and is not connected to ordinary Showdown play. A failed
or incomplete grid retains the policy proposal with recorded failures, rather
than dropping failed branches from action averages. Its soft budget depends on
bounded simulator/inference calls; a deployment wrapper still needs a hard
process deadline around the complete decision.

Run the canary with an immutable copy of the earlier native worker as a control:

```powershell
python -m fog.scripts.native_search_canary --checkpoint inputs/kakuna34.pt `
  --pool inputs/development.json --reference-source inputs/native_control_v4.py `
  --out results --games 8 --known-teams-diagnostic
```

The canary compares ordinary probabilities against the frozen control on the
same executed history, checks live parent histories after branch inference,
verifies in-memory weights and checkpoint bytes, and records failures, cleanup
and complete decision latency. Eight crossed games per arm are engineering
coverage, not a strength confirmation. Smaller capped traces are local prefix
checks and must not be reported as completed battle evaluations.

The first cloud canary used three alternatives and two chance draws. It stopped
on a soft-budget timeout after 96 search decisions; ordinary probabilities still
matched the control exactly and the weights were unchanged. Its completed games
are incomplete engineering evidence. The retry defaults to two alternatives and
one draw with a 3.5-second soft budget. These are mechanics-screen settings, not
an assertion that a single draw adequately estimates tactical uncertainty.
Parent-history and cleanup checks are reported independently of other failures.

The reduced retry also stopped on a soft timeout, on decision 115, while history,
cleanup, unchanged weights and probability checks passed. The subsequent diagnostic
instruments fork, value and simulation cost and continues the original policy on
explicit soft-budget timeouts. It allows at most 5% such fallbacks across complete
games. Other search failures remain fatal and every complete measured decision
must still fit five seconds. This is a changed operational gate, not evidence that
latency improved. The timings must guide optimization if fallbacks grow with history.

That profiling run completed all 16 games, but failed the declared fallback gate:
47 of 572 search decisions (8.22%) exceeded the soft budget. Its maximum measured
decision was 4.971 seconds; probability, weight, parent-history and cleanup checks
passed. Forking consumed 430.9 seconds, compared with 265.1 for leaf values and
12.8 for simulator steps. The two crossed matchup blocks do not establish strength.

The next implementation defers the opponent's fork until an opponent-only forced
replacement actually needs another policy prediction. Ordinary and terminal leaves
only read the acting player's history, so eager opponent copies performed no useful
work there. Required forks still copy the complete parser, cache and RNG state;
multiple replacements reuse that branch. Operation counts accompany stage timings.
Serialization alternatives were slower in local synthetic-history tests and were
not adopted. The five-second deadline, 3.5-second soft budget and 5% fallback gate
remain unchanged; a fresh full-game canary must validate the optimization.

The next bridge must construct worlds from own private information and public
evidence, using training-only set donors or a compatible learned decoder. Never
use the hidden chance transcript from `reconstruct_session_strict` as belief
evidence. Compare direct native play and search under matched settings before
claiming a contribution; compare inference methods with the same backbone and
search before attributing a benefit to learned beliefs.
