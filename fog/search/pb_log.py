"""v2 decision logger + game runner for two-sided search (PROPOSAL_IMPL_PLAN Phase F carve-out;
PROJECT_DESIGN_PROPOSAL.md §11 steps 2-5, §14 step 11).

`play_and_log_pb_games` plays whole engine games in which the learner's non-trivial decisions go
through `pb_search_action` in persistent mode, maintains ONE `JointBeliefTable` per battle, and
writes decision rows for the later trainers: the solver's executed distribution is the policy-
distillation target, the root observation is the deployment input, and rho / sigma2 /
queried_beliefs feed the Phase-G value work.

Belief-update discipline after each real turn (§14 step 11):

  * OUR observed action's likelihood column comes from the solver's averaged sigma1 — the
    distribution that actually acted (invariant 19.6). Trivial decisions (one legal action) carry
    no behavioral content and pass None.
  * The OPPONENT's observed public label is looked up in the solver's label space and its
    likelihood column comes from the solver's averaged sigma2 — a MODEL of the opponent, floored
    inside the table (§7.2), honest only insofar as the real opponent resembles the solve; with a
    scripted or policy opponent this is the documented §17.16 approximation. Labels the solver
    never enumerated (no solve this turn) pass None.
  * Mechanical evidence then zeroes candidates on BOTH sides whose sheets contradict the
    revealed tokens (`keep_mask_from_reveals` — move evidence uses SUBSET semantics, so the
    reveal-order vs paste-order move-slot convention cannot bite). The own side uses the mirror
    view (the OPPONENT's censored observation of us — what we have publicly revealed); wired in
    Phase G2 because the stable-axes own source (`StablePoolOwnStates`) no longer filters
    reveals itself. An initial mechanical step runs BEFORE the first decision: both leads are
    public from turn 1, and pb_search must never fork a reveal-inconsistent own candidate (the
    fork's species-matching silently degrades on them).

Phase G2 budget features:

  * **Posterior-mass truncation**: `search_kwargs={"top_k": ..., "top_j": ...}` solves over the
    top candidates by current table mass (zero-mass candidates always dropped — exact). The
    table keeps FULL axes; the result's kept maps route the behavior update so truncated-out
    candidates take the floor likelihood (§7.2's unmodeled-is-not-impossible stance) and rows
    log the kept sets + dropped mass.
  * **Opening-solve cache (L1, exact)**: pass an `OpeningSolveCache`. On a pool, the first
    decision of every game of the same matchup is byte-identical (same pastes, same prior, and
    grid content is fork-seed-independent — pinned by the duplicate-world invariance test), so
    the stored solve is reused EXACTLY and only the executed action is re-sampled from the
    stored distribution with this game's rng. Zero approximation; kills the most expensive
    decision of every game after the first per matchup.

One-sided decisions (PROPOSAL_IMPL_PLAN Phase F2, proposal §10.5) flow through the same path with
no special-casing. When only the learner is pending (a forced replacement with several live bench,
or a pivot force-switch), pb_search solves a degenerate B = 1 grid; `opp_label` is None (the
opponent owed nothing), so the belief step multiplies in ONLY our replacement sigma1 — the
signaling content of the choice — and the row is logged like any other for distillation and value
training. When only the OPPONENT is pending (its solo replacement while we owe nothing), no solve
exists: we log nothing and skip its behavioral likelihood (the honest §7.2 no-model stance, the
documented F2.4 approximation until solver-vs-solver self-play supplies its sigma), applying only
mechanical evidence.

The runner requires an own_source with a STABLE candidate count across the battle (the table's
axes are fixed identities, §7.4): the full-pool sources qualify, reveal-filtering subsets do not.
"""
from __future__ import annotations

import random
from dataclasses import replace as _dc_replace
from typing import Any

import numpy as np

from fog.search.belief_table import JointBeliefTable
from fog.search.pb_grid import public_label
from fog.search.pb_search import N_ACTIONS, pb_search_action

# SLOT_ATTRS order (schema): species, item, ability, move0..3, spread, tera
_MOVE_COLS = slice(3, 7)
_EXACT_COLS = [0, 1, 2, 7, 8]


def keep_mask_from_reveals(revealed_tokens: np.ndarray, candidates: np.ndarray,
                           *, no_item_token: int | None = None) -> np.ndarray:
    """[6, 9] revealed token grid (-1 = hidden) x [J, 6, 9] candidate sheets -> [J] bool: which
    candidates remain mechanically possible. Non-move attributes must match every revealed value
    exactly — with two carve-outs (both real false-elimination bugs the G2 scale smoke caught):

      * a candidate sheet's -1 (paste-unspecified attribute — tera type in practice: the engine
        assigns a default, which becomes public on terastallizing) constrains nothing;
      * a revealed CURRENT item equal to `no_item_token` contradicts nothing — the census records
        item STATE, and a consumed / popped / knocked-off item (Air Balloon in POOL-A team 3)
        legitimately shows `<NO_ITEM>` on a candidate whose sheet has an item. A revealed REAL
        item still eliminates mismatched sheets; item ACQUISITION (Trick and friends — the §7.2
        item-movement carve-out) stays the caller's duty to exclude from evidence.

    Revealed moves must be a SUBSET of the candidate slot's move set (order-free, so the two
    coexisting move-subslot conventions — reveal order vs paste order — cannot cause a false
    elimination). Callers own the remaining §7.2 truthful-evidence carve-outs (Illusion,
    Transform); the current pools contain none of those."""
    revealed = np.asarray(revealed_tokens, dtype=np.int64)
    cands = np.asarray(candidates, dtype=np.int64)
    keep = np.ones(len(cands), dtype=bool)
    for j, cand in enumerate(cands):
        ok = True
        for slot in range(revealed.shape[0]):
            for col in _EXACT_COLS:
                v = revealed[slot, col]
                if v < 0 or cand[slot, col] < 0:
                    continue
                if col == 1 and no_item_token is not None and v == no_item_token:
                    continue
                if cand[slot, col] != v:
                    ok = False
                    break
            if not ok:
                break
            shown = revealed[slot, _MOVE_COLS]
            cand_moves = set(int(m) for m in cand[slot, _MOVE_COLS])
            if any(int(m) not in cand_moves for m in shown if m >= 0):
                ok = False
                break
        keep[j] = ok
    return keep


class OpeningSolveCache:
    """L1 (PROPOSAL_IMPL_PLAN §6.2): exact reuse of the opening solve per pool matchup.

    Valid because the solve is a deterministic function of (public state, candidate sets, rho,
    budget params) and none of those depend on the game seed at the factory-fresh opening state:
    grid content is fork-seed-independent (pinned) and the solver is deterministic. The key
    hashes the matchup truths (they determine the opening public state AND which sigma1 row is
    executable), both candidate token sets, the rounded rho, and the search kwargs. Exact-hit-
    only semantics: any difference is a miss, never an approximation. The runner re-samples the
    executed action from the stored distribution with the CURRENT game's rng, so exact sampling
    of the solved strategy (invariant 19.7) is untouched."""

    def __init__(self) -> None:
        self._store: dict[tuple, Any] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(battle, role: str, own_tokens: np.ndarray, opp_pool_tokens: np.ndarray,
                 rho: np.ndarray, search_kwargs: dict | None) -> tuple:
        opp_role = "p2" if role == "p1" else "p1"
        return (
            role,
            battle.truth_tokens(role).tobytes(),
            battle.truth_tokens(opp_role).tobytes(),
            np.asarray(own_tokens, dtype=np.int64).tobytes(),
            np.asarray(opp_pool_tokens, dtype=np.int64).tobytes(),
            np.round(np.asarray(rho, dtype=np.float64), 12).tobytes(),
            repr(sorted((search_kwargs or {}).items())),
        )

    def get(self, key: tuple, rng: random.Random):
        """Stored result with a freshly sampled action, or None."""
        stored = self._store.get(key)
        if stored is None:
            self.misses += 1
            return None
        self.hits += 1
        action = int(rng.choices(range(N_ACTIONS), weights=stored.action_dist)[0])
        return _dc_replace(stored, action=action)

    def put(self, key: tuple, result: Any) -> None:
        self._store[key] = result


def play_and_log_pb_games(
    agent: Any,
    battle_factory: Any,
    own_source: Any,
    opp_pool_tokens: np.ndarray,
    *,
    n_games: int = 1,
    role: str = "p1",
    opponent_policy: Any | None = None,
    out_path: str | None = None,
    seed: int = 0,
    max_decisions: int = 200,
    search_kwargs: dict | None = None,
    solve_cache: "OpeningSolveCache | None" = None,
) -> list[dict]:
    """Play `n_games` engine games; the learner (`role`) decides via pb_search_action in
    persistent mode over `opp_pool_tokens`, the opponent via `opponent_policy` (default: the
    scripted max-damage bot — data-plumbing default, NOT the self-play consistency target).
    Returns the decision rows (and torch.saves {rows, meta} to `out_path` if given).

    Row fields: game_id, turn, role, obs (deployment input), action, action_dist (the
    distillation target), labels1/labels2, sigma1, sigma2, rho, true_index (our k* on the row's
    KEPT own axis), opp_true_index (the opponent's j* on the row's KEPT opp axis, -1 if off-pool
    or truncated out this decision), own_tokens [K,6,9] / opp_tokens [J,6,9] (the KEPT candidate
    identities rho indexes — the Phase-G value trainer embeds these and labels V[k*, j*] with
    the outcome), own_kept / opp_kept (full-pool indices behind those axes) with
    dropped_own_mass / dropped_opp_mass (G2 truncation accounting), j_used, sims_run,
    iterations, converged, pruned_mass_max, queried_beliefs, sample_beliefs / sample_iteration
    (the G1 random-iteration mid-solve posteriors, §9.3 mitigation 2), outcome (+1/-1/0 for
    `role`, backfilled at game end), truncated (True when the game hit `max_decisions` — its
    outcome 0 is an artifact, not a draw; the value trainer must drop those rows). `meta` also
    carries `opp_pool_tokens` and `search_kwargs`. Trivial one-legal-action decisions are
    executed but not logged (no behavioral content)."""
    from fog.encoding.observation import Observation

    rng = random.Random(seed)
    opp_role = "p2" if role == "p1" else "p1"
    opp_pool_tokens = np.asarray(opp_pool_tokens, dtype=np.int64)

    all_rows: list[dict] = []
    for game_id in range(int(n_games)):
        battle = battle_factory(game_id, rng.randrange(2**31))
        policy = opponent_policy
        if policy is None:
            from fog.env.engine_scripted import EngineMaxDamagePolicy

            policy = EngineMaxDamagePolicy(tables=battle.tables, seed=rng.randrange(2**31))
        own0 = own_source.states(battle, role)
        with np.errstate(divide="ignore"):
            table = JointBeliefTable.from_margins(
                np.log(np.asarray(own0.weights, dtype=np.float64)),
                np.zeros(len(opp_pool_tokens)) - np.log(len(opp_pool_tokens)))
        # j*: the opponent's TRUE team index in the pool the belief axis is built over — the
        # Phase-G value label is V[k*, j*] ~= outcome. -1 when the real opponent team is not in
        # `opp_pool_tokens` (an off-pool opponent); the value trainer drops those rows.
        opp_truth = np.asarray(battle.truth_tokens(opp_role), dtype=np.int64)
        opp_true_index = next(
            (j for j in range(len(opp_pool_tokens))
             if np.array_equal(opp_pool_tokens[j], opp_truth)), -1)
        if opp_true_index < 0:
            print(f"[pb_log] game {game_id}: opponent team not in opp_pool_tokens — "
                  "value rows for this game carry opp_true_index=-1 (droppable).", flush=True)

        from fog.constants.schema import NO_ITEM

        no_item = int(battle.tables.item_index(NO_ITEM))

        def _mechanical(msg: str) -> None:
            """Both-side reveal filtering; the own side uses the opponent's mirror view of us."""
            own_view = Observation.from_battle(battle.view(opp_role),
                                               tracker=battle.trackers[opp_role])
            opp_view = Observation.from_battle(battle.view(role),
                                               tracker=battle.trackers[role])
            keep_own = keep_mask_from_reveals(own_view.opp_revealed["tokens"], own0.tokens,
                                              no_item_token=no_item)
            keep_opp = keep_mask_from_reveals(opp_view.opp_revealed["tokens"], opp_pool_tokens,
                                              no_item_token=no_item)
            if not keep_own[own0.true_index]:
                raise RuntimeError(f"{msg}: the learner's TRUE candidate was mechanically "
                                   "eliminated by its own reveals — a reveal-census bug")
            if opp_true_index >= 0 and not keep_opp[opp_true_index]:
                raise RuntimeError(f"{msg}: the opponent's TRUE pool team was mechanically "
                                   "eliminated — a reveal-census bug")
            table.mechanical_step(keep_own=keep_own, keep_opp=keep_opp)

        # Leads are public before the first decision — apply their evidence now, both so rho is
        # honest and so pb_search never forks a lead-inconsistent own candidate (G2: the fork's
        # species matching silently degrades on reveal-inconsistent worlds).
        _mechanical("initial mechanical step")

        game_rows: list[dict] = []
        outcome = 0.0
        stepped = False                     # False until this game's first battle.step — the
        #                                     factory-fresh state the opening cache keys on
        for _ in range(int(max_decisions)):
            if battle.finished:
                break
            pending = set(battle.pending_sides())
            moves: dict[str, str] = {}
            result = None
            if role in pending:
                cache_key = None
                if solve_cache is not None and not stepped:
                    cache_key = OpeningSolveCache.make_key(
                        battle, role, own0.tokens, opp_pool_tokens,
                        table.normalized(), search_kwargs)
                    result = solve_cache.get(cache_key, rng)
                if result is None:
                    result = pb_search_action(
                        agent, battle, role, own_source=own_source,
                        opp_tokens=opp_pool_tokens, table=table,
                        rng=rng, **(search_kwargs or {}))
                    if cache_key is not None and result.sims_run > 0:
                        solve_cache.put(cache_key, result)
                moves[role] = battle.engine_action(role, result.action)
            if opp_role in pending:
                opp_idx = policy.choose(battle, opp_role)
                moves[opp_role] = battle.engine_action(opp_role, int(opp_idx))
                opp_label = public_label(battle, opp_role, int(opp_idx))
            else:
                opp_label = None

            solver_row = result is not None and result.sims_run > 0
            if solver_row:
                my_label = public_label(battle, role, result.action)
                # j* on THIS row's kept axis (the axes rho/sigma2/opp_tokens live on)
                row_j = -1
                if opp_true_index >= 0:
                    pos = np.flatnonzero(result.opp_kept == opp_true_index)
                    row_j = int(pos[0]) if len(pos) else -1
                game_rows.append({
                    "game_id": game_id, "turn": battle.turn, "role": role,
                    "obs": Observation.from_battle(battle.view(role),
                                                   tracker=battle.trackers[role]),
                    "action": result.action, "action_dist": result.action_dist,
                    "labels1": result.labels1, "labels2": result.labels2,
                    "sigma1": result.sigma1, "sigma2": result.sigma2, "rho": result.rho,
                    "true_index": result.true_index, "opp_true_index": row_j,
                    "own_tokens": result.own_tokens, "opp_tokens": result.opp_tokens,
                    "own_kept": result.own_kept, "opp_kept": result.opp_kept,
                    "dropped_own_mass": result.dropped_own_mass,
                    "dropped_opp_mass": result.dropped_opp_mass,
                    "j_used": result.j_used,
                    "sims_run": result.sims_run, "iterations": result.iterations,
                    "converged": result.converged,
                    "pruned_mass_max": result.pruned_mass_max,
                    "queried_beliefs": result.queried_beliefs,
                    "sample_beliefs": result.sample_beliefs,
                    "sample_iteration": result.sample_iteration,
                    "outcome": None, "truncated": None,
                })

            battle.step(moves)
            stepped = True

            # ---- §14 step 11: behavior likelihoods (the distributions that acted) ----------
            if solver_row:
                col1 = result.labels1.index(my_label) if my_label in result.labels1 else None
                col2 = (result.labels2.index(opp_label)
                        if opp_label is not None and opp_label in result.labels2 else None)
                table.behavior_step(
                    result.sigma1 if col1 is not None else None,
                    result.avail1 if col1 is not None else None, col1,
                    result.sigma2 if col2 is not None else None,
                    result.avail2 if col2 is not None else None, col2,
                    kept1=result.own_kept if col1 is not None else None,
                    kept2=result.opp_kept if col2 is not None else None)
            # ---- then mechanical evidence from the updated reveal state -------------------
            _mechanical(f"turn {battle.turn}")

        if battle.finished:
            outcome = (1.0 if battle.winner == role
                       else -1.0 if battle.winner is not None else 0.0)
        for row in game_rows:
            row["outcome"] = outcome
            row["truncated"] = not battle.finished
        all_rows.extend(game_rows)

    if out_path is not None:
        import torch

        torch.save({"rows": all_rows,
                    "meta": {"version": 2, "n_games": int(n_games), "role": role,
                             "seed": seed, "j_pool": len(opp_pool_tokens),
                             "opp_pool_tokens": opp_pool_tokens,
                             "search_kwargs": dict(search_kwargs or {})}}, out_path)
    return all_rows
