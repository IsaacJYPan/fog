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
  * Mechanical evidence then zeroes opponent candidates whose sheets contradict the revealed
    tokens (`keep_mask_from_reveals` — move evidence uses SUBSET semantics, so the reveal-order
    vs paste-order move-slot convention cannot bite). Own-side mechanical filtering is not yet
    wired (needs the mirror view; recorded limitation — exact own-pool sources already filter
    reveals internally, so on pools this loses nothing).

The runner requires an own_source with a STABLE candidate count across the battle (the table's
axes are fixed identities, §7.4): the full-pool sources qualify, reveal-filtering subsets do not.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from fog.search.belief_table import JointBeliefTable
from fog.search.pb_grid import public_label
from fog.search.pb_search import pb_search_action

# SLOT_ATTRS order (schema): species, item, ability, move0..3, spread, tera
_MOVE_COLS = slice(3, 7)
_EXACT_COLS = [0, 1, 2, 7, 8]


def keep_mask_from_reveals(revealed_tokens: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """[6, 9] revealed token grid (-1 = hidden) x [J, 6, 9] candidate sheets -> [J] bool: which
    candidates remain mechanically possible. Non-move attributes must match every revealed value
    exactly; revealed moves must be a SUBSET of the candidate slot's move set (order-free, so the
    two coexisting move-subslot conventions — reveal order vs paste order — cannot cause a false
    elimination). Callers own the §7.2 truthful-evidence carve-outs; the current pools contain
    none of the deception mechanics."""
    revealed = np.asarray(revealed_tokens, dtype=np.int64)
    cands = np.asarray(candidates, dtype=np.int64)
    keep = np.ones(len(cands), dtype=bool)
    for j, cand in enumerate(cands):
        ok = True
        for slot in range(revealed.shape[0]):
            for col in _EXACT_COLS:
                v = revealed[slot, col]
                if v >= 0 and cand[slot, col] != v:
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
) -> list[dict]:
    """Play `n_games` engine games; the learner (`role`) decides via pb_search_action in
    persistent mode over `opp_pool_tokens`, the opponent via `opponent_policy` (default: the
    scripted max-damage bot — data-plumbing default, NOT the self-play consistency target).
    Returns the decision rows (and torch.saves {rows, meta} to `out_path` if given).

    Row fields: game_id, turn, role, obs (deployment input), action, action_dist (the
    distillation target), labels1/labels2, sigma1, sigma2, rho, true_index, j_used, sims_run,
    iterations, converged, pruned_mass_max, queried_beliefs, outcome (+1/-1/0 for `role`,
    backfilled at game end). Trivial one-legal-action decisions are executed but not logged
    (no behavioral content); a game hitting `max_decisions` is truncated with outcome 0."""
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
        k = own_source.states(battle, role).k
        table = JointBeliefTable.from_margins(
            np.zeros(k), np.zeros(len(opp_pool_tokens)))
        game_rows: list[dict] = []
        outcome = 0.0
        for _ in range(int(max_decisions)):
            if battle.finished:
                break
            pending = set(battle.pending_sides())
            moves: dict[str, str] = {}
            result = None
            if role in pending:
                result = pb_search_action(
                    agent, battle, role, own_source=own_source,
                    opp_tokens=opp_pool_tokens, table=table,
                    rng=rng, **(search_kwargs or {}))
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
                game_rows.append({
                    "game_id": game_id, "turn": battle.turn, "role": role,
                    "obs": Observation.from_battle(battle.view(role),
                                                   tracker=battle.trackers[role]),
                    "action": result.action, "action_dist": result.action_dist,
                    "labels1": result.labels1, "labels2": result.labels2,
                    "sigma1": result.sigma1, "sigma2": result.sigma2, "rho": result.rho,
                    "true_index": result.true_index, "j_used": result.j_used,
                    "sims_run": result.sims_run, "iterations": result.iterations,
                    "converged": result.converged,
                    "pruned_mass_max": result.pruned_mass_max,
                    "queried_beliefs": result.queried_beliefs,
                    "outcome": None,
                })

            battle.step(moves)

            # ---- §14 step 11: behavior likelihoods (the distributions that acted) ----------
            if solver_row:
                col1 = result.labels1.index(my_label) if my_label in result.labels1 else None
                col2 = (result.labels2.index(opp_label)
                        if opp_label is not None and opp_label in result.labels2 else None)
                table.behavior_step(
                    result.sigma1 if col1 is not None else None,
                    result.avail1 if col1 is not None else None, col1,
                    result.sigma2 if col2 is not None else None,
                    result.avail2 if col2 is not None else None, col2)
            # ---- then mechanical evidence from the updated reveal state -------------------
            obs_after = Observation.from_battle(battle.view(role),
                                                tracker=battle.trackers[role])
            table.mechanical_step(
                keep_opp=keep_mask_from_reveals(obs_after.opp_revealed["tokens"],
                                                opp_pool_tokens))

        if battle.finished:
            outcome = (1.0 if battle.winner == role
                       else -1.0 if battle.winner is not None else 0.0)
        for row in game_rows:
            row["outcome"] = outcome
        all_rows.extend(game_rows)

    if out_path is not None:
        import torch

        torch.save({"rows": all_rows,
                    "meta": {"version": 2, "n_games": int(n_games), "role": role,
                             "seed": seed, "j_pool": len(opp_pool_tokens)}}, out_path)
    return all_rows
