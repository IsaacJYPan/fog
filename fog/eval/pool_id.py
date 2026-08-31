"""Pool-id belief gate: does the belief net identify WHICH pool team the opponent brought?

This is NOT the dashboard's attribute-level top-1 (fraction of individual attributes right),
which is a different, weaker statistic. Team
identification is scored generatively: one belief forward at the observation's information state
(the training input convention — hidden attributes feed <UNK>), then each pool candidate's joint
log-probability is the sum of its 90 tokens' log-softmax scores under those per-position
conditionals (see pool_id_logprobs for why candidate-token teacher forcing is wrong).
Prediction = argmax over candidates (uniform pool prior — a constant under uniform team
sampling); correct = it is the opponent's true paste.

The claim-bearing evaluator runs on direct, in-memory Pokemon Showdown. The older engine-backed
path remains available as a fast diagnostic, but it must not be used as the mechanics authority.
In both cases exact truth comes from the sampled paste and every learner decision scores all pool
candidates.

Legality masks from the live observation are applied inside the belief forward, exactly as in
training; a candidate whose token is masked impossible scores -inf and is eliminated — on
shared-species pools (POOL-A, pool_gen) the masks are species-keyed and identical across
candidates, so this never misfires. Pools that mix species rely on the species softmax itself to
bury revealed-species mismatches.

Two numbers reported:
  * decision-level top-1 accuracy by turn (the identification curve);
  * game-level "identified by turn 4": the prediction at the game's LAST learner decision with
    turn <= 4 is correct. THE GATE: >= 0.80 of games (games over before any turn-<=4 decision are
    excluded and counted).

    python -m fog.eval.pool_id --ckpt runs/set_ar_overfit8/checkpoints/ppo_final.pt \
        --config configs/set_ar_overfit8.yaml --pool data/teams/pool_a --games 32
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LEARNER = "p1"
OPPONENT = "p2"
GATE_TURN = 4
GATE_ACCURACY = 0.80

# SLOT_ATTRS column -> single-token belief family (moves handled as the [6,4] block, cols 3..6)
_SCALAR_ATTR_COLS = {
    "species": 0, "item": 1, "ability": 2, "nature": 7, "tera": 14,
}


def pool_id_logprobs(agent: Any, batch: dict, cand_tokens: np.ndarray) -> np.ndarray:
    """[C] joint log-prob of each candidate team under the belief model, for ONE observation
    (batch is B=1, built the standard way).

    Scoring runs ONE forward at the observation's INFORMATION STATE (revealed tokens, hidden ->
    <UNK>) — exactly the input convention the model was trained under — and reads every
    candidate's tokens off those same per-position conditionals. Two consequences: candidates
    contradicting a revealed attribute are killed by that position's (sharp) conditional, and the
    chain's conditioning is on the revealed prefix, matching training. Teacher-forcing the
    CANDIDATE's own tokens instead evaluates the net off its training input distribution and
    discriminates nothing — a real bug this replaced: a trained checkpoint scored chance-level
    while its belief loss said otherwise.

    Candidate move subslots are ALIGNED to the observation's packing before the gather (the one
    ordering rule everywhere): the chain clamps revealed moves into the earliest subslots, while
    pastes are paste-order — gathering paste-order tokens used to pick up the move-distinctness
    -inf and eliminate the TRUE candidate whenever a revealed move's paste position != its clamp
    position (the runs/pool_id_m15 flat-curve bug). After alignment, -inf survives only for a
    candidate whose moveset genuinely lacks a revealed move — a real contradiction, mapped to
    -inf deliberately.

    Candidate tokens can be -1 (empty slots on short pastes): no evidence, contribute 0."""
    import torch

    from fog.encoding.reveals import align_truth_to_views

    device = next(agent.parameters()).device
    c = cand_tokens.shape[0]
    view_np = batch["opp_revealed"]["tokens"][0].detach().cpu().numpy()  # [6,15]
    cand_aligned, cand_ok = align_truth_to_views(cand_tokens, view_np)   # [C,6,15], [C,6]
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        belief_context = agent.belief_context(contexts)
        out = agent.belief(belief_context, batch["opp_revealed"])         # B=1, info-state input
        cand = torch.as_tensor(cand_aligned, dtype=torch.long, device=device)  # [C,6,15]

        lp = torch.zeros(c, device=device)
        for fam, col in _SCALAR_ATTR_COLS.items():
            ls = torch.log_softmax(out.logits[fam][0], dim=-1)            # [6, n]
            tok = cand[:, :, col]                                         # [C, 6]
            got = ls.unsqueeze(0).expand(c, -1, -1).gather(
                -1, tok.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            lp += torch.where(tok >= 0, got, torch.zeros_like(got)).sum(-1)
        mv = torch.log_softmax(out.logits["move"][0], dim=-1)             # [6, 4, n]
        mtok = cand[:, :, 3:7]                                            # [C, 6, 4]
        mgot = mv.unsqueeze(0).expand(c, -1, -1, -1).gather(
            -1, mtok.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        lp += torch.where(mtok >= 0, mgot, torch.zeros_like(mgot)).sum((-1, -2))
        ev = torch.log_softmax(out.logits["ev"][0], dim=-1)               # [6,6,64]
        etok = cand[:, :, 8:14]                                           # [C,6,6]
        egot = ev.unsqueeze(0).expand(c, -1, -1, -1).gather(
            -1, etok.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        lp += torch.where(etok >= 0, egot, torch.zeros_like(egot)).sum((-1, -2))
    scores = lp.cpu().numpy()
    scores[~cand_ok.all(axis=1)] = -np.inf  # revealed move not in this candidate's moveset
    return scores


def pool_id_joint_logprobs(
    agent: Any, batch: dict, cand_tokens: np.ndarray, *, tail_order: str = "random",
    generator=None, m_orders: int = 8,
) -> np.ndarray:
    """[C] order-averaged set-autoregressive score for each complete candidate team.

    One route score is the exact product of matched conditionals along one dependency-valid
    generation route. The unordered completed-team probability is a mixture over routes and over
    permutations of each Pokemon's hidden move set. Enumerating that mixture is factorial, so the
    headline score is a deterministic Monte Carlo average over ``m_orders`` routes:

        log q(team | reveals) ~= logmeanexp_r log q(team | route_r, reveals).

    Each route conditional and ancestral sample remains normalized exactly. A canonical route is
    retained only as a diagnostic factorization.
    """
    import torch

    from fog.encoding.reveals import align_truth_to_views

    device = next(agent.parameters()).device
    c = cand_tokens.shape[0]
    opp = batch["opp_revealed"]
    view_np = opp["tokens"][0].detach().cpu().numpy()                    # [6,15]
    cand_aligned, cand_ok = align_truth_to_views(cand_tokens, view_np)   # [C,6,15], [C,6]
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        belief_context = agent.belief_context(contexts)
        candidate_context = belief_context.expand(c, -1).contiguous()
        opp_c = {
            "tokens": opp["tokens"].expand(c, -1, -1).contiguous(),
            "masks": {k: v.expand(c, *v.shape[1:]).contiguous()
                      for k, v in (opp.get("masks") or {}).items()},
        }
        cand = torch.as_tensor(cand_aligned, dtype=torch.long, device=device)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(0)
        draws = 1 if tail_order == "canonical" else max(1, int(m_orders))
        route_scores = [
            agent.belief.path_logprob(
                candidate_context, opp_c, cand, order_mode=tail_order, generator=generator
            )
            for _ in range(draws)
        ]
        lp = torch.logsumexp(torch.stack(route_scores), dim=0) - math.log(draws)
    scores = lp.cpu().numpy()
    view = np.broadcast_to(view_np, cand_aligned.shape)
    scalar_cols = [0, 1, 2, 7, 14]
    scalar_values = view[:, :, scalar_cols]
    scalar_known = scalar_values >= 0
    # <NO_ITEM> is current battle state, while pool candidates carry the submitted/original item.
    # The set-AR scorer deliberately predicts the original item from history instead of treating
    # every real candidate as contradicting a consumed or knocked-off item.
    scalar_known[:, :, 1] &= scalar_values[:, :, 1] != agent.belief.no_item_index
    scalar_match = (
        ~scalar_known
        | (cand_aligned[:, :, scalar_cols] == scalar_values)
    ).all(axis=(1, 2))
    scores[~(cand_ok.all(axis=1) & scalar_match)] = -np.inf
    return scores


def pool_id_order_sensitivity(
    agent: Any, batch: dict, cand_tokens: np.ndarray, true_index: int, *, m_orders: int = 4,
    seed: int = 0,
) -> dict:
    """Order-sensitivity diagnostic (§4.5.3): recompute the joint score under `m_orders` random tail
    orders and report the dispersion of the true candidate's log-prob and how often the argmax stays
    the true candidate. Large dispersion = the learned joint is order-inconsistent (watch it across
    training). Returns {"true_logprob_std", "true_logprob_mean", "argmax_stability"}."""
    import torch

    device = next(agent.parameters()).device
    scores = []
    for i in range(int(m_orders)):
        gen = torch.Generator(device=device).manual_seed(seed + i)
        scores.append(pool_id_joint_logprobs(
            agent, batch, cand_tokens, tail_order="random", generator=gen, m_orders=1))
    arr = np.stack(scores)                                                # [M, C]
    true_lp = arr[:, true_index]
    argmax_true = np.mean([int(row.argmax() == true_index) for row in arr])
    finite = true_lp[np.isfinite(true_lp)]
    return {
        "true_logprob_std": float(np.std(finite)) if finite.size else float("nan"),
        "true_logprob_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "argmax_stability": float(argmax_true),
    }


def _posterior_diagnostic(scores: np.ndarray, true_index: int) -> tuple[float, float]:
    """Return (true-team probability, entropy) after normalizing candidate joint scores."""
    finite = np.isfinite(scores)
    if not finite.any() or not finite[int(true_index)]:
        return 0.0, float("nan")
    top = float(np.max(scores[finite]))
    weights = np.zeros_like(scores, dtype=np.float64)
    weights[finite] = np.exp(scores[finite] - top)
    weights /= weights.sum()
    positive = weights > 0
    entropy = -float(np.sum(weights[positive] * np.log(weights[positive])))
    return float(weights[int(true_index)]), entropy


def _add_posterior_diagnostics(
    report: dict, records: list[tuple[int, int, float, float]]
) -> None:
    """Add calibrated whole-team confidence curves and a game-level turn-4 confidence gate."""
    prob_by_turn: dict[int, list[float]] = {}
    entropy_by_turn: dict[int, list[float]] = {}
    last_by_game: dict[int, tuple[float, float]] = {}
    for game, turn, probability, entropy in records:
        bucket = min(turn, 10)
        prob_by_turn.setdefault(bucket, []).append(probability)
        if math.isfinite(entropy):
            entropy_by_turn.setdefault(bucket, []).append(entropy)
        if turn <= GATE_TURN:
            last_by_game[game] = (probability, entropy)
    report["true_probability_by_turn"] = {
        turn: float(np.mean(values)) for turn, values in sorted(prob_by_turn.items())
    }
    report["posterior_entropy_by_turn"] = {
        turn: float(np.mean(values)) for turn, values in sorted(entropy_by_turn.items())
    }
    last = list(last_by_game.values())
    report["gate_mean_true_probability"] = (
        float(np.mean([value[0] for value in last])) if last else float("nan")
    )
    report["gate_confidently_identified_90"] = (
        float(np.mean([value[0] >= 0.90 for value in last])) if last else float("nan")
    )


def _print_report(report: dict) -> None:
    print(
        f"[pool_id] joint-vs-fast argmax agreement: "
        f"{report['joint_fast_argmax_agreement']:.3f}",
        flush=True,
    )
    if "order_sensitivity" in report:
        os_ = report["order_sensitivity"]
        print(
            f"[pool_id] order-sensitivity (n={os_['n']}): true-logprob std "
            f"{os_['true_logprob_std_mean']:.3f}, argmax stability "
            f"{os_['argmax_stability_mean']:.3f}",
            flush=True,
        )
    print(
        f"[pool_id] {report['games']} games, {report['decisions']} scored decisions "
        f"(pool {report['pool_size']}, chance {report['chance']:.3f}, "
        f"backend {report.get('backend', 'unknown')})"
    )
    print(
        "[pool_id] top-1 by turn: "
        + "  ".join(
            f"t{turn}{'+' if turn == 10 else ''}:{accuracy:.2f}"
            for turn, accuracy in report["curve_by_turn"].items()
        )
    )
    if report["ceiling_by_turn"]:
        print(
            "[pool_id] hard-filter baseline (1/K-consistent) by turn: "
            + "  ".join(
                f"t{turn}{'+' if turn == 10 else ''}:{accuracy:.2f}"
                for turn, accuracy in report["ceiling_by_turn"].items()
            )
        )
    if report.get("true_probability_by_turn"):
        print(
            "[pool_id] mean true-team probability by turn: "
            + "  ".join(
                f"t{turn}{'+' if turn == 10 else ''}:{probability:.2f}"
                for turn, probability in report["true_probability_by_turn"].items()
            )
        )
    print(
        f"[pool_id] uniquely identifiable states: "
        f"{report['identifiable_top1_accuracy']:.3f} over "
        f"{report['identifiable_decisions']} decisions",
        flush=True,
    )
    print(
        f"[pool_id] ambiguous states: top-1 {report['ambiguous_top1_accuracy']:.3f} vs "
        f"uniform-consistent {report['ambiguous_uniform_accuracy']:.3f} over "
        f"{report['ambiguous_decisions']} decisions",
        flush=True,
    )
    if report.get("ambiguous_prediction_counts"):
        print(
            f"[pool_id] ambiguous predicted candidates: "
            f"{report['ambiguous_prediction_counts']}",
            flush=True,
        )
    print(
        f"[pool_id] GATE (identified by turn {GATE_TURN}, >= {GATE_ACCURACY:.2f}): "
        f"{report['gate_accuracy']:.3f} over {report['gate_games']} games -> "
        f"{'PASS' if report['gate_passed'] else 'FAIL'}",
        flush=True,
    )


def run_pool_id(agent: Any, pool: list[str], *, games: int = 200, seed: int = 0,
                temperature: float = 0.0, max_turns: int = 300, tables=None,
                os_sample: int = 64) -> dict:
    """Play engine games with the agent's raw policy vs MaxDamage and score opponent pool-id at
    every learner decision. Headline = the JOINT score (§4.5.1); the fast marginal (§4.5.2) is
    computed alongside for the joint-vs-fast argmax agreement, and order-sensitivity (§4.5.3) is
    sampled on up to `os_sample` decisions. Returns the report dict (also printed)."""
    import random

    from fog.encoding.observation import Observation
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_runner import _device, _stack_obs
    from fog.env.engine_scripted import EngineMaxDamagePolicy
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.eval.search_gauntlet import ArmPlayer, _play_game

    if tables is None:
        from fog.constants.tables import TABLES as tables
    from fog.encoding.reveals import opponent_views_from_battle
    from fog.env.team_truth import PoolTruth

    rng = random.Random(seed)
    agent.eval()
    cand_tokens = np.stack([truth_tokens_from_paste(p, tables) for p in pool])  # [C,6,15]
    from fog.env.team_truth import assert_uniform_slot_order

    assert_uniform_slot_order(cand_tokens)  # slot-by-slot scoring needs a uniform paste order
    pool_truth = PoolTruth(pool, tables)

    # (game, turn, correct_joint, K_consistent, true_index, predicted_index)
    records: list[tuple[int, int, bool, int, int, int]] = []
    posterior_records: list[tuple[int, int, float, float]] = []
    agree: list[bool] = []            # joint and fast argmax pick the same candidate
    os_reports: list[dict] = []       # sampled order-sensitivity diagnostics (§4.5.3)
    failed = 0

    class _ScoringArm(ArmPlayer):
        """Policy arm that scores pool-id for the opponent's team before each real decision.
        Alongside the model's prediction it records K = how many pool teams remain CONSISTENT
        with the opponent's reveals — the hard-filter ceiling (expected top-1 = 1/K with a
        uniform prior): it separates "the information isn't there yet" from "the model fails
        to use it"."""

        def __init__(self, game_idx: int, true_idx: int, **kw):
            super().__init__("policy", agent, tables=tables, **kw)
            self.game_idx = game_idx
            self.true_idx = true_idx

        def choose(self, battle, role):
            legal = np.flatnonzero(battle.action_mask(role))
            if len(legal) > 0:  # score even forced decisions — identification is turn-, not
                obs = Observation.from_battle(  # choice-, driven
                    battle.view(role), tracker=battle.trackers[role]
                )
                b = _stack_obs([obs], _device(agent))
                joint = pool_id_joint_logprobs(agent, b, cand_tokens)     # headline
                fast = pool_id_logprobs(agent, b, cand_tokens)           # reported alongside
                views = opponent_views_from_battle(battle.view(role), tables)
                k_cons = len(pool_truth.consistent_indices(views))
                prediction = int(joint.argmax())
                records.append(
                    (
                        self.game_idx,
                        int(battle.turn),
                        prediction == self.true_idx,
                        max(1, k_cons),
                        self.true_idx,
                        prediction,
                    )
                )
                probability, entropy = _posterior_diagnostic(joint, self.true_idx)
                posterior_records.append(
                    (self.game_idx, int(battle.turn), probability, entropy)
                )
                agree.append(bool(int(joint.argmax()) == int(fast.argmax())))
                if len(os_reports) < os_sample and len(legal) > 1:
                    os_reports.append(pool_id_order_sensitivity(
                        agent, b, cand_tokens, self.true_idx, seed=self.game_idx))
            return super().choose(battle, role)

    for g in range(int(games)):
        my_paste = rng.choice(pool)
        opp_idx = rng.randrange(len(pool))
        battle = EngineBattle(my_paste, pool[opp_idx], seed=rng.randrange(2**31),
                              tables=tables, battle_tag=f"poolid-{g}")
        learner = _ScoringArm(g, opp_idx, temperature=temperature,
                              seed=rng.randrange(2**31))
        opponent = EngineMaxDamagePolicy(tables=tables, seed=rng.randrange(2**31))
        try:
            _play_game(battle, {LEARNER: learner, OPPONENT: opponent}, max_turns=max_turns)
        except Exception as exc:  # noqa: BLE001 - one failed game is reported and bounded below
            failed += 1
            print(f"[pool_id] game {g} failed and was dropped: {exc!r}", flush=True)
            continue
    if failed > max(1, int(games) // 5):
        raise RuntimeError(f"{failed}/{games} games failed — a defect, not noise. Fix it first.")

    report = aggregate_pool_id(records, games=int(games) - failed, failed=failed,
                               pool_size=len(pool))
    report["backend"] = "engine"
    _add_posterior_diagnostics(report, posterior_records)
    report["joint_fast_argmax_agreement"] = float(np.mean(agree)) if agree else float("nan")
    if os_reports:
        report["order_sensitivity"] = {
            "n": len(os_reports),
            "true_logprob_std_mean": float(np.nanmean([r["true_logprob_std"] for r in os_reports])),
            "argmax_stability_mean": float(np.mean([r["argmax_stability"] for r in os_reports])),
        }
    _print_report(report)
    return report


def run_pool_id_direct(
    agent: Any,
    pool: list[str],
    *,
    games: int = 32,
    seed: int = 0,
    temperature: float = 0.0,
    max_turns: int = 300,
    tables=None,
    os_sample: int = 32,
    score_orders: int = 4,
    fmt: str = "gen9ou",
    showdown_root: str | None = None,
    actor_agent: Any | None = None,
) -> dict:
    """Authoritative direct-Showdown whole-team identification evaluation.

    ``actor_agent`` can be held fixed while ``agent`` changes, producing identical battle
    trajectories for checkpoint comparisons. If omitted, the scored agent also acts.
    """
    import random

    from fog.encoding.observation import Observation
    from fog.encoding.reveals import opponent_views_from_battle
    from fog.env.engine_runner import _device, _stack_obs
    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import bind_fog_player, play_local_battle
    from fog.env.team_truth import PoolTruth, assert_uniform_slot_order, truth_tokens_from_paste
    from fog.env.teams import to_packed

    if tables is None:
        from fog.constants.tables import TABLES as tables

    rng = random.Random(seed)
    agent.eval()
    actor_agent = actor_agent or agent
    actor_agent.eval()
    cand_tokens = np.stack([truth_tokens_from_paste(paste, tables) for paste in pool])
    assert_uniform_slot_order(cand_tokens)
    pool_truth = PoolTruth(pool, tables)
    packed = [to_packed(paste) for paste in pool]

    records: list[tuple[int, int, bool, int, int, int]] = []
    posterior_records: list[tuple[int, int, float, float]] = []
    agree: list[bool] = []
    os_reports: list[dict] = []
    failed = 0

    class _ScoringFogPlayer(FogPlayer):
        def __init__(self, game_idx: int, true_idx: int):
            super().__init__(actor_agent, temperature=temperature)
            self.game_idx = game_idx
            self.true_idx = true_idx

        def act(self, battle):
            # Bring history/tracker state to exactly the point FogPlayer.act will consume. Its
            # fed-turn guard makes the immediately following super().act call idempotent.
            _tag, ledger, tracker, fed = self._battle_state(battle)
            tracker.record_private_team(battle)
            self._feed_events(battle, ledger, tracker, fed)
            obs = Observation.from_battle(battle, tracker=tracker)
            batch = _stack_obs([obs], _device(agent))
            joint = pool_id_joint_logprobs(
                agent, batch, cand_tokens, m_orders=max(1, int(score_orders))
            )
            fast = pool_id_logprobs(agent, batch, cand_tokens)
            views = opponent_views_from_battle(battle, tables)
            k_cons = max(1, len(pool_truth.consistent_indices(views)))
            turn = int(getattr(battle, "turn", 0) or 0)
            prediction = int(joint.argmax())
            records.append(
                (
                    self.game_idx,
                    turn,
                    prediction == self.true_idx,
                    k_cons,
                    self.true_idx,
                    prediction,
                )
            )
            probability, entropy = _posterior_diagnostic(joint, self.true_idx)
            posterior_records.append((self.game_idx, turn, probability, entropy))
            agree.append(bool(int(joint.argmax()) == int(fast.argmax())))
            if len(os_reports) < os_sample and np.count_nonzero(obs.legality_mask) > 1:
                os_reports.append(
                    pool_id_order_sensitivity(
                        agent, batch, cand_tokens, self.true_idx, seed=self.game_idx
                    )
                )
            return super().act(battle)

    client_kwargs = {"showdown_root": showdown_root} if showdown_root else {}
    with ShowdownDirectClient(**client_kwargs) as client:
        for game_idx in range(int(games)):
            my_idx = rng.randrange(len(pool))
            true_idx = rng.randrange(len(pool))
            scorer = bind_fog_player(_ScoringFogPlayer(game_idx, true_idx))
            try:
                play_local_battle(
                    client,
                    {
                        "formatid": fmt,
                        "seed": f"sodium,{rng.getrandbits(256):064x}",
                        "strictChoices": True,
                        "p1": {"name": "FOG", "team": packed[my_idx]},
                        "p2": {"name": "Opponent", "team": packed[true_idx]},
                    },
                    scorer,
                    "maxdamage",
                    seed=rng.randrange(2**31),
                    battle_tag=f"battle-poolid-direct-{game_idx}",
                    max_turns=max_turns,
                )
            except Exception as exc:  # noqa: BLE001 - one failed game is reported and bounded below
                failed += 1
                print(
                    f"[pool_id] direct Showdown game {game_idx} failed and was dropped: {exc!r}",
                    flush=True,
                )
    if failed > max(1, int(games) // 5):
        raise RuntimeError(f"{failed}/{games} games failed -- a defect, not noise. Fix it first.")

    report = aggregate_pool_id(
        records, games=int(games) - failed, failed=failed, pool_size=len(pool)
    )
    report["backend"] = "showdown_direct"
    _add_posterior_diagnostics(report, posterior_records)
    report["joint_fast_argmax_agreement"] = float(np.mean(agree)) if agree else float("nan")
    if os_reports:
        report["order_sensitivity"] = {
            "n": len(os_reports),
            "true_logprob_std_mean": float(
                np.nanmean([item["true_logprob_std"] for item in os_reports])
            ),
            "argmax_stability_mean": float(
                np.mean([item["argmax_stability"] for item in os_reports])
            ),
        }
    _print_report(report)
    return report


def aggregate_pool_id(records: list[tuple], *, games: int, failed: int, pool_size: int) -> dict:
    """(game, turn, correct[, K_consistent]) records -> the report dict. Records must be appended
    in decision order within each game (the game loop guarantees it); the gate reads each game's
    LAST decision with turn <= GATE_TURN. When K_consistent is present, the report also carries
    the hard-filter ceiling (expected top-1 = 1/K under a uniform prior over consistent teams)."""
    by_turn: dict[int, list[bool]] = {}
    ceil_by_turn: dict[int, list[float]] = {}
    for rec in records:
        turn, ok = rec[1], rec[2]
        bucket = min(turn, 10)  # 10 = "turn 10+"
        by_turn.setdefault(bucket, []).append(ok)
        if len(rec) > 3:
            ceil_by_turn.setdefault(bucket, []).append(1.0 / max(1, rec[3]))
    curve = {t: float(np.mean(oks)) for t, oks in sorted(by_turn.items())}
    ceiling = {t: float(np.mean(v)) for t, v in sorted(ceil_by_turn.items())}
    identifiable = [rec for rec in records if len(rec) > 3 and rec[3] == 1]
    ambiguous = [rec for rec in records if len(rec) > 3 and rec[3] > 1]
    ambiguous_accuracy = (
        float(np.mean([rec[2] for rec in ambiguous])) if ambiguous else float("nan")
    )
    ambiguous_uniform = (
        float(np.mean([1.0 / rec[3] for rec in ambiguous])) if ambiguous else float("nan")
    )
    indexed_ambiguous = [rec for rec in ambiguous if len(rec) >= 6]
    prediction_counts = {
        str(index): sum(int(rec[5]) == index for rec in indexed_ambiguous)
        for index in range(pool_size)
    }
    true_counts = {
        str(index): sum(int(rec[4]) == index for rec in indexed_ambiguous)
        for index in range(pool_size)
    }

    last_by_game: dict[int, bool] = {}
    for rec in records:
        if rec[1] <= GATE_TURN:
            last_by_game[rec[0]] = rec[2]
    gate_games = len(last_by_game)
    gate_acc = float(np.mean(list(last_by_game.values()))) if gate_games else float("nan")

    return {
        "games": games, "failed_games": failed, "decisions": len(records),
        "pool_size": pool_size, "chance": 1.0 / pool_size,
        "curve_by_turn": curve,
        "ceiling_by_turn": ceiling,
        "identifiable_decisions": len(identifiable),
        "identifiable_top1_accuracy": (
            float(np.mean([rec[2] for rec in identifiable])) if identifiable else float("nan")
        ),
        "ambiguous_decisions": len(ambiguous),
        "ambiguous_top1_accuracy": ambiguous_accuracy,
        "ambiguous_uniform_accuracy": ambiguous_uniform,
        "ambiguous_excess_accuracy": ambiguous_accuracy - ambiguous_uniform,
        "ambiguous_prediction_counts": prediction_counts,
        "ambiguous_true_counts": true_counts,
        "gate_turn": GATE_TURN, "gate_games": gate_games,
        "gate_accuracy": gate_acc,
        "gate_passed": bool(gate_games) and gate_acc >= GATE_ACCURACY,
    }


# ------------------------------------------------------------------------------------------ CLI
def main() -> None:
    import argparse

    import torch

    from fog.config import load_config_with_args
    from fog.env.teams import load_pool
    from fog.models.agent import FogAgent

    ap = argparse.ArgumentParser(description="Whole-team pool-identification belief gate.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/archive/m1_5_belief.yaml",
                    help="config that BUILDS this checkpoint (architecture must match)")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--actor-ckpt", type=str, default=None,
        help="optional fixed policy checkpoint; keeps trajectories matched while scoring ckpts",
    )
    ap.add_argument(
        "--actor-config", type=str, default=None,
        help="architecture config for --actor-ckpt (defaults to --config)",
    )
    ap.add_argument(
        "--score-orders", type=int, default=4,
        help="Monte Carlo set/move orderings per candidate joint score",
    )
    ap.add_argument(
        "--order-sensitivity-samples", type=int, default=32,
        help="number of decisions receiving the slower order-sensitivity diagnostic",
    )
    ap.add_argument(
        "--backend", choices=("showdown_direct", "engine"), default="showdown_direct",
        help="claim-bearing direct Showdown (default), or approximate engine diagnostic",
    )
    ap.add_argument("--out", type=Path, default=None, help="optional report.json path")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    actor_agent = None
    if args.actor_ckpt:
        actor_cfg = load_config_with_args(args.actor_config or args.config)
        actor_agent = FogAgent.from_tables(actor_cfg).to(device)
        actor_agent.load_state_dict(torch.load(args.actor_ckpt, map_location=device))

    pool = load_pool(args.pool)
    if args.backend == "showdown_direct":
        report = run_pool_id_direct(
            agent,
            pool,
            games=args.games,
            seed=args.seed,
            temperature=args.temperature,
            max_turns=args.max_turns,
            os_sample=args.order_sensitivity_samples,
            score_orders=args.score_orders,
            fmt=cfg.get("env", {}).get("battle_format", "gen9ou"),
            showdown_root=cfg.get("env", {}).get("showdown_root"),
            actor_agent=actor_agent,
        )
    else:
        report = run_pool_id(
            agent,
            pool,
            games=args.games,
            seed=args.seed,
            temperature=args.temperature,
            max_turns=args.max_turns,
            os_sample=args.order_sensitivity_samples,
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[pool_id] report -> {args.out}")


if __name__ == "__main__":
    main()
