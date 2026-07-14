"""Pool-id belief gate (M1.5 gate, spec §10): does the belief net identify WHICH pool team the
opponent brought — top-1 over TEAM IDENTITY — by turn 4?

This is NOT the dashboard's attribute-level top-1 (fraction of individual attributes right),
which is a different, weaker statistic — the HANDOFF flagged exactly that confusion. Team
identification is scored generatively: one belief forward at the observation's information state
(the training input convention — hidden attributes feed <UNK>), then each pool candidate's joint
log-probability is the sum of its 54 tokens' log-softmax scores under those per-position
conditionals (see pool_id_logprobs for why candidate-token teacher forcing is wrong).
Prediction = argmax over candidates (uniform pool prior — a constant under uniform team
sampling); correct = it is the opponent's true paste.

Measured on ENGINE games (exact truth from the sampled pastes, no server, view slot order = paste
order so candidate token grids align with the observation by construction): the checkpoint's raw
policy plays a scripted opponent and every learner decision scores the candidates.

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

    python -m fog.eval.pool_id --ckpt runs/m1_5_belief/checkpoints/ppo_final.pt \
        --config configs/m1_5_belief.yaml --pool data/teams/pool_a --games 200
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LEARNER = "p1"
OPPONENT = "p2"
GATE_TURN = 4
GATE_ACCURACY = 0.80

# SLOT_ATTRS column -> single-token belief family (moves handled as the [6,4] block, cols 3..6)
_SCALAR_ATTR_COLS = {"species": 0, "item": 1, "ability": 2, "spread": 7, "tera": 8}


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
    view_np = batch["opp_revealed"]["tokens"][0].detach().cpu().numpy()   # [6, 9]
    cand_aligned, cand_ok = align_truth_to_views(cand_tokens, view_np)    # [C,6,9], [C,6]
    with torch.no_grad():
        _, c_dyn = agent._backbone_pass(batch)                            # c_dyn conditions belief
        out = agent.belief(c_dyn, batch["opp_revealed"])                  # B=1, info-state input
        cand = torch.as_tensor(cand_aligned, dtype=torch.long, device=device)  # [C, 6, 9]

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
    scores = lp.cpu().numpy()
    scores[~cand_ok.all(axis=1)] = -np.inf  # revealed move not in this candidate's moveset
    return scores


def pool_id_joint_logprobs(
    agent: Any, batch: dict, cand_tokens: np.ndarray, *, tail_order: str = "canonical",
    generator=None,
) -> np.ndarray:
    """[C] TRUE joint log-prob of each candidate team (BELIEF_POLICY_PLAN §4.4, the headline score).

    Unlike the fast marginal (`pool_id_logprobs`, one forward with each candidate read off the SAME
    per-position conditionals), this feeds each candidate's own hidden values back into the chain
    (pass-B teacher forcing) and sums the chain-rule factors: log p(hidden = candidate | R) =
    sum_k log p(h_k = cand_k | R, cand_1..cand_{k-1}). One batched forward over the C candidates
    (canonical tail order by default; pass a random `tail_order` + `generator` for the §4.5.3
    order-sensitivity probe). Teacher-forcing the candidate here is CORRECT — pass B trains exactly
    this input pattern (it was a bug only when that pattern was untrained, commit a8a96eb).

    Revealed attrs are delta rows: a candidate consistent with a reveal contributes 0 there, an
    inconsistent one hits the -inf and is eliminated (the hard filter, same as the fast path)."""
    import torch

    from fog.encoding.reveals import align_truth_to_views

    device = next(agent.parameters()).device
    c = cand_tokens.shape[0]
    opp = batch["opp_revealed"]
    view_np = opp["tokens"][0].detach().cpu().numpy()                     # [6, 9]
    cand_aligned, cand_ok = align_truth_to_views(cand_tokens, view_np)    # [C,6,9], [C,6]
    with torch.no_grad():
        _, c_dyn = agent._backbone_pass(batch)                            # c_dyn conditions belief
        s_c = c_dyn.expand(c, -1).contiguous()
        opp_c = {
            "tokens": opp["tokens"].expand(c, -1, -1).contiguous(),
            "masks": {k: v.expand(c, *v.shape[1:]).contiguous()
                      for k, v in (opp.get("masks") or {}).items()},
        }
        cand = torch.as_tensor(cand_aligned, dtype=torch.long, device=device)  # [C, 6, 9]
        out = agent.belief(s_c, opp_c, teacher_values=cand, tail_order=tail_order,
                           generator=generator)

        lp = torch.zeros(c, device=device)
        for fam, col in _SCALAR_ATTR_COLS.items():
            ls = torch.log_softmax(out.logits[fam], dim=-1)               # [C, 6, n]
            tok = cand[:, :, col]                                         # [C, 6]
            got = ls.gather(-1, tok.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            lp += torch.where(tok >= 0, got, torch.zeros_like(got)).sum(-1)
        mv = torch.log_softmax(out.logits["move"], dim=-1)               # [C, 6, 4, n]
        mtok = cand[:, :, 3:7]
        mgot = mv.gather(-1, mtok.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        lp += torch.where(mtok >= 0, mgot, torch.zeros_like(mgot)).sum((-1, -2))
    scores = lp.cpu().numpy()
    scores[~cand_ok.all(axis=1)] = -np.inf
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
            agent, batch, cand_tokens, tail_order="random", generator=gen))
    arr = np.stack(scores)                                                # [M, C]
    true_lp = arr[:, true_index]
    argmax_true = np.mean([int(row.argmax() == true_index) for row in arr])
    finite = true_lp[np.isfinite(true_lp)]
    return {
        "true_logprob_std": float(np.std(finite)) if finite.size else float("nan"),
        "true_logprob_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "argmax_stability": float(argmax_true),
    }


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
    cand_tokens = np.stack([truth_tokens_from_paste(p, tables) for p in pool])  # [C, 6, 9]
    from fog.env.team_truth import assert_uniform_slot_order

    assert_uniform_slot_order(cand_tokens)  # slot-by-slot scoring needs a uniform paste order
    pool_truth = PoolTruth(pool, tables)

    records: list[tuple[int, int, bool, int]] = []  # (game, turn, correct_joint, K_consistent)
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
                records.append((self.game_idx, int(battle.turn),
                                bool(int(joint.argmax()) == self.true_idx), max(1, k_cons)))
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
        except Exception as exc:
            failed += 1
            print(f"[pool_id] game {g} failed and was dropped: {exc!r}", flush=True)
            continue
    if failed > max(1, int(games) // 5):
        raise RuntimeError(f"{failed}/{games} games failed — a defect, not noise. Fix it first.")

    report = aggregate_pool_id(records, games=int(games) - failed, failed=failed,
                               pool_size=len(pool))
    report["joint_fast_argmax_agreement"] = float(np.mean(agree)) if agree else float("nan")
    if os_reports:
        report["order_sensitivity"] = {
            "n": len(os_reports),
            "true_logprob_std_mean": float(np.nanmean([r["true_logprob_std"] for r in os_reports])),
            "argmax_stability_mean": float(np.mean([r["argmax_stability"] for r in os_reports])),
        }
    print(f"[pool_id] joint-vs-fast argmax agreement: "
          f"{report['joint_fast_argmax_agreement']:.3f}", flush=True)
    if "order_sensitivity" in report:
        os_ = report["order_sensitivity"]
        print(f"[pool_id] order-sensitivity (n={os_['n']}): true-logprob std "
              f"{os_['true_logprob_std_mean']:.3f}, argmax stability "
              f"{os_['argmax_stability_mean']:.3f}", flush=True)
    print(f"[pool_id] {report['games']} games, {report['decisions']} scored decisions "
          f"(pool {report['pool_size']}, chance {report['chance']:.3f})")
    print("[pool_id] top-1 by turn: "
          + "  ".join(f"t{t}{'+' if t == 10 else ''}:{a:.2f}"
                      for t, a in report["curve_by_turn"].items()))
    if report["ceiling_by_turn"]:
        print("[pool_id] hard-filter ceiling (1/K_consistent) by turn: "
              + "  ".join(f"t{t}{'+' if t == 10 else ''}:{a:.2f}"
                          for t, a in report["ceiling_by_turn"].items()))
    print(f"[pool_id] GATE (identified by turn {GATE_TURN}, >= {GATE_ACCURACY:.2f}): "
          f"{report['gate_accuracy']:.3f} over {report['gate_games']} games -> "
          f"{'PASS' if report['gate_passed'] else 'FAIL'}", flush=True)
    return report


def aggregate_pool_id(records: list[tuple[int, int, bool] | tuple[int, int, bool, int]], *,
                      games: int, failed: int, pool_size: int) -> dict:
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

    ap = argparse.ArgumentParser(description="Pool-id belief gate (M1.5, spec §10).")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/m1_5_belief.yaml",
                    help="config that BUILDS this checkpoint (architecture must match)")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path, default=None, help="optional report.json path")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))

    report = run_pool_id(agent, load_pool(args.pool), games=args.games, seed=args.seed,
                         temperature=args.temperature, max_turns=args.max_turns)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[pool_id] report -> {args.out}")


if __name__ == "__main__":
    main()
