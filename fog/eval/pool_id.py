"""Pool-id belief gate (M1.5 gate, spec §10): does the belief net identify WHICH pool team the
opponent brought — top-1 over TEAM IDENTITY — by turn 4?

This is NOT the dashboard's attribute-level top-1 (fraction of individual attributes right),
which is a different, weaker statistic — the HANDOFF flagged exactly that confusion. Team
identification is scored generatively: each pool candidate's joint log-probability under the
belief model is the sum of its 54 per-attribute log-softmax scores with the candidate's OWN
tokens teacher-forced through the AR chain (the model's factorized joint, conditioned on the
current information state `s`). Prediction = argmax over candidates (uniform pool prior — a
constant under uniform team sampling); correct = it is the opponent's true paste.

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
    (batch is B=1, built the standard way). Teacher-forces each candidate's full token grid; masks
    come from the observation (module docstring)."""
    import torch

    device = next(agent.parameters()).device
    c = cand_tokens.shape[0]
    with torch.no_grad():
        s = agent._backbone_pass(batch)                                   # [1, state_dim]
        masks = {k: m.expand(c, *m.shape[1:]) for k, m in
                 ((batch.get("opp_revealed") or {}).get("masks") or {}).items()}
        cand = torch.as_tensor(cand_tokens, dtype=torch.long, device=device)  # [C, 6, 9]
        out = agent.belief(s.expand(c, -1), {"tokens": cand, "masks": masks})

        # tokens can be -1 (empty team slots on short pastes, or an attribute the truth builder
        # could not resolve): no evidence either way — contribute exactly 0, never crash gather
        lp = torch.zeros(c, device=device)
        for fam, col in _SCALAR_ATTR_COLS.items():
            ls = torch.log_softmax(out.logits[fam], dim=-1)               # [C, 6, n]
            tok = cand[:, :, col]
            got = ls.gather(-1, tok.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            lp += torch.where(tok >= 0, got, torch.zeros_like(got)).sum(-1)
        mv = torch.log_softmax(out.logits["move"], dim=-1)                # [C, 6, 4, n]
        mtok = cand[:, :, 3:7]
        mgot = mv.gather(-1, mtok.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        lp += torch.where(mtok >= 0, mgot, torch.zeros_like(mgot)).sum((-1, -2))
    return lp.cpu().numpy()


def run_pool_id(agent: Any, pool: list[str], *, games: int = 200, seed: int = 0,
                temperature: float = 0.0, max_turns: int = 300, tables=None) -> dict:
    """Play engine games with the agent's raw policy vs MaxDamage and score opponent pool-id at
    every learner decision. Returns the report dict (also printed)."""
    import random

    from fog.encoding.observation import Observation
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_runner import _device, _stack_obs
    from fog.env.engine_scripted import EngineMaxDamagePolicy
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.eval.search_gauntlet import ArmPlayer, _play_game

    if tables is None:
        from fog.constants.tables import TABLES as tables
    rng = random.Random(seed)
    agent.eval()
    cand_tokens = np.stack([truth_tokens_from_paste(p, tables) for p in pool])  # [C, 6, 9]

    records: list[tuple[int, int, bool]] = []  # (game, turn, correct)
    failed = 0

    class _ScoringArm(ArmPlayer):
        """Policy arm that scores pool-id for the opponent's team before each real decision."""

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
                lp = pool_id_logprobs(agent, _stack_obs([obs], _device(agent)), cand_tokens)
                records.append((self.game_idx, int(battle.turn),
                                bool(int(lp.argmax()) == self.true_idx)))
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
    print(f"[pool_id] {report['games']} games, {report['decisions']} scored decisions "
          f"(pool {report['pool_size']}, chance {report['chance']:.3f})")
    print("[pool_id] top-1 by turn: "
          + "  ".join(f"t{t}{'+' if t == 10 else ''}:{a:.2f}"
                      for t, a in report["curve_by_turn"].items()))
    print(f"[pool_id] GATE (identified by turn {GATE_TURN}, >= {GATE_ACCURACY:.2f}): "
          f"{report['gate_accuracy']:.3f} over {report['gate_games']} games -> "
          f"{'PASS' if report['gate_passed'] else 'FAIL'}", flush=True)
    return report


def aggregate_pool_id(records: list[tuple[int, int, bool]], *, games: int, failed: int,
                      pool_size: int) -> dict:
    """(game, turn, correct) records -> the report dict. Records must be appended in decision
    order within each game (the game loop guarantees it); the gate reads each game's LAST
    decision with turn <= GATE_TURN."""
    by_turn: dict[int, list[bool]] = {}
    for _, turn, ok in records:
        by_turn.setdefault(min(turn, 10), []).append(ok)  # 10 = "turn 10+"
    curve = {t: float(np.mean(oks)) for t, oks in sorted(by_turn.items())}

    last_by_game: dict[int, bool] = {}
    for game, turn, ok in records:
        if turn <= GATE_TURN:
            last_by_game[game] = ok
    gate_games = len(last_by_game)
    gate_acc = float(np.mean(list(last_by_game.values()))) if gate_games else float("nan")

    return {
        "games": games, "failed_games": failed, "decisions": len(records),
        "pool_size": pool_size, "chance": 1.0 / pool_size,
        "curve_by_turn": curve,
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
