"""Belief retraining on the CURRENT agent's self-play — SEARCH_PLAN §7's consistency step.

Distillation shifts the policy; the belief and self-belief nets still describe the agent that
played BEFORE distillation, which reintroduces exactly the inconsistency approach A exists to
kill (SEARCH_PLAN §8 risk 4: "skipping it reintroduces..."). This module closes one generation of
that loop: engine self-play with the agent AS IT NOW PLAYS (both sides), then supervised retrain
of ONLY the belief + self-belief nets on the recorded truth labels — policy, value, trunk and
preview untouched, so the agent's play is byte-identical afterward and only its beliefs moved.

Collection reuses the standard recording stack end-to-end (EngineRunner -> RolloutBuffer ->
to_batch): the same frag structure, mirror recording and truth plumbing PPO trains on — no
parallel data path to drift. The holdout is a SEPARATE collection run (disjoint seed), so the
split is game-level by construction (belief targets are shared by every row of a game; a row
split would leak them).

Gate (reported, not raised): held-out belief CE and self-belief CE both <= their pre-refresh
values. A refresh that worsens held-out CE means the new play distribution needs more games, not
that the step should be skipped.

    python -m fog.training.belief_refresh --ckpt runs/distill_g1/distill.pt \
        --pool data/teams/pool_a --out runs/belief_refresh_g1
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _collect(agent: Any, pool: list[str], games: int, *, engine_batch: int, temperature: float,
             seed: int, device) -> dict:
    """Self-play decision batch via the standard recording stack. Opponent = a frozen copy of the
    same agent (true self-play — §7 wants beliefs about the agent that actually plays)."""
    import copy

    from fog.env.engine_runner import EngineRunner, FrozenAgentPolicy
    from fog.training.rollout import RolloutBuffer

    buffer = RolloutBuffer()
    runner = EngineRunner(agent, buffer, pool=pool, batch_size=engine_batch, seed=seed,
                          temperature=temperature)
    opponent = FrozenAgentPolicy(copy.deepcopy(agent).eval(), name="self",
                                 temperature=temperature)
    was_training = agent.training
    agent.eval()
    try:
        runner.play_games(opponent, games)
    finally:
        agent.train(was_training)
    batch = buffer.to_batch(device)
    batch.pop("preview", None)  # per-game rows; irrelevant here and break row-count invariants
    return batch


def _belief_ces(agent: Any, batch: dict, *, chunk: int = 512) -> tuple[float, float, float, float]:
    """Marginal and complete-team chain losses for both belief directions on held-out rows."""
    import torch

    from fog.training.losses import belief_loss, belief_output_loss
    from fog.training.ppo import _index_nested

    was_training = agent.training
    agent.eval()
    n = batch["actions"].shape[0]
    b_tot = sb_tot = bc_tot = sbc_tot = 0.0
    try:
        with torch.no_grad():
            for lo in range(0, n, chunk):
                idx = torch.arange(lo, min(lo + chunk, n), device=batch["actions"].device)
                mb = _index_nested(batch, idx)
                mb["compute_belief_chain"] = True
                out = agent(mb)
                if out.belief_chain is None or out.self_belief_chain is None:
                    raise RuntimeError("held-out refresh evaluation did not produce chain logits")
                w = len(idx) / n
                b_tot += w * float(belief_loss(
                    out.belief.logits, mb["belief_targets"], mb["belief_valid"]))
                sb_tot += w * float(belief_loss(
                    out.self_belief.logits, mb["self_belief_targets"], mb["self_belief_valid"]))
                bc_tot += w * float(belief_output_loss(
                    out.belief_chain, mb["belief_targets"], mb["belief_valid"]))
                sbc_tot += w * float(belief_output_loss(
                    out.self_belief_chain,
                    mb["self_belief_targets"], mb["self_belief_valid"],
                ))
    finally:
        agent.train(was_training)
    return b_tot, sb_tot, bc_tot, sbc_tot


def refresh_beliefs(agent: Any, pool: list[str], *, games: int = 256, holdout_games: int = 64,
                    epochs: int = 3, lr: float = 1e-4, minibatch_size: int = 1024,
                    engine_batch: int = 32, temperature: float = 1.0, seed: int = 0,
                    max_grad_norm: float = 0.5) -> dict:
    """One consistency-loop generation: collect self-play, retrain belief + self-belief only.
    Mutates `agent` in place; returns the report dict."""
    import torch

    from fog.training.losses import belief_loss, belief_output_loss
    from fog.training.ppo import _index_nested

    device = next(agent.parameters()).device
    if agent.belief is None or agent.self_belief is None:
        raise ValueError("refresh_beliefs needs an agent with belief + self-belief nets")

    train_batch = _collect(agent, pool, games, engine_batch=engine_batch,
                           temperature=temperature, seed=seed, device=device)
    hold_batch = _collect(agent, pool, holdout_games, engine_batch=engine_batch,
                          temperature=temperature, seed=seed + 7_919_919, device=device)
    for name, b in (("train", train_batch), ("holdout", hold_batch)):
        if b.get("belief_targets") is None or b.get("self_belief_targets") is None:
            raise RuntimeError(f"{name} collection produced no belief/self-belief labels — "
                               "the recording stack is broken, not sparse")

    before_b, before_sb, before_bc, before_sbc = _belief_ces(agent, hold_batch)

    params = list(agent.belief.parameters()) + list(agent.self_belief.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    n = train_batch["actions"].shape[0]
    was_training = agent.training
    agent.train()  # PPO trained the belief nets under trunk dropout; match that distribution
    train_curve = []
    try:
        for _ in range(int(epochs)):
            perm = torch.randperm(n, generator=gen).to(device)
            total = 0.0
            for lo in range(0, n, int(minibatch_size)):
                idx = perm[lo : lo + int(minibatch_size)]
                mb = _index_nested(train_batch, idx)
                out = agent(mb)
                if out.belief_chain is None or out.self_belief_chain is None:
                    raise RuntimeError("refresh training did not produce complete-team chain logits")
                loss = (belief_loss(out.belief.logits, mb["belief_targets"], mb["belief_valid"])
                        + belief_loss(out.self_belief.logits, mb["self_belief_targets"],
                                      mb["self_belief_valid"])
                        + belief_output_loss(out.belief_chain,
                                             mb["belief_targets"], mb["belief_valid"])
                        + belief_output_loss(out.self_belief_chain,
                                             mb["self_belief_targets"],
                                             mb["self_belief_valid"]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()
                total += float(loss.detach()) * len(idx)
            train_curve.append(total / n)
    finally:
        agent.train(was_training)

    after_b, after_sb, after_bc, after_sbc = _belief_ces(agent, hold_batch)
    report = {
        "games": int(games), "holdout_games": int(holdout_games),
        "n_train": int(n), "n_holdout": int(hold_batch["actions"].shape[0]),
        "epochs": int(epochs), "lr": float(lr), "temperature": float(temperature),
        "seed": int(seed),
        "belief_ce_before": before_b, "belief_ce_after": after_b,
        "self_belief_ce_before": before_sb, "self_belief_ce_after": after_sb,
        "belief_chain_ce_before": before_bc, "belief_chain_ce_after": after_bc,
        "self_belief_chain_ce_before": before_sbc,
        "self_belief_chain_ce_after": after_sbc,
        "train_loss_first": train_curve[0], "train_loss_last": train_curve[-1],
        "improved": (after_b <= before_b and after_sb <= before_sb
                     and after_bc <= before_bc and after_sbc <= before_sbc),
    }
    print(f"[belief_refresh] held-out CE: belief marginal {before_b:.3f} -> {after_b:.3f}, "
          f"chain {before_bc:.3f} -> {after_bc:.3f} | self marginal "
          f"{before_sb:.3f} -> {after_sb:.3f}, chain {before_sbc:.3f} -> {after_sbc:.3f} | "
          f"{'improved' if report['improved'] else 'NOT improved (more games?)'}", flush=True)
    return report


# ------------------------------------------------------------------------------------------ CLI
def main() -> None:
    import argparse

    import torch

    from fog.config import load_config_with_args
    from fog.env.teams import load_pool
    from fog.training.value_finetune import load_posterior_agent

    ap = argparse.ArgumentParser(description="Belief consistency refresh (SEARCH_PLAN §7).")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="the distilled checkpoint (blind ones are expanded)")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    ap.add_argument("--out", type=Path, default=Path("runs/belief_refresh"))
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--holdout-games", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--minibatch-size", type=int, default=1024)
    ap.add_argument("--engine-batch", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = load_posterior_agent(cfg, args.ckpt, device)

    report = refresh_beliefs(
        agent, load_pool(args.pool), games=args.games, holdout_games=args.holdout_games,
        epochs=args.epochs, lr=args.lr, minibatch_size=args.minibatch_size,
        engine_batch=args.engine_batch, temperature=args.temperature, seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), args.out / "belief_refresh.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[belief_refresh] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
