"""Distill authoritative SimpleHeuristics choices into only the policy head.

The collector plays real in-memory Showdown battles.  FogPlayer builds the same observations used
by PPO, but an action override executes poke-env's SimpleHeuristics order after strictly mapping it
back to the project's 14-way action space.  Rows come from the RolloutBuffer so rejected choices
are rolled back by the normal direct-runner retry path.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

_POLICY_KEYS = {
    "entities",
    "entity_ids",
    "field",
    "history",
    "history_ids",
    "history_state",
    "history_len",
    "active_idx",
    "legality_mask",
    "action_features",
    "opp_revealed",
}


def collect_rows(
    cfg: dict,
    checkpoint: str,
    *,
    games: int,
    workers: int,
    actor_threads: int,
    seed: int,
    retain_public_action_features: bool = False,
) -> list[dict]:
    """Collect strict (observation, SimpleHeuristics action) rows."""
    import torch

    from fog.env.actions import map_order_to_action
    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import (
        _scripted_order,
        bind_fog_player,
        play_local_battle,
    )
    from fog.env.teams import load_pool, to_packed
    from fog.models.agent import FogAgent
    from fog.training.direct_parallel import WorkerGame, partition_games, run_workers
    from fog.training.rollout import RolloutBuffer

    pool = load_pool(cfg["env"]["team_pool"])
    packed = [to_packed(paste) for paste in pool]
    schedule_rng = random.Random(seed)
    jobs = [
        WorkerGame(
            worker_id=-1,
            game_number=index,
            seed=schedule_rng.getrandbits(256),
            # Cycle the acting team before reshuffling opponents so all 171 teams are covered.
            p1_index=index % len(pool),
            p2_index=schedule_rng.randrange(len(pool)),
        )
        for index in range(games)
    ]
    assignments = partition_games(jobs, workers)
    showdown_root = cfg.get("env", {}).get("showdown_root")

    def worker(worker_id: int, assigned: list[WorkerGame]) -> list[dict]:
        torch.set_num_threads(max(1, actor_threads))
        agent = FogAgent.from_tables(cfg).cpu()
        agent.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
        agent.eval()
        adapter = bind_fog_player(FogPlayer(agent, temperature=0.0))
        adapter.record_mirror = False
        buffer = RolloutBuffer()
        adapter.attach_buffer(buffer)
        teacher_rng = random.Random(seed ^ (worker_id + 1) * 0x9E3779B1)

        def override(battle, _obs, _output) -> int:
            return map_order_to_action(
                battle, _scripted_order("simpleheuristics", battle, teacher_rng)
            )

        adapter.action_override = override
        rows: list[dict] = []
        client_kwargs = {"showdown_root": showdown_root} if showdown_root else {}
        with ShowdownDirectClient(**client_kwargs) as client:
            for game in assigned:
                opponent = "maxdamage" if game.game_number % 2 == 0 else "simpleheuristics"
                play_local_battle(
                    client,
                    {
                        "formatid": cfg["env"].get("battle_format", "gen9ou"),
                        "seed": f"sodium,{game.seed:064x}",
                        "strictChoices": True,
                        "p1": {"name": f"Teacher{worker_id}", "team": packed[game.p1_index]},
                        "p2": {"name": "Opponent", "team": packed[game.p2_index]},
                    },
                    adapter,
                    opponent,
                    seed=game.seed & (2**31 - 1),
                    battle_tag=game.battle_tag,
                    max_turns=160,
                )
                if len(buffer._games) != 1:
                    raise RuntimeError("expected exactly one finalized teacher game")
                finalized = next(iter(buffer._games.values()))
                if finalized.outcome is None:
                    raise RuntimeError("teacher game was not finalized")
                for frag, action in zip(finalized.frags, finalized.actions, strict=True):
                    policy_frag = {
                        key: frag[key]
                        for key in _POLICY_KEYS
                        if key in frag and key != "action_features"
                    }
                    if retain_public_action_features:
                        from fog.encoding.public_action_features import public_action_features

                        if "action_features" not in frag:
                            raise RuntimeError("public action features were requested but absent")
                        policy_frag["action_features"] = public_action_features(
                            frag["action_features"]
                        )
                    rows.append(
                        {
                            "game": game.game_number,
                            "team_index": game.p1_index,
                            "opponent_team_index": game.p2_index,
                            "opponent": opponent,
                            "frag": policy_frag,
                            "action": int(action),
                        }
                    )
                buffer.clear()
                if len(rows) and (len([g for g in assigned if g.game_number <= game.game_number]) % 10 == 0):
                    print(
                        f"[heuristic-collect] worker={worker_id} "
                        f"through_game={game.game_number} rows={len(rows)}",
                        flush=True,
                    )
        return rows

    nested = run_workers(assignments, worker)
    rows = [row for worker_rows in nested for row in worker_rows]
    rows.sort(key=lambda row: (row["game"], int(row["frag"].get("turn", 0))))
    return rows


def train_policy_head(
    cfg: dict,
    checkpoint: str,
    rows: list[dict],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    out_dir: Path,
) -> dict:
    """Train only ``policy_value.policy`` with a replay-level holdout."""
    import torch

    from fog.models.agent import FogAgent
    from fog.training.rollout import _stack_frags
    from fog.training.supervised import grouped_row_split, set_trainable_prefixes

    if not rows:
        raise ValueError("no heuristic rows were collected")
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    donor_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    student = FogAgent.from_tables(cfg).to(device)
    student.load_state_dict(donor_state)
    parameters = set_trainable_prefixes(student, ("policy_value.policy.",))
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)

    split = grouped_row_split(rows, "game", seed=seed)
    game_ids = list(split.all_groups)
    hold_count = len(split.holdout_groups)
    train_indices = split.train
    hold_indices = split.holdout

    def make_batch(indices: np.ndarray) -> tuple[dict, torch.Tensor]:
        batch = _stack_frags([rows[int(i)]["frag"] for i in indices], device=device)
        labels = torch.as_tensor(
            [rows[int(i)]["action"] for i in indices], dtype=torch.long, device=device
        )
        return batch, labels

    @torch.no_grad()
    def evaluate(indices: np.ndarray) -> dict[str, float]:
        student.eval()
        loss_sum = correct = 0.0
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            batch, labels = make_batch(part)
            output = student(batch)
            nll = -output.log_probs.gather(1, labels[:, None]).squeeze(1)
            loss_sum += float(nll.sum())
            correct += float((output.log_probs.argmax(-1) == labels).sum())
        return {"nll": loss_sum / len(indices), "top1": correct / len(indices)}

    baseline = evaluate(hold_indices)
    history: list[dict] = []
    best_nll = baseline["nll"]
    best_epoch = 0
    generator = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        student.train()
        order = generator.permutation(train_indices)
        train_loss = 0.0
        for start in range(0, len(order), batch_size):
            part = order[start : start + batch_size]
            batch, labels = make_batch(part)
            output = student(batch)
            loss = -output.log_probs.gather(1, labels[:, None]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite imitation loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite policy gradient at epoch {epoch}")
            optimizer.step()
            train_loss += float(loss.detach()) * len(part)
        metrics = evaluate(hold_indices)
        metrics.update(epoch=epoch, train_nll=train_loss / len(train_indices))
        history.append(metrics)
        print(
            f"[heuristic-distill] epoch={epoch} train_nll={metrics['train_nll']:.5f} "
            f"hold_nll={metrics['nll']:.5f} hold_top1={metrics['top1']:.4f}",
            flush=True,
        )
        torch.save(student.state_dict(), out_dir / "student_latest.pt")
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_epoch = epoch
            torch.save(student.state_dict(), out_dir / "student_best.pt")

    if best_epoch == 0:
        raise RuntimeError("held-out imitation NLL did not improve over the donor")
    best_state = torch.load(out_dir / "student_best.pt", map_location="cpu", weights_only=False)
    changed = []
    for name, donor_tensor in donor_state.items():
        if not name.startswith("policy_value.policy.") and not torch.equal(
            donor_tensor.cpu(), best_state[name].cpu()
        ):
            changed.append(name)
    if changed:
        raise RuntimeError(f"non-policy tensors changed during head-only distillation: {changed[:8]}")

    report = {
        "games": len(game_ids),
        "rows": len(rows),
        "train_games": len(game_ids) - hold_count,
        "holdout_games": hold_count,
        "train_rows": len(train_indices),
        "holdout_rows": len(hold_indices),
        "baseline": baseline,
        "history": history,
        "best_epoch": best_epoch,
        "best_nll": best_nll,
        "outside_policy_byte_identical": True,
        "checkpoint": str(out_dir / "student_best.pt"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    from fog.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    started = time.monotonic()
    rows = collect_rows(
        cfg,
        args.checkpoint,
        games=args.games,
        workers=args.workers,
        actor_threads=args.actor_threads,
        seed=args.seed,
    )
    print(
        f"[heuristic-distill] collected rows={len(rows)} games={args.games} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    report = train_policy_head(
        cfg,
        args.checkpoint,
        rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
