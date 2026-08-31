"""Restore a three-encoder policy from the pre-split native-EV teacher."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np


def collect_teacher_rows(
    cfg: dict,
    checkpoint: str,
    *,
    games: int,
    workers: int,
    actor_threads: int,
    seed: int,
) -> list[dict]:
    """Collect exact live teacher distributions on authoritative Showdown observations."""
    import torch

    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import bind_fog_player, play_local_battle
    from fog.env.teams import load_pool, to_packed
    from fog.models.legacy_native_teacher import LegacyNativeTeacher
    from fog.training.direct_parallel import WorkerGame, partition_games, run_workers
    from fog.training.rollout import RolloutBuffer

    pool = load_pool(cfg["env"]["team_pool"])
    packed = [to_packed(paste) for paste in pool]
    rng = random.Random(seed)
    jobs = [
        WorkerGame(
            worker_id=-1,
            game_number=index,
            seed=rng.getrandbits(256),
            p1_index=rng.randrange(len(pool)),
            p2_index=rng.randrange(len(pool)),
        )
        for index in range(games)
    ]
    torch.set_num_threads(max(1, actor_threads))
    assignments = partition_games(jobs, workers)
    showdown_root = cfg.get("env", {}).get("showdown_root")

    def worker(worker_id: int, assigned: list[WorkerGame]) -> list[dict]:
        teacher = LegacyNativeTeacher.from_checkpoint(cfg, checkpoint, device="cpu")
        adapter = bind_fog_player(FogPlayer(teacher, temperature=1.0))
        adapter.record_mirror = False
        buffer = RolloutBuffer()
        adapter.attach_buffer(buffer)
        rows: list[dict] = []

        def observe(tag: str, frag: dict, output) -> None:
            keep = {
                key: value
                for key, value in frag.items()
                if key
                in {
                    "entities",
                    "entity_ids",
                    "field",
                    "history",
                    "history_ids",
                    "history_state",
                    "history_len",
                    "active_idx",
                    "legality_mask",
                }
            }
            rows.append(
                {
                    "game": tag,
                    "frag": keep,
                    "teacher_probs": output.log_probs[0].detach().exp().cpu(),
                    "teacher_value": output.value[0].detach().cpu(),
                }
            )

        adapter.decision_observer = observe
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
                        "p1": {"name": "Teacher", "team": packed[game.p1_index]},
                        "p2": {"name": "Opponent", "team": packed[game.p2_index]},
                    },
                    adapter,
                    opponent,
                    seed=game.seed & (2**31 - 1),
                    battle_tag=game.battle_tag,
                )
                buffer.clear()
        return rows

    nested = run_workers(assignments, worker)
    return [row for worker_rows in nested for row in worker_rows]


def distill_policy(
    cfg: dict,
    student_checkpoint: str,
    rows: list[dict],
    *,
    epochs: int,
    batch_size: int,
    new_lr: float,
    retained_lr: float,
    value_coef: float,
    seed: int,
    out_dir: Path,
) -> dict:
    """Fit new representations to the teacher while belief heads remain frozen."""
    import torch

    from fog.models.agent import FogAgent
    from fog.training.rollout import _stack_frags
    from fog.training.supervised import grouped_row_split

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    student = FogAgent.from_tables(cfg).to(device)
    student.load_state_dict(torch.load(student_checkpoint, map_location=device))
    student.eval()

    high_prefixes = (
        "backbone.current_state_encoder.",
        "backbone.private_state_encoder.",
        "backbone.turn_history_encoder.",
        "backbone.public_fusion.",
    )
    high = [p for name, p in student.named_parameters() if name.startswith(high_prefixes)]
    low = [
        p
        for name, p in student.named_parameters()
        if name.startswith(("backbone.embeds.", "policy_value."))
    ]
    trainable = {id(parameter) for parameter in high + low}
    for parameter in student.parameters():
        parameter.requires_grad_(id(parameter) in trainable)
    optimizer = torch.optim.AdamW(
        [{"params": high, "lr": new_lr}, {"params": low, "lr": retained_lr}]
    )

    split = grouped_row_split(rows, "game", seed=seed)
    games = list(split.all_groups)
    train_indices = split.train
    hold_indices = split.holdout
    generator = np.random.default_rng(seed)

    def batch_for(indices: np.ndarray) -> tuple[dict, torch.Tensor, torch.Tensor]:
        frags = [rows[int(index)]["frag"] for index in indices]
        batch = _stack_frags(frags, device=device)
        probs = torch.stack([rows[int(index)]["teacher_probs"] for index in indices]).to(device)
        values = torch.stack([rows[int(index)]["teacher_value"] for index in indices]).to(device)
        return batch, probs, values

    @torch.no_grad()
    def evaluate(indices: np.ndarray) -> dict[str, float]:
        kl_sum = value_sum = agree_sum = 0.0
        count = 0
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            batch, target, target_value = batch_for(part)
            output = student(batch)
            kl = (
                target
                * (target.clamp_min(1e-12).log() - output.log_probs.clamp_min(-30))
            ).sum(-1)
            kl_sum += float(kl.sum())
            value_sum += float(((output.value - target_value) ** 2).sum())
            agree_sum += float(
                (output.log_probs.argmax(-1) == target.argmax(-1)).float().sum()
            )
            count += len(part)
        return {
            "kl": kl_sum / count,
            "value_mse": value_sum / count,
            "top1_agreement": agree_sum / count,
        }

    before = evaluate(hold_indices)
    history = []
    peak_allocated = peak_reserved = 0
    for epoch in range(1, epochs + 1):
        order = generator.permutation(train_indices)
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            part = order[start : start + batch_size]
            batch, target, target_value = batch_for(part)
            output = student(batch)
            policy_loss = -(target * output.log_probs.clamp_min(-30)).sum(-1).mean()
            value_loss = ((output.value - target_value) ** 2).mean()
            loss = policy_loss + value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(high + low, 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(part)
            if device.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        metrics = evaluate(hold_indices)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss_sum / len(train_indices)
        history.append(metrics)
        print(
            f"[policy-distill] epoch={epoch} train={metrics['train_loss']:.4f} "
            f"hold_kl={metrics['kl']:.4f} top1={metrics['top1_agreement']:.3f} "
            f"value_mse={metrics['value_mse']:.4f}",
            flush=True,
        )
        torch.save(student.state_dict(), out_dir / "student_latest.pt")

    report = {
        "rows": len(rows),
        "games": len(games),
        "train_rows": len(train_indices),
        "holdout_rows": len(hold_indices),
        "epochs": epochs,
        "batch_size": batch_size,
        "new_lr": new_lr,
        "retained_lr": retained_lr,
        "value_coef": value_coef,
        "before": before,
        "after": history[-1],
        "history": history,
        "peak_gpu_allocated_mib": peak_allocated / 2**20,
        "peak_gpu_reserved_mib": peak_reserved / 2**20,
    }
    torch.save(student.state_dict(), out_dir / "student_final.pt")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    import torch

    from fog.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/set_ar_train.yaml")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--actor-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--new-lr", type=float, default=3e-4)
    parser.add_argument("--retained-lr", type=float, default=3e-5)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = collect_teacher_rows(
        cfg,
        args.teacher,
        games=args.games,
        workers=args.workers,
        actor_threads=args.actor_threads,
        seed=args.seed,
    )
    torch.save({"rows": rows}, args.out / "teacher_rows.pt")
    print(
        f"[policy-distill] collected {len(rows)} decisions from {args.games} games "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    report = distill_policy(
        cfg,
        args.student,
        rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        new_lr=args.new_lr,
        retained_lr=args.retained_lr,
        value_coef=args.value_coef,
        seed=args.seed,
        out_dir=args.out,
    )
    print(json.dumps({key: report[key] for key in ("before", "after")}, indent=2))


if __name__ == "__main__":
    main()
