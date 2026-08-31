"""Outcome-weighted, team-heldout SimpleHeuristics representation distillation."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from fog.training.heuristic_distill import _POLICY_KEYS
from fog.training.supervised import holdout_groups


def advantage_weight(
    outcome: float,
    donor_value: float,
    *,
    temperature: float = 0.5,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> float:
    """Clipped Monte-Carlo advantage weight for one accepted teacher decision."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    value = math.exp((float(outcome) - float(donor_value)) / temperature)
    return float(min(maximum, max(minimum, value)))


def team_holdout(team_count: int, fraction: float, seed: int) -> set[int]:
    """Deterministic whole-team holdout; no acting-team identity crosses the split."""
    return holdout_groups(range(team_count), fraction, seed)


def collect_rows(
    cfg: dict,
    checkpoint: str,
    *,
    games: int,
    workers: int,
    actor_threads: int,
    seed: int,
) -> tuple[list[dict], int]:
    """Collect accepted teacher rows with outcome, donor value, and donor distribution."""
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
        teacher = bind_fog_player(FogPlayer(agent, temperature=0.0))
        teacher.record_mirror = False
        teacher.record_policy_reference = True
        donor_opponent = bind_fog_player(FogPlayer(agent, temperature=0.0))
        donor_opponent.record_mirror = False
        buffer = RolloutBuffer()
        teacher.attach_buffer(buffer)
        teacher_rng = random.Random(seed ^ (worker_id + 1) * 0x9E3779B1)

        def override(battle, _obs, _output) -> int:
            return map_order_to_action(
                battle, _scripted_order("simpleheuristics", battle, teacher_rng)
            )

        teacher.action_override = override
        rows: list[dict] = []
        client_kwargs = {"showdown_root": showdown_root} if showdown_root else {}
        with ShowdownDirectClient(**client_kwargs) as client:
            for game in assigned:
                selector = game.game_number % 4
                opponent_name = (
                    "simpleheuristics" if selector < 2 else "maxdamage" if selector == 2 else "donor"
                )
                opponent = donor_opponent if opponent_name == "donor" else opponent_name
                play_local_battle(
                    client,
                    {
                        "formatid": cfg["env"].get("battle_format", "gen9ou"),
                        "seed": f"sodium,{game.seed:064x}",
                        "strictChoices": True,
                        "p1": {"name": f"Teacher{worker_id}", "team": packed[game.p1_index]},
                        "p2": {"name": "Opponent", "team": packed[game.p2_index]},
                    },
                    teacher,
                    opponent,
                    seed=game.seed & (2**31 - 1),
                    battle_tag=game.battle_tag,
                    max_turns=160,
                )
                if len(buffer._games) != 1:
                    raise RuntimeError("expected exactly one finalized outcome-weighted game")
                finalized = next(iter(buffer._games.values()))
                if finalized.outcome is None:
                    raise RuntimeError("teacher game was not finalized")
                if not (
                    len(finalized.frags)
                    == len(finalized.actions)
                    == len(finalized.values)
                ):
                    raise RuntimeError("finalized teacher row arrays are misaligned")
                for frag, action, value in zip(
                    finalized.frags, finalized.actions, finalized.values, strict=True
                ):
                    reference = frag.get("reference_log_probs")
                    if reference is None:
                        raise RuntimeError("accepted row lacks donor policy reference")
                    keep = {key: frag[key] for key in _POLICY_KEYS if key in frag}
                    rows.append(
                        {
                            "game": int(game.game_number),
                            "team": int(game.p1_index),
                            "opponent": opponent_name,
                            "outcome": float(finalized.outcome),
                            "donor_value": float(value),
                            "action": int(action),
                            "reference_log_probs": np.asarray(reference, dtype=np.float32),
                            "frag": keep,
                        }
                    )
                buffer.clear()
        return rows

    nested = run_workers(assignments, worker)
    rows = [row for worker_rows in nested for row in worker_rows]
    rows.sort(key=lambda row: (row["game"], row["team"]))
    return rows, len(pool)


def train_representation(
    cfg: dict,
    checkpoint: str,
    rows: list[dict],
    *,
    team_count: int,
    epochs: int,
    batch_size: int,
    head_lr: float,
    representation_lr: float,
    kl_coef: float,
    weight_temperature: float,
    seed: int,
    out_dir: Path,
) -> dict:
    """Train policy plus final encoder/fusion blocks with a whole-team holdout."""
    import torch

    from fog.models.agent import FogAgent
    from fog.training.ppo import _masked_kl
    from fog.training.rollout import _stack_frags

    if not rows:
        raise ValueError("no rows collected")
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    donor_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    student = FogAgent.from_tables(cfg).to(device)
    student.load_state_dict(donor_state)

    head_prefix = "policy_value.policy."
    representation_prefixes = (
        "backbone.public_fusion.",
        "backbone.current_state_encoder.head.",
        "backbone.current_state_encoder.field_skip.",
        "backbone.private_state_encoder.head.",
        "backbone.turn_history_encoder.head.",
    )
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    head_parameters = [
        parameter for name, parameter in student.named_parameters() if name.startswith(head_prefix)
    ]
    representation_parameters = [
        parameter
        for name, parameter in student.named_parameters()
        if name.startswith(representation_prefixes)
    ]
    for parameter in head_parameters + representation_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": head_lr},
            {"params": representation_parameters, "lr": representation_lr},
        ],
        weight_decay=1e-4,
    )

    holdout_teams = team_holdout(team_count, 0.1, seed)
    train_indices = np.asarray(
        [i for i, row in enumerate(rows) if int(row["team"]) not in holdout_teams],
        dtype=np.int64,
    )
    hold_indices = np.asarray(
        [i for i, row in enumerate(rows) if int(row["team"]) in holdout_teams],
        dtype=np.int64,
    )
    if not len(train_indices) or not len(hold_indices):
        raise ValueError("whole-team split produced an empty partition")

    raw_weights = np.asarray(
        [
            advantage_weight(
                row["outcome"],
                row["donor_value"],
                temperature=weight_temperature,
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    raw_weights /= max(1e-8, float(raw_weights[train_indices].mean()))

    def make_batch(indices: np.ndarray) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = _stack_frags([rows[int(i)]["frag"] for i in indices], device=device)
        labels = torch.as_tensor(
            [rows[int(i)]["action"] for i in indices], dtype=torch.long, device=device
        )
        reference = torch.as_tensor(
            np.stack([rows[int(i)]["reference_log_probs"] for i in indices]),
            dtype=torch.float32,
            device=device,
        )
        weights = torch.as_tensor(raw_weights[indices], dtype=torch.float32, device=device)
        return batch, labels, reference, weights

    @torch.no_grad()
    def evaluate(indices: np.ndarray) -> dict[str, float]:
        student.eval()
        nll_sum = weighted_sum = weight_sum = correct = kl_sum = 0.0
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            batch, labels, reference, weights = make_batch(part)
            output = student(batch)
            nll = -output.log_probs.gather(1, labels[:, None]).squeeze(1)
            nll_sum += float(nll.sum())
            weighted_sum += float((weights * nll).sum())
            weight_sum += float(weights.sum())
            correct += float((output.log_probs.argmax(-1) == labels).sum())
            kl_sum += float(_masked_kl(output.log_probs, reference).sum())
        return {
            "nll": nll_sum / len(indices),
            "weighted_nll": weighted_sum / max(1e-8, weight_sum),
            "top1": correct / len(indices),
            "kl_to_donor": kl_sum / len(indices),
        }

    baseline = evaluate(hold_indices)
    history: list[dict] = []
    best_weighted_nll = baseline["weighted_nll"]
    best_epoch = 0
    generator = np.random.default_rng(seed)
    parameters = head_parameters + representation_parameters
    for epoch in range(1, epochs + 1):
        student.train()
        order = generator.permutation(train_indices)
        loss_sum = nll_sum = kl_sum = 0.0
        for start in range(0, len(order), batch_size):
            part = order[start : start + batch_size]
            batch, labels, reference, weights = make_batch(part)
            output = student(batch)
            nll = -output.log_probs.gather(1, labels[:, None]).squeeze(1)
            weighted_nll = (weights * nll).sum() / weights.sum().clamp_min(1e-8)
            kl = _masked_kl(output.log_probs, reference).mean()
            loss = weighted_nll + kl_coef * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite outcome distillation loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite outcome distillation gradient at epoch {epoch}")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(part)
            nll_sum += float(weighted_nll.detach()) * len(part)
            kl_sum += float(kl.detach()) * len(part)
        metrics = evaluate(hold_indices)
        metrics.update(
            epoch=epoch,
            train_loss=loss_sum / len(train_indices),
            train_weighted_nll=nll_sum / len(train_indices),
            train_kl=kl_sum / len(train_indices),
        )
        history.append(metrics)
        print(
            f"[outcome-distill] epoch={epoch} train={metrics['train_loss']:.5f} "
            f"hold_weighted_nll={metrics['weighted_nll']:.5f} "
            f"hold_top1={metrics['top1']:.4f} hold_kl={metrics['kl_to_donor']:.5f}",
            flush=True,
        )
        torch.save(student.state_dict(), out_dir / "student_latest.pt")
        if metrics["weighted_nll"] < best_weighted_nll:
            best_weighted_nll = metrics["weighted_nll"]
            best_epoch = epoch
            torch.save(student.state_dict(), out_dir / "student_best.pt")

    if best_epoch == 0:
        raise RuntimeError("whole-team heldout weighted NLL did not improve")
    best_state = torch.load(out_dir / "student_best.pt", map_location="cpu", weights_only=False)
    allowed_prefixes = (head_prefix,) + representation_prefixes
    unexpected_changes = [
        name
        for name, donor_tensor in donor_state.items()
        if not name.startswith(allowed_prefixes)
        and not torch.equal(donor_tensor.cpu(), best_state[name].cpu())
    ]
    if unexpected_changes:
        raise RuntimeError(f"frozen tensors changed: {unexpected_changes[:8]}")

    outcomes = {}
    for opponent in ("simpleheuristics", "maxdamage", "donor"):
        selected = [row["outcome"] for row in rows if row["opponent"] == opponent]
        outcomes[opponent] = {
            "games": len({row["game"] for row in rows if row["opponent"] == opponent}),
            "row_mean_outcome": float(np.mean(selected)) if selected else None,
        }
    report = {
        "games": len({row["game"] for row in rows}),
        "rows": len(rows),
        "team_count": team_count,
        "holdout_teams": sorted(holdout_teams),
        "train_rows": len(train_indices),
        "holdout_rows": len(hold_indices),
        "head_lr": head_lr,
        "representation_lr": representation_lr,
        "kl_coef": kl_coef,
        "weight_temperature": weight_temperature,
        "weight_min": float(raw_weights.min()),
        "weight_max": float(raw_weights.max()),
        "baseline": baseline,
        "history": history,
        "best_epoch": best_epoch,
        "best_weighted_nll": best_weighted_nll,
        "outside_allowed_prefixes_byte_identical": True,
        "allowed_trainable_prefixes": list(allowed_prefixes),
        "outcomes": outcomes,
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
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--representation-lr", type=float, default=5e-6)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--weight-temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    started = time.monotonic()
    rows, team_count = collect_rows(
        cfg,
        args.checkpoint,
        games=args.games,
        workers=args.workers,
        actor_threads=args.actor_threads,
        seed=args.seed,
    )
    print(
        f"[outcome-distill] collected rows={len(rows)} games={args.games} "
        f"teams={team_count} elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    report = train_representation(
        cfg,
        args.checkpoint,
        rows,
        team_count=team_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        representation_lr=args.representation_lr,
        kl_coef=args.kl_coef,
        weight_temperature=args.weight_temperature,
        seed=args.seed,
        out_dir=args.out,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
