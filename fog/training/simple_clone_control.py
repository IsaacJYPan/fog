"""Exact-private end-to-end cloneability control for SimpleHeuristics.

SimpleHeuristics is the acting policy in authoritative in-memory Showdown battles.  The
student receives the ordinary deployable observation: its exact submitted team and only
chronologically visible opponent information.  The teacher action is stored solely as a
label.  This isolates policy/training capacity from public-replay private-set imputation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fog.training.simple_imitation import split_by_acting_team


def action_family(action: int) -> str:
    if 0 <= int(action) < 4:
        return "move"
    if int(action) < 8:
        return "tera"
    if int(action) < 13:
        return "switch"
    return "forced"


def _unique_parameters(modules) -> list:
    seen: set[int] = set()
    result = []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def train_policy_clone(
    cfg: dict,
    checkpoint: str,
    rows: list[dict],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    holdout_fraction: float,
    seed: int,
    out_dir: Path,
) -> dict:
    import torch

    from fog.models.agent import FogAgent
    from fog.training.bc import _policy_modules
    from fog.training.rollout import _stack_frags

    if not rows:
        raise ValueError("no exact-private SimpleHeuristics rows collected")
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    donor_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    student = FogAgent.from_tables(cfg).to(device)
    student.load_state_dict(donor_state)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    parameters = _unique_parameters(_policy_modules(student))
    parameter_ids = {id(parameter) for parameter in parameters}
    allowed_names = {
        name for name, parameter in student.named_parameters() if id(parameter) in parameter_ids
    }
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)

    train_indices, hold_indices, holdout_teams = split_by_acting_team(
        rows, holdout_fraction=holdout_fraction, seed=seed
    )
    if not len(train_indices) or not len(hold_indices):
        raise ValueError("whole-team cloneability split produced an empty partition")

    def make_batch(indices: np.ndarray):
        batch = _stack_frags([rows[int(index)]["frag"] for index in indices], device=device)
        labels = torch.as_tensor(
            [rows[int(index)]["action"] for index in indices], dtype=torch.long, device=device
        )
        return batch, labels

    @torch.no_grad()
    def evaluate(indices: np.ndarray) -> dict[str, float]:
        student.eval()
        loss_sum = correct = 0.0
        family_total = {family: 0 for family in ("move", "tera", "switch", "forced")}
        family_correct = dict(family_total)
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            batch, labels = make_batch(part)
            output = student(batch)
            nll = -output.log_probs.gather(1, labels[:, None]).squeeze(1)
            if not torch.isfinite(nll).all():
                raise FloatingPointError("non-finite Simple clone evaluation")
            predictions = output.log_probs.argmax(-1)
            loss_sum += float(nll.sum())
            correct += float((predictions == labels).sum())
            for label, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
                family = action_family(label)
                family_total[family] += 1
                family_correct[family] += int(label == prediction)
        result = {"nll": loss_sum / len(indices), "top1": correct / len(indices)}
        for family, total in family_total.items():
            result[f"{family}_rows"] = float(total)
            result[f"{family}_top1"] = (
                family_correct[family] / total if total else 0.0
            )
        return result

    baseline = evaluate(hold_indices)
    best_nll = baseline["nll"]
    best_epoch = 0
    history: list[dict] = []
    generator = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        student.train()
        order = generator.permutation(train_indices)
        train_sum = 0.0
        for start in range(0, len(order), batch_size):
            part = order[start : start + batch_size]
            batch, labels = make_batch(part)
            output = student(batch)
            loss = -output.log_probs.gather(1, labels[:, None]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Simple clone loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite Simple clone gradient at epoch {epoch}")
            optimizer.step()
            train_sum += float(loss.detach()) * len(part)
        metrics = evaluate(hold_indices)
        metrics.update(epoch=epoch, train_nll=train_sum / len(train_indices))
        history.append(metrics)
        print(
            f"[simple-clone] epoch={epoch} train={metrics['train_nll']:.5f} "
            f"hold_nll={metrics['nll']:.5f} hold_top1={metrics['top1']:.4f}",
            flush=True,
        )
        torch.save(student.state_dict(), out_dir / "student_latest.pt")
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_epoch = epoch
            torch.save(student.state_dict(), out_dir / "student_best.pt")
    if best_epoch == 0:
        raise RuntimeError("exact-private Simple clone did not improve whole-team heldout NLL")

    best_state = torch.load(out_dir / "student_best.pt", map_location="cpu", weights_only=False)
    changed = [
        name
        for name, donor_tensor in donor_state.items()
        if not torch.equal(donor_tensor.cpu(), best_state[name].cpu())
    ]
    illegal = [name for name in changed if name not in allowed_names]
    if illegal:
        raise RuntimeError(f"non-policy tensors changed during cloneability control: {illegal[:8]}")
    report = {
        "status": "complete",
        "diagnostic_only": True,
        "teacher": "SimpleHeuristics",
        "actor_private_input": "exact_submitted_player_set",
        "actor_opponent_input": "chronological_visible_information_only",
        "teacher_action_is_input": False,
        "validation_teams_used": False,
        "games": len({int(row["game"]) for row in rows}),
        "rows": len(rows),
        "acting_teams": len({int(row["team_index"]) for row in rows}),
        "holdout_teams": holdout_teams,
        "train_rows": len(train_indices),
        "holdout_rows": len(hold_indices),
        "baseline": baseline,
        "history": history,
        "best_epoch": best_epoch,
        "best_nll": best_nll,
        "changed_tensor_count": len(changed),
        "changed_tensors": changed,
        "allowed_policy_tensors": sorted(allowed_names),
        "outside_policy_byte_identical": True,
        "checkpoint": str(out_dir / "student_best.pt"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    from fog.config import load_config
    from fog.training.heuristic_distill import collect_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261331)
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
        retain_public_action_features=True,
    )
    print(
        f"[simple-clone] collected games={args.games} rows={len(rows)} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    report = train_policy_clone(
        cfg,
        args.checkpoint,
        rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        out_dir=args.out,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
