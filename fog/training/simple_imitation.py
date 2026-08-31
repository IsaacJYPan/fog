"""Train a fresh model to imitate SimpleHeuristics without exposing its answer as input.

The collector runs the model forward on an ordinary public battle observation first.  It then
queries SimpleHeuristics, stores that action only as the supervised label, and executes it.  The
model configuration must disable action features and the saved rows explicitly omit them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fog.training.supervised import grouped_row_split


def split_by_acting_team(
    rows: list[dict], *, holdout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Hold out entire acting teams so adjacent states cannot cross the split."""
    split = grouped_row_split(
        rows, "team_index", holdout_fraction=holdout_fraction, seed=seed
    )
    return split.train, split.holdout, list(split.holdout_groups)


def assert_teacher_answer_is_not_an_input(cfg: dict, rows: list[dict]) -> None:
    model_cfg = cfg.get("model", {})
    if bool(model_cfg.get("use_action_features", False)):
        raise ValueError("model.use_action_features must be false for independent imitation")
    if float(model_cfg.get("simple_rule_prior_strength", 0.0)) != 0.0:
        raise ValueError("simple_rule_prior_strength must be zero")
    contaminated = [i for i, row in enumerate(rows) if "action_features" in row["frag"]]
    if contaminated:
        raise ValueError(f"teacher-derived action features found in rows {contaminated[:8]}")


def train_from_random_init(
    cfg: dict,
    init_checkpoint: Path,
    rows: list[dict],
    *,
    epochs: int,
    batch_size: int,
    backbone_lr: float,
    policy_lr: float,
    holdout_fraction: float,
    seed: int,
    out_dir: Path,
) -> dict:
    import torch

    from fog.models.agent import FogAgent
    from fog.training.rollout import _stack_frags

    assert_teacher_answer_is_not_an_input(cfg, rows)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    student = FogAgent.from_tables(cfg).to(device)
    student.load_state_dict(
        torch.load(init_checkpoint, map_location=device, weights_only=False)
    )
    policy_parameters = list(student.policy_value.policy.parameters())
    policy_ids = {id(parameter) for parameter in policy_parameters}
    backbone_parameters = [
        parameter for parameter in student.parameters() if id(parameter) not in policy_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": policy_parameters, "lr": policy_lr},
        ],
        weight_decay=1e-4,
    )
    train_indices, hold_indices, hold_teams = split_by_acting_team(
        rows, holdout_fraction=holdout_fraction, seed=seed
    )

    def make_batch(indices: np.ndarray) -> tuple[dict, torch.Tensor]:
        batch = _stack_frags([rows[int(i)]["frag"] for i in indices], device=device)
        labels = torch.as_tensor(
            [rows[int(i)]["action"] for i in indices], dtype=torch.long, device=device
        )
        return batch, labels

    @torch.no_grad()
    def evaluate(indices: np.ndarray) -> dict[str, float]:
        student.eval()
        nll_sum = correct = 0.0
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            batch, labels = make_batch(part)
            output = student(batch)
            nll = -output.log_probs.gather(1, labels[:, None]).squeeze(1)
            if not torch.isfinite(nll).all():
                raise FloatingPointError("non-finite imitation evaluation")
            nll_sum += float(nll.sum())
            correct += float((output.log_probs.argmax(-1) == labels).sum())
        return {"nll": nll_sum / len(indices), "top1": correct / len(indices)}

    baseline = evaluate(hold_indices)
    history: list[dict] = []
    best_nll = baseline["nll"]
    best_epoch = 0
    generator = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        student.train()
        order = generator.permutation(train_indices)
        train_nll_sum = 0.0
        grad_norm_max = 0.0
        for start in range(0, len(order), batch_size):
            part = order[start : start + batch_size]
            batch, labels = make_batch(part)
            output = student(batch)
            loss = -output.log_probs.gather(1, labels[:, None]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite imitation loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite imitation gradient at epoch {epoch}")
            optimizer.step()
            train_nll_sum += float(loss.detach()) * len(part)
            grad_norm_max = max(grad_norm_max, float(grad_norm))
        metrics = evaluate(hold_indices)
        metrics.update(
            epoch=epoch,
            train_nll=train_nll_sum / len(train_indices),
            grad_norm_max=grad_norm_max,
        )
        history.append(metrics)
        print(
            f"[simple-imitation] epoch={epoch} train_nll={metrics['train_nll']:.5f} "
            f"hold_nll={metrics['nll']:.5f} hold_top1={metrics['top1']:.4f}",
            flush=True,
        )
        torch.save(student.state_dict(), out_dir / "student_latest.pt")
        torch.save(student.state_dict(), out_dir / f"student_epoch_{epoch}.pt")
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_epoch = epoch
            torch.save(student.state_dict(), out_dir / "student_best.pt")

    if best_epoch == 0:
        raise RuntimeError("whole-team holdout imitation loss did not improve")
    report = {
        "status": "complete",
        "training_method": "fresh supervised imitation of SimpleHeuristics",
        "random_initialization": True,
        "teacher_answer_available_to_model": False,
        "action_features_present": False,
        "validation_teams_used": False,
        "games": len({int(row["game"]) for row in rows}),
        "rows": len(rows),
        "acting_teams": len({int(row["team_index"]) for row in rows}),
        "heldout_team_indices": hold_teams,
        "train_rows": len(train_indices),
        "holdout_rows": len(hold_indices),
        "baseline": baseline,
        "history": history,
        "best_epoch": best_epoch,
        "best_nll": best_nll,
        "checkpoint": str(out_dir / "student_best.pt"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import torch

    from fog.config import load_config
    from fog.models.agent import FogAgent
    from fog.training.heuristic_distill import collect_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260839)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    assert_teacher_answer_is_not_an_input(cfg, [])
    torch.manual_seed(args.seed)
    random_init = FogAgent.from_tables(cfg).cpu()
    init_checkpoint = args.out / "random_init.pt"
    torch.save(random_init.state_dict(), init_checkpoint)
    started = time.monotonic()
    rows = collect_rows(
        cfg,
        str(init_checkpoint),
        games=args.games,
        workers=args.workers,
        actor_threads=args.actor_threads,
        seed=args.seed,
    )
    assert_teacher_answer_is_not_an_input(cfg, rows)
    print(
        f"[simple-imitation] collected games={args.games} rows={len(rows)} "
        f"elapsed={time.monotonic() - started:.1f}s answer_in_inputs=false",
        flush=True,
    )
    report = train_from_random_init(
        cfg,
        init_checkpoint,
        rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        policy_lr=args.policy_lr,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        out_dir=args.out,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
