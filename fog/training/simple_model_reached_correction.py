"""Correct an independent SimpleHeuristics imitator on states reached by its own policy.

The model acts first. SimpleHeuristics is queried afterward and stored only as a label. No
teacher-derived action feature is retained or consumed. This file deliberately contains training
logic only; it does not alter the model architecture.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fog.training.dagger_heuristic_distill import collect_rows, correction_weight
from fog.training.outcome_heuristic_distill import team_holdout


def assert_clean_model_reached_rows(cfg: dict, rows: list[dict]) -> None:
    model_cfg = cfg.get("model", {})
    use_action_features = bool(model_cfg.get("use_action_features", False))
    public_only = bool(model_cfg.get("public_action_features_only", False))
    if use_action_features and not public_only:
        raise ValueError(
            "model.use_action_features requires public_action_features_only=true"
        )
    if public_only and not use_action_features:
        raise ValueError(
            "public_action_features_only requires model.use_action_features=true"
        )
    if float(model_cfg.get("simple_rule_prior_strength", 0.0)) != 0.0:
        raise ValueError("simple_rule_prior_strength must be zero")
    if not use_action_features:
        contaminated = [
            i for i, row in enumerate(rows) if "action_features" in row["frag"]
        ]
        if contaminated:
            raise ValueError(
                f"teacher-derived action features found in rows {contaminated[:8]}"
            )
        return

    from fog.encoding.public_action_features import public_action_features

    missing = [i for i, row in enumerate(rows) if "action_features" not in row["frag"]]
    if missing:
        raise ValueError(f"public action features missing from rows {missing[:8]}")
    contaminated = []
    for i, row in enumerate(rows):
        features = np.asarray(row["frag"]["action_features"], dtype=np.float32)
        if not np.array_equal(features, public_action_features(features)):
            contaminated.append(i)
    if contaminated:
        raise ValueError(
            f"answer-derived action feature values found in rows {contaminated[:8]}"
        )


def train_full_model(
    cfg: dict,
    checkpoint: str,
    rows: list[dict],
    *,
    team_count: int,
    epochs: int,
    batch_size: int,
    backbone_lr: float,
    policy_lr: float,
    pullback_coef: float,
    disagreement_weight: float,
    seed: int,
    out_dir: Path,
) -> dict:
    import torch

    from fog.models.agent import FogAgent
    from fog.training.ppo import _masked_kl
    from fog.training.rollout import _stack_frags

    if not rows:
        raise ValueError("no model-reached rows collected")
    assert_clean_model_reached_rows(cfg, rows)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    starting_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    student = FogAgent.from_tables(cfg).to(device)
    student.load_state_dict(starting_state)
    policy_parameters = list(student.policy_value.policy.parameters())
    if student.policy_value.action_scorer is not None:
        policy_parameters.extend(student.policy_value.action_scorer.parameters())
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
        raise ValueError("whole-team split produced an empty side")
    weights = np.asarray(
        [
            correction_weight(
                row["executed"], row["target"], disagreement=disagreement_weight
            )
            for row in rows
        ],
        dtype=np.float32,
    )

    def make_batch(indices: np.ndarray):
        batch = _stack_frags([rows[int(i)]["frag"] for i in indices], device=device)
        labels = torch.as_tensor(
            [rows[int(i)]["target"] for i in indices], dtype=torch.long, device=device
        )
        reference = torch.as_tensor(
            np.stack([rows[int(i)]["reference_log_probs"] for i in indices]),
            dtype=torch.float32,
            device=device,
        )
        batch_weights = torch.as_tensor(weights[indices], dtype=torch.float32, device=device)
        return batch, labels, reference, batch_weights

    @torch.no_grad()
    def evaluate(indices: np.ndarray) -> dict[str, float]:
        student.eval()
        nll_sum = correct = kl_sum = 0.0
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            batch, labels, reference, _ = make_batch(part)
            output = student(batch)
            nll = -output.log_probs.gather(1, labels[:, None]).squeeze(1)
            row_kl = _masked_kl(output.log_probs, reference)
            if not torch.isfinite(nll).all() or not torch.isfinite(row_kl).all():
                raise FloatingPointError("non-finite model-reached evaluation")
            nll_sum += float(nll.sum())
            correct += float((output.log_probs.argmax(-1) == labels).sum())
            kl_sum += float(row_kl.sum())
        return {
            "nll": nll_sum / len(indices),
            "top1": correct / len(indices),
            "pullback": kl_sum / len(indices),
        }

    baseline = evaluate(hold_indices)
    history: list[dict] = []
    best_nll = baseline["nll"]
    best_epoch = 0
    generator = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        student.train()
        order = generator.permutation(train_indices)
        train_sum = 0.0
        grad_norm_max = 0.0
        for start in range(0, len(order), batch_size):
            part = order[start : start + batch_size]
            batch, labels, reference, batch_weights = make_batch(part)
            output = student(batch)
            nll = -output.log_probs.gather(1, labels[:, None]).squeeze(1)
            weighted_nll = (batch_weights * nll).sum() / batch_weights.sum().clamp_min(1e-8)
            pullback = _masked_kl(output.log_probs, reference).mean()
            loss = weighted_nll + pullback_coef * pullback
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite correction loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite correction gradient at epoch {epoch}")
            optimizer.step()
            train_sum += float(loss.detach()) * len(part)
            grad_norm_max = max(grad_norm_max, float(grad_norm))
        metrics = evaluate(hold_indices)
        metrics.update(
            epoch=epoch,
            train_loss=train_sum / len(train_indices),
            grad_norm_max=grad_norm_max,
        )
        history.append(metrics)
        print(
            f"[model-reached-correction] epoch={epoch} "
            f"train={metrics['train_loss']:.5f} hold_nll={metrics['nll']:.5f} "
            f"hold_top1={metrics['top1']:.4f} pullback={metrics['pullback']:.5f}",
            flush=True,
        )
        torch.save(student.state_dict(), out_dir / "student_latest.pt")
        torch.save(student.state_dict(), out_dir / f"student_epoch_{epoch}.pt")
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_epoch = epoch
            torch.save(student.state_dict(), out_dir / "student_best.pt")

    if best_epoch == 0:
        raise RuntimeError("whole-team holdout correction loss did not improve")
    disagreements = sum(int(row["executed"] != row["target"]) for row in rows)
    report = {
        "status": "complete",
        "training_method": "full-model correction on model-reached states",
        "starting_checkpoint": checkpoint,
        "teacher_answer_available_to_model": False,
        "action_features_present": False,
        "validation_teams_used": False,
        "model_actions_executed_live": True,
        "teacher_queried_after_action": True,
        "games": len({int(row["game"]) for row in rows}),
        "rows": len(rows),
        "team_count": team_count,
        "holdout_teams": sorted(holdout_teams),
        "train_rows": len(train_indices),
        "holdout_rows": len(hold_indices),
        "disagreements": disagreements,
        "disagreement_rate": disagreements / len(rows),
        "backbone_lr": backbone_lr,
        "policy_lr": policy_lr,
        "pullback_coef": pullback_coef,
        "disagreement_weight": disagreement_weight,
        "baseline": baseline,
        "history": history,
        "best_epoch": best_epoch,
        "best_nll": best_nll,
        "checkpoint": str(out_dir / "student_best.pt"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    from fog.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--policy-lr", type=float, default=2e-5)
    parser.add_argument("--pullback-coef", type=float, default=0.1)
    parser.add_argument("--disagreement-weight", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260844)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    assert_clean_model_reached_rows(cfg, [])
    started = time.monotonic()
    rows, team_count = collect_rows(
        cfg,
        args.checkpoint,
        games=args.games,
        workers=args.workers,
        actor_threads=args.actor_threads,
        seed=args.seed,
    )
    assert_clean_model_reached_rows(cfg, rows)
    print(
        f"[model-reached-correction] collected games={args.games} rows={len(rows)} "
        f"elapsed={time.monotonic() - started:.1f}s answer_in_inputs=false",
        flush=True,
    )
    report = train_full_model(
        cfg,
        args.checkpoint,
        rows,
        team_count=team_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        policy_lr=args.policy_lr,
        pullback_coef=args.pullback_coef,
        disagreement_weight=args.disagreement_weight,
        seed=args.seed,
        out_dir=args.out,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
