"""Compare public and privileged critics on the same disjoint simulator split.

This is a non-promotable asymmetric-critic control.  Both models start from the same accepted
agent and train only the ordinary value head plus its shared-latent adapter.  The public control
receives the deployable observation.  The privileged control additionally replaces the belief
input tokens with exact simulator truth; its policy output is ignored and never acts.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from fog.config import load_config_with_args
from fog.encoding.entities import (
    EV_ID_FIELDS,
    NATURE_ID_FIELD,
    ORIGINAL_ABILITY_ID_FIELD,
    ORIGINAL_ITEM_ID_FIELD,
)
from fog.models.agent import FogAgent
from fog.models.oracle import apply_set_oracle
from fog.models.self_belief import tokens_from_labels
from fog.training.dataset import StreamingReplayDataset

VALUE_PREFIXES = ("policy_value.value.", "shared_to_value.")
VALUE_INTERFACE_PREFIXES = VALUE_PREFIXES + ("shared_latent.",)


def privileged_value_batch(batch: dict) -> dict:
    """Return a critic-only copy whose opponent set identities contain exact training truth."""
    if batch.get("belief_targets") is None:
        raise ValueError("privileged value control requires exact opponent belief targets")
    truth = tokens_from_labels(batch["belief_targets"])
    result = dict(batch)
    result["opp_revealed"] = apply_set_oracle(batch.get("opp_revealed") or {}, truth)
    # The deployable value path consumes entity identities, not the belief input.  Inject the
    # simulator-only set sheet into opponent slots 6..11 of this critic copy so the control is
    # genuinely privileged while leaving public battle condition/history unchanged.
    ids = batch["entity_ids"].clone()
    opponent = ids[:, 6:12]

    def replace(target: torch.Tensor, source: torch.Tensor) -> None:
        target.copy_(torch.where(source >= 0, source.to(target.dtype), target))

    replace(opponent[..., 0], truth[..., 0])
    replace(opponent[..., 1], truth[..., 1])
    replace(opponent[..., 2], truth[..., 2])
    replace(opponent[..., 3:7], truth[..., 3:7])
    replace(opponent[..., NATURE_ID_FIELD], truth[..., 7])
    ev_columns = list(EV_ID_FIELDS)
    opponent[..., ev_columns] = torch.where(
        truth[..., 8:14] >= 0,
        truth[..., 8:14].to(opponent.dtype),
        opponent[..., ev_columns],
    )
    replace(opponent[..., ORIGINAL_ITEM_ID_FIELD], truth[..., 1])
    replace(opponent[..., ORIGINAL_ABILITY_ID_FIELD], truth[..., 2])
    result["entity_ids"] = ids
    return result


def select_value_parameters(
    agent: FogAgent, *, adapt_shared_latent: bool = False
) -> list[torch.nn.Parameter]:
    prefixes = VALUE_INTERFACE_PREFIXES if adapt_shared_latent else VALUE_PREFIXES
    selected = []
    for name, parameter in agent.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        if parameter.requires_grad:
            selected.append(parameter)
    if not selected:
        raise ValueError("no ordinary value parameters matched the diagnostic scope")
    return selected


def _batches(dataset: StreamingReplayDataset, batch_size: int, seed: int,
             device: torch.device, limit: int):
    for index, batch in enumerate(
        dataset.batches(batch_size, shuffle=True, seed=seed, device=device)
    ):
        if limit and index >= limit:
            break
        yield batch


@torch.no_grad()
def evaluate(model: FogAgent, dataset: StreamingReplayDataset, *, privileged: bool,
             batch_size: int, seed: int, device: torch.device, limit: int) -> dict[str, float]:
    model.eval()
    squared = 0.0
    absolute = 0.0
    target_sum = 0.0
    target_sq_sum = 0.0
    count = 0
    for batch in _batches(dataset, batch_size, seed, device, limit):
        model_batch = privileged_value_batch(batch) if privileged else batch
        prediction = model(model_batch).value.reshape(-1)
        target = batch["outcome"].reshape(-1)
        error = prediction - target
        squared += float(error.square().sum())
        absolute += float(error.abs().sum())
        target_sum += float(target.sum())
        target_sq_sum += float(target.square().sum())
        count += int(target.numel())
    if not count:
        raise ValueError("value control evaluation produced zero rows")
    mse = squared / count
    target_mean = target_sum / count
    target_variance = max(0.0, target_sq_sum / count - target_mean * target_mean)
    return {
        "rows": count,
        "mse": mse,
        "mae": absolute / count,
        "target_variance": target_variance,
        "explained_variance_proxy": 1.0 - mse / max(target_variance, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--heldout-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-batches", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261337)
    parser.add_argument(
        "--adapt-shared-latent",
        action="store_true",
        help=(
            "Nonpromotable interface control: train the shared typed latent as well as the "
            "ordinary value reader/head in both paired copies."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config_with_args(args.config)
    base = FogAgent.from_tables(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base.load_state_dict(state)
    public = copy.deepcopy(base)
    privileged = copy.deepcopy(base)
    public_parameters = select_value_parameters(
        public, adapt_shared_latent=args.adapt_shared_latent
    )
    privileged_parameters = select_value_parameters(
        privileged, adapt_shared_latent=args.adapt_shared_latent
    )
    public_optimizer = torch.optim.AdamW(public_parameters, lr=args.lr)
    privileged_optimizer = torch.optim.AdamW(privileged_parameters, lr=args.lr)
    train = StreamingReplayDataset(args.train_manifest, shard_buffer=16)
    heldout = StreamingReplayDataset(args.heldout_manifest, shard_buffer=16)

    baseline_public = evaluate(
        base, heldout, privileged=False, batch_size=args.batch_size,
        seed=args.seed + 1, device=device, limit=args.eval_batches,
    )
    baseline_privileged = evaluate(
        base, heldout, privileged=True, batch_size=args.batch_size,
        seed=args.seed + 1, device=device, limit=args.eval_batches,
    )
    history = []
    best_public = baseline_public["mse"]
    best_privileged = baseline_privileged["mse"]
    best_privileged_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        public.eval()
        privileged.eval()
        train_public_sum = 0.0
        train_privileged_sum = 0.0
        train_rows = 0
        for batch in _batches(train, args.batch_size, args.seed + epoch, device,
                              args.train_batches):
            target = batch["outcome"].reshape(-1)
            public_optimizer.zero_grad(set_to_none=True)
            public_prediction = public(batch).value.reshape(-1)
            public_loss = (public_prediction - target).square().mean()
            public_loss.backward()
            public_optimizer.step()

            privileged_optimizer.zero_grad(set_to_none=True)
            privileged_prediction = privileged(privileged_value_batch(batch)).value.reshape(-1)
            privileged_loss = (privileged_prediction - target).square().mean()
            privileged_loss.backward()
            privileged_optimizer.step()
            n = int(target.numel())
            train_public_sum += float(public_loss.detach()) * n
            train_privileged_sum += float(privileged_loss.detach()) * n
            train_rows += n

        public_eval = evaluate(
            public, heldout, privileged=False, batch_size=args.batch_size,
            seed=args.seed + 100 + epoch, device=device, limit=args.eval_batches,
        )
        privileged_eval = evaluate(
            privileged, heldout, privileged=True, batch_size=args.batch_size,
            seed=args.seed + 100 + epoch, device=device, limit=args.eval_batches,
        )
        row = {
            "epoch": epoch,
            "train_rows": train_rows,
            "train_public_mse": train_public_sum / max(1, train_rows),
            "train_privileged_mse": train_privileged_sum / max(1, train_rows),
            "public_heldout": public_eval,
            "privileged_heldout": privileged_eval,
        }
        history.append(row)
        print(
            f"[privileged-value] epoch={epoch} public={public_eval['mse']:.6f} "
            f"privileged={privileged_eval['mse']:.6f} "
            f"train={row['train_public_mse']:.6f}/{row['train_privileged_mse']:.6f}",
            flush=True,
        )
        best_public = min(best_public, public_eval["mse"])
        if privileged_eval["mse"] < best_privileged - 1e-5:
            best_privileged = privileged_eval["mse"]
            best_privileged_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    report = {
        "status": "complete",
        "diagnostic_only": True,
        "agent_checkpoint_changed": False,
        "actor_received_privileged_truth": False,
        "trainable_prefixes": list(
            VALUE_INTERFACE_PREFIXES if args.adapt_shared_latent else VALUE_PREFIXES
        ),
        "adapt_shared_latent": bool(args.adapt_shared_latent),
        "public_trainable_parameters": sum(p.numel() for p in public_parameters),
        "privileged_trainable_parameters": sum(p.numel() for p in privileged_parameters),
        "train_manifest": str(args.train_manifest),
        "heldout_manifest": str(args.heldout_manifest),
        "baseline_public": baseline_public,
        "baseline_privileged_unadapted": baseline_privileged,
        "best_public_heldout_mse": best_public,
        "best_privileged_heldout_mse": best_privileged,
        "best_privileged_epoch": best_privileged_epoch,
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
