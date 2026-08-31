"""Test whether exact visible initial inputs support an unseen-team matchup prior.

This read-only architectural control trains a small permutation-symmetric DeepSets regressor on
materialized initial-state shards. It never changes an agent checkpoint. A strong held-out result
would implicate the learned trunk/value interface; failure alongside the frozen-trunk head points
instead to data coverage or intrinsically weak visible matchup signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _load(manifest_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = pl.read_parquet(manifest_path)
    entities, identities, fields, targets = [], [], [], []
    for shard_path in manifest["shard"].to_list():
        with np.load(shard_path) as shard:
            entities.append(np.asarray(shard["entities"], dtype=np.float32))
            identities.append(np.asarray(shard["entity_ids"], dtype=np.int64))
            fields.append(np.asarray(shard["field"], dtype=np.float32))
            targets.append(np.asarray(shard["outcome"], dtype=np.float32))
    return tuple(map(np.concatenate, (entities, identities, fields, targets)))


class DirectInitialPrior(nn.Module):
    """Permutation-symmetric own/opponent team encoder over exact visible token inputs."""

    def __init__(self, entity_dim: int, field_dim: int, vocab_sizes: list[int]) -> None:
        super().__init__()
        self.identity_embeddings = nn.ModuleList(
            nn.Embedding(size, 8) for size in vocab_sizes
        )
        token_in = entity_dim + 8 * len(vocab_sizes)
        self.entity = nn.Sequential(
            nn.Linear(token_in, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU()
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 2 + field_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, entities: torch.Tensor, identities: torch.Tensor,
                field: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat(
            [table(identities[..., index]) for index, table in enumerate(self.identity_embeddings)],
            dim=-1,
        )
        tokens = self.entity(torch.cat([entities, embedded], dim=-1))
        pooled = torch.cat([tokens[:, :6].mean(1), tokens[:, 6:12].mean(1), field], dim=-1)
        return self.value(pooled).squeeze(-1)


@torch.no_grad()
def _mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    total = 0.0
    count = 0
    model.eval()
    for entities, identities, field, target in loader:
        prediction = model(entities.to(device), identities.to(device), field.to(device))
        total += float(((prediction - target.to(device)) ** 2).sum())
        count += int(target.numel())
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261303)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = _load(args.train_manifest)
    heldout = _load(args.heldout_manifest)
    vocab_sizes = [
        max(int(train[1][..., i].max()), int(heldout[1][..., i].max())) + 1
        for i in range(train[1].shape[-1])
    ]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DirectInitialPrior(train[0].shape[-1], train[2].shape[-1], vocab_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_ds = TensorDataset(*(torch.from_numpy(value) for value in train))
    heldout_ds = TensorDataset(*(torch.from_numpy(value) for value in heldout))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    eval_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    heldout_loader = DataLoader(heldout_ds, batch_size=args.batch_size, shuffle=False)
    best = float("inf")
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for entities, identities, field, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(entities.to(device), identities.to(device), field.to(device))
            loss = ((prediction - target.to(device)) ** 2).mean()
            loss.backward()
            optimizer.step()
        train_mse = _mse(model, eval_train_loader, device)
        heldout_mse = _mse(model, heldout_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "heldout_mse": heldout_mse})
        print(f"[direct-prior] epoch={epoch} train={train_mse:.6f} heldout={heldout_mse:.6f}",
              flush=True)
        if heldout_mse < best - 1e-5:
            best = heldout_mse
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    report = {
        "status": "complete",
        "diagnostic_only": True,
        "agent_checkpoint_changed": False,
        "permutation_symmetric_team_pooling": True,
        "train_manifest": str(args.train_manifest),
        "heldout_manifest": str(args.heldout_manifest),
        "train_rows": len(train[3]),
        "heldout_rows": len(heldout[3]),
        "best_epoch": best_epoch,
        "best_heldout_mse": best,
        "final_train_mse": history[-1]["train_mse"],
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
