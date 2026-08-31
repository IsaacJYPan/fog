"""Stage-one encoder pretraining from exact live observations; policy/value/belief stay frozen."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fog.config import load_config
from fog.constants.tables import TABLES
from fog.eval.encoder_information_probe import (
    HistoryProbe,
    SharedEntityProbe,
    SharedVectorProbe,
    _last_history,
)
from fog.eval.encoder_information_probe_live import TensorObservationDataset
from fog.models.three_encoder import CurrentStateEncoder, PrivateStateEncoder


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    from fog.encoding.dims import DIMS
    from fog.models.agent import FogAgent

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    initial = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(initial)
    agent.requires_grad_(False)
    agent.backbone.requires_grad_(True)
    policy_dim = agent.backbone.state_dim + agent.backbone.dyn_dim + (4 if agent.use_posterior_features else 0)
    sizes = {"species": TABLES.n_species, "item": TABLES.n_items, "ability": TABLES.n_abilities, "move": TABLES.n_moves}
    public_sources = {"public_token": 128, "public_pooled": 256, "fused_public": 256, "policy_input": policy_dim}
    private_sources = {"private_token": 128, "private_pooled": 256, "policy_input": policy_dim}
    history_sources = {"history_pooled": 256, "fused_public": 256, "policy_input": policy_dim}
    vector_sources = {"public_pooled": 256, "fused_public": 256, "policy_input": policy_dim}
    probes = nn.ModuleDict({
        "public": SharedEntityProbe(public_sources, sizes),
        "private": SharedEntityProbe(private_sources, sizes),
        "history": HistoryProbe(history_sources, sizes),
        "field": SharedVectorProbe(vector_sources, DIMS.field_dim),
        "legality": SharedVectorProbe(vector_sources, 14),
    }).to(device)
    probes.load_state_dict(torch.load(args.probe, map_location=device, weights_only=False))
    datasets = {name: TensorObservationDataset(args.data / f"{name}.pt") for name in ("train", "team_holdout", "battle_holdout")}
    embed_params = list(agent.backbone.embeds.parameters())
    embed_ids = {id(parameter) for parameter in embed_params}
    encoder_params = [parameter for parameter in agent.backbone.parameters() if id(parameter) not in embed_ids]
    optimizer = torch.optim.AdamW([
        {"params": embed_params, "lr": args.embedding_lr},
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": probes.parameters(), "lr": args.decoder_lr},
    ])

    def trainable_sources(batch: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        contexts = agent.encode_contexts(batch)
        fused = agent.public_context(contexts)
        posterior = fused.new_zeros(fused.shape[0], 4) if agent.use_posterior_features else fused[:, :0]
        policy = torch.cat([contexts.private_state, fused, posterior], dim=-1)
        return {
            "public_token": contexts.public_state_tokens,
            "public_pooled": contexts.public_state,
            "fused_public": fused,
            "policy_input": policy,
            "private_token": contexts.private_tokens,
            "private_pooled": contexts.private_state,
            "history_pooled": contexts.history,
        }, {"fused": fused, "policy": policy}

    def objective(batch: dict) -> torch.Tensor:
        sources, compact = trainable_sources(batch)
        public_raw, public_ids = CurrentStateEncoder.public_inputs(batch["entities"], batch["entity_ids"], agent.backbone.embeds)
        private_raw = PrivateStateEncoder._private_raw(batch["entities"])
        hist_raw, hist_ids, hist_state = _last_history(batch)
        public_loss, _ = probes["public"].loss_metrics("public_token", sources["public_token"], public_raw, public_ids)
        private_loss, _ = probes["private"].loss_metrics("private_token", sources["private_token"], private_raw, batch["entity_ids"][:, :6])
        history_loss, _ = probes["history"].loss_metrics("history_pooled", sources["history_pooled"], hist_raw, hist_ids, hist_state)
        fused_history_loss, _ = probes["history"].loss_metrics("fused_public", compact["fused"], hist_raw, hist_ids, hist_state)
        field_public = F.mse_loss(probes["field"]("public_pooled", sources["public_pooled"]), batch["field"])
        field_fused = F.mse_loss(probes["field"]("fused_public", compact["fused"]), batch["field"])
        return torch.stack([public_loss, private_loss, history_loss, fused_history_loss, field_public, field_fused]).mean()

    def evaluate(dataset: TensorObservationDataset) -> float:
        agent.eval(); probes.eval(); total = 0.0; batches = 0
        with torch.no_grad():
            for batch in dataset.batches(args.batch_size, shuffle=False, seed=args.seed, device=device):
                total += float(objective(batch)); batches += 1
        return total / max(1, batches)

    rows = []; best_score = float("inf"); best_agent = None; best_probe = None; stale = 0
    for epoch in range(args.epochs):
        agent.train(); probes.train(); running = 0.0; batches = 0
        for batch in datasets["train"].batches(args.batch_size, shuffle=True, seed=args.seed + epoch, device=device):
            optimizer.zero_grad(set_to_none=True); loss = objective(batch); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(agent.backbone.parameters()) + list(probes.parameters()), 5.0)
            optimizer.step(); running += float(loss.detach()); batches += 1
        team = evaluate(datasets["team_holdout"]); battle = evaluate(datasets["battle_holdout"])
        score = (team + battle) / 2
        row = {"epoch": epoch + 1, "train_loss": running / max(1, batches), "team_holdout_loss": team, "battle_holdout_loss": battle}
        rows.append(row); print(f"[encoder-reconstruction] {row}", flush=True)
        if score < best_score - args.min_delta:
            best_score = score; best_agent = copy.deepcopy(agent.state_dict()); best_probe = copy.deepcopy(probes.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_agent is None or best_probe is None:
        raise RuntimeError("encoder reconstruction did not produce a finite best checkpoint")
    changed = sorted(name for name, value in best_agent.items() if _digest(value) != _digest(initial[name]))
    if not changed:
        raise RuntimeError("encoder reconstruction produced no encoder parameter changes")
    illegal = [name for name in changed if not name.startswith("backbone.")]
    if illegal:
        raise RuntimeError(f"frozen non-backbone tensors changed: {illegal[:10]}")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(best_agent, args.out / "encoder_best.pt")
    torch.save(best_probe, args.out / "decoder_best.pt")
    report = {
        "status": "complete", "source": str(args.checkpoint), "policy_trained": False,
        "value_trained": False, "belief_trained": False, "data": str(args.data),
        "validation_teams_used": False, "changed_tensor_count": len(changed),
        "changed_tensor_prefixes_valid": True, "epochs": rows, "best_holdout_loss": best_score,
        "learning_rates": {"embeddings": args.embedding_lr, "encoders": args.encoder_lr, "decoders": args.decoder_lr},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-lr", type=float, default=2e-6)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20261053)
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
