"""Frozen-encoder probes for decision-relevant observation reconstruction.

The agent is never optimized here. Small decoders measure which legitimate current-input facts can
be recovered from entity tokens and from each pooled bottleneck. Replay splits must be supplied
explicitly so decisions from one battle cannot cross train/heldout boundaries.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fog.config import load_config
from fog.constants.schema import N_EV_UNITS, N_NATURES, N_TERA_TYPES
from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, EV_ID_FIELDS, NATURE_ID_FIELD
from fog.encoding.events import (
    H_FIRST_ACTOR,
    H_MY_ABILITY,
    H_MY_ITEM,
    H_MY_MOVE,
    H_MY_SWITCH_PHASE,
    H_OPP_ABILITY,
    H_OPP_ITEM,
    H_OPP_MOVE,
    H_OPP_SWITCH_PHASE,
)
from fog.models.three_encoder import CurrentStateEncoder, PrivateStateEncoder


def _known_ce(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    valid = (targets >= 0) & (targets < logits.shape[-1])
    if not bool(valid.any()):
        return logits.sum() * 0.0, 0, 0
    selected = logits[valid]
    truth = targets[valid]
    return F.cross_entropy(selected, truth), int((selected.argmax(-1) == truth).sum()), int(valid.sum())


def _majority_correct(targets: torch.Tensor, classes: int) -> int:
    valid = targets[(targets >= 0) & (targets < classes)]
    if not valid.numel():
        return 0
    return int(torch.bincount(valid.reshape(-1), minlength=classes).max())


class SharedEntityProbe(nn.Module):
    """One target decoder with a source-specific adapter for fair bottleneck comparisons."""

    def __init__(self, source_dims: dict[str, int], sizes: dict[str, int]) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict({name: nn.Linear(width, 256) for name, width in source_dims.items()})
        self.slot = nn.Embedding(12, 16)
        self.trunk = nn.Sequential(nn.Linear(272, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.raw = nn.Linear(256, ENTITY_RAW_DIM)
        self.species = nn.Linear(256, sizes["species"])
        self.item = nn.Linear(256, sizes["item"])
        self.ability = nn.Linear(256, sizes["ability"])
        self.move = nn.Linear(256, sizes["move"])
        self.nature = nn.Linear(256, N_NATURES)
        self.ev = nn.Linear(256, N_EV_UNITS)

    def hidden(self, source: str, values: torch.Tensor, slots: int) -> torch.Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(1).expand(-1, slots, -1)
        positions = torch.arange(slots, device=values.device)
        slot = self.slot(positions).unsqueeze(0).expand(values.shape[0], -1, -1)
        return self.trunk(torch.cat([self.adapters[source](values), slot], dim=-1))

    def loss_metrics(
        self, source: str, values: torch.Tensor, raw: torch.Tensor, ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        slots = raw.shape[1]
        h = self.hidden(source, values, slots)
        raw_pred = self.raw(h)
        raw_loss = F.mse_loss(raw_pred, raw)
        terms: list[torch.Tensor] = []
        metrics: dict[str, float] = {
            "raw_mse": float(raw_loss.detach()),
            "raw_zero_mse": float(raw.square().mean()),
        }

        specs = [
            ("species", self.species(h), ids[..., 0]),
            ("item", self.item(h), ids[..., 1]),
            ("ability", self.ability(h), ids[..., 2]),
            ("nature", self.nature(h), ids[..., NATURE_ID_FIELD]),
        ]
        for move_index in range(DIMS.n_move_slots):
            specs.append((f"move{move_index}", self.move(h), ids[..., 3 + move_index]))
        for ev_index, column in enumerate(EV_ID_FIELDS):
            specs.append((f"ev{ev_index}", self.ev(h), ids[..., column]))

        for name, logits, target in specs:
            loss, correct, count = _known_ce(logits, target)
            if count:
                terms.append(loss)
            metrics[f"{name}_correct"] = float(correct)
            metrics[f"{name}_count"] = float(count)
            metrics[f"{name}_majority_correct"] = float(
                _majority_correct(target, logits.shape[-1])
            )
        cat_loss = torch.stack(terms).mean() if terms else raw_loss * 0.0
        metrics["categorical_ce"] = float(cat_loss.detach())
        return raw_loss + cat_loss, metrics


class SharedVectorProbe(nn.Module):
    def __init__(self, source_dims: dict[str, int], output_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, output_dim))
                for name, width in source_dims.items()
            }
        )

    def forward(self, source: str, values: torch.Tensor) -> torch.Tensor:
        return self.heads[source](values)


class HistoryProbe(nn.Module):
    def __init__(self, source_dims: dict[str, int], sizes: dict[str, int]) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict({name: nn.Linear(width, 256) for name, width in source_dims.items()})
        self.trunk = nn.Sequential(nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.raw = nn.Linear(256, DIMS.turn_dim)
        self.state = nn.Linear(256, DIMS.hist_state)
        self.species = nn.Linear(256, sizes["species"])
        self.item = nn.Linear(256, sizes["item"])
        self.ability = nn.Linear(256, sizes["ability"])
        self.move = nn.Linear(256, sizes["move"])
        self.tera = nn.Linear(256, N_TERA_TYPES + 1)
        self.actor = nn.Linear(256, 3)
        self.phase = nn.Linear(256, 3)

    def loss_metrics(
        self, source: str, values: torch.Tensor, raw: torch.Tensor, ids: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        h = self.trunk(self.adapters[source](values))
        raw_loss = F.mse_loss(self.raw(h), raw)
        state_loss = F.mse_loss(self.state(h), state)
        species_fields = (0, 1, 3, 5, 6, 7)
        specs = [(f"species{index}", self.species(h), ids[..., column]) for index, column in enumerate(species_fields)]
        specs += [
            ("my_move", self.move(h), ids[..., H_MY_MOVE]),
            ("opp_move", self.move(h), ids[..., H_OPP_MOVE]),
            ("my_item", self.item(h), ids[..., H_MY_ITEM]),
            ("opp_item", self.item(h), ids[..., H_OPP_ITEM]),
            ("my_ability", self.ability(h), ids[..., H_MY_ABILITY]),
            ("opp_ability", self.ability(h), ids[..., H_OPP_ABILITY]),
            ("my_tera", self.tera(h), ids[..., 12]),
            ("opp_tera", self.tera(h), ids[..., 13]),
            ("first_actor", self.actor(h), ids[..., H_FIRST_ACTOR]),
            ("my_switch_phase", self.phase(h), ids[..., H_MY_SWITCH_PHASE]),
            ("opp_switch_phase", self.phase(h), ids[..., H_OPP_SWITCH_PHASE]),
        ]
        terms: list[torch.Tensor] = []
        metrics: dict[str, float] = {
            "raw_mse": float(raw_loss.detach()),
            "raw_zero_mse": float(raw.square().mean()),
            "state_mse": float(state_loss.detach()),
            "state_zero_mse": float(state.square().mean()),
        }
        for name, logits, target in specs:
            loss, correct, count = _known_ce(logits, target)
            if count:
                terms.append(loss)
            metrics[f"{name}_correct"] = float(correct)
            metrics[f"{name}_count"] = float(count)
            metrics[f"{name}_majority_correct"] = float(
                _majority_correct(target, logits.shape[-1])
            )
        cat_loss = torch.stack(terms).mean() if terms else raw_loss * 0.0
        metrics["categorical_ce"] = float(cat_loss.detach())
        return raw_loss + state_loss + cat_loss, metrics


def _sources(agent, batch: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    with torch.no_grad():
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


def _last_history(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Stored replay histories are right-aligned. Zero-length rows receive all-zero targets.
    present = batch["history_len"] > 0
    raw = batch["history"][:, -1].clone()
    ids = batch["history_ids"][:, -1].clone()
    state = batch["history_state"][:, -1].clone()
    raw[~present] = 0
    ids[~present] = 0
    state[~present] = 0
    return raw, ids, state


def _accumulate(total: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        total[key] += float(value)


def _finalize(total: dict[str, float], batches: int) -> dict[str, float]:
    report: dict[str, float] = {}
    for key, value in total.items():
        if key.endswith(("_correct", "_count")):
            report[key] = value
        else:
            report[key] = value / max(1, batches)
    for key in list(report):
        if key.endswith("_majority_correct"):
            stem = key[:-17]
            count = report.get(stem + "_count", 0.0)
            report[stem + "_majority_top1"] = report[key] / max(1.0, count)
        elif key.endswith("_correct"):
            stem = key[:-8]
            count = report.get(stem + "_count", 0.0)
            report[stem + "_top1"] = report[key] / max(1.0, count)
    return report


def run(args: argparse.Namespace) -> dict:
    from fog.constants.tables import TABLES
    from fog.models.agent import FogAgent
    from fog.training.dataset import StreamingReplayDataset

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    agent.eval()
    agent.requires_grad_(False)

    sizes = {
        "species": TABLES.n_species,
        "item": TABLES.n_items,
        "ability": TABLES.n_abilities,
        "move": TABLES.n_moves,
    }
    policy_dim = agent.backbone.state_dim + agent.backbone.dyn_dim + (4 if agent.use_posterior_features else 0)
    public_sources = {"public_token": 128, "public_pooled": 256, "fused_public": 256, "policy_input": policy_dim}
    private_sources = {"private_token": 128, "private_pooled": 256, "policy_input": policy_dim}
    history_sources = {"history_pooled": 256, "fused_public": 256, "policy_input": policy_dim}
    vector_sources = {"public_pooled": 256, "fused_public": 256, "policy_input": policy_dim}

    probes = nn.ModuleDict(
        {
            "public": SharedEntityProbe(public_sources, sizes),
            "private": SharedEntityProbe(private_sources, sizes),
            "history": HistoryProbe(history_sources, sizes),
            "field": SharedVectorProbe(vector_sources, DIMS.field_dim),
            "legality": SharedVectorProbe(vector_sources, 14),
        }
    ).to(device)
    if args.init_probe:
        probes.load_state_dict(torch.load(args.init_probe, map_location=device, weights_only=False))
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr)
    train = StreamingReplayDataset(args.train_manifest, shard_buffer=args.shard_buffer)
    heldout = StreamingReplayDataset(args.holdout_manifest, shard_buffer=args.shard_buffer)

    def process(batch: dict, training: bool) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
        sources, compact = _sources(agent, batch)
        public_raw, public_ids = CurrentStateEncoder.public_inputs(batch["entities"], batch["entity_ids"], agent.backbone.embeds)
        private_raw = PrivateStateEncoder._private_raw(batch["entities"])
        private_ids = batch["entity_ids"][:, :6]
        hist_raw, hist_ids, hist_state = _last_history(batch)
        losses: list[torch.Tensor] = []
        metrics: dict[str, dict[str, float]] = {}
        for name in public_sources:
            loss, item = probes["public"].loss_metrics(name, sources[name], public_raw, public_ids)
            losses.append(loss)
            metrics[f"public/{name}"] = item
        for name in private_sources:
            loss, item = probes["private"].loss_metrics(name, sources[name], private_raw, private_ids)
            losses.append(loss)
            metrics[f"private/{name}"] = item
        history_values = {"history_pooled": sources["history_pooled"], "fused_public": compact["fused"], "policy_input": compact["policy"]}
        for name, values in history_values.items():
            loss, item = probes["history"].loss_metrics(name, values, hist_raw, hist_ids, hist_state)
            losses.append(loss)
            metrics[f"history/{name}"] = item
        vector_values = {"public_pooled": sources["public_pooled"], "fused_public": compact["fused"], "policy_input": compact["policy"]}
        for name, values in vector_values.items():
            field_pred = probes["field"](name, values)
            field_loss = F.mse_loss(field_pred, batch["field"])
            losses.append(field_loss)
            metrics[f"field/{name}"] = {
                "mse": float(field_loss.detach()),
                "zero_mse": float(batch["field"].square().mean()),
            }
            legal_pred = probes["legality"](name, values)
            legal_loss = F.binary_cross_entropy_with_logits(legal_pred, batch["legality_mask"].float())
            legal_bits = (legal_pred > 0) == batch["legality_mask"]
            losses.append(legal_loss)
            metrics[f"legality/{name}"] = {
                "bce": float(legal_loss.detach()),
                "bit_accuracy": float(legal_bits.float().mean()),
                "row_accuracy": float(legal_bits.all(-1).float().mean()),
            }
        return torch.stack(losses).mean(), metrics

    history_rows: list[dict] = []
    for epoch in range(args.epochs):
        probes.train()
        running = 0.0
        batches = 0
        for batch in train.batches(args.batch_size, shuffle=True, seed=args.seed + epoch, device=device):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = process(batch, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probes.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
            if args.max_train_batches and batches >= args.max_train_batches:
                break
        history_rows.append({"epoch": epoch + 1, "train_loss": running / max(1, batches)})
        print(f"[encoder-probe] epoch={epoch+1} train_loss={history_rows[-1]['train_loss']:.5f}", flush=True)

    probes.eval()
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for batch_index, batch in enumerate(heldout.batches(args.batch_size, shuffle=False, seed=args.seed, device=device)):
            _, metrics = process(batch, False)
            for group, item in metrics.items():
                _accumulate(totals[group], item)
                counts[group] += 1
            if args.max_eval_batches and batch_index + 1 >= args.max_eval_batches:
                break
    heldout_report = {group: _finalize(values, counts[group]) for group, values in totals.items()}
    agent_changed = any(parameter.grad is not None for parameter in agent.parameters())
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "train_manifest": str(args.train_manifest),
        "holdout_manifest": str(args.holdout_manifest),
        "split_contract": "caller-supplied replay-level split; no validation data",
        "agent_frozen": not agent_changed and not any(parameter.requires_grad for parameter in agent.parameters()),
        "epochs": history_rows,
        "heldout": heldout_report,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(probes.state_dict(), args.out / "probe.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history_rows)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-probe", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--shard-buffer", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261048)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    run(args)


if __name__ == "__main__":
    main()
