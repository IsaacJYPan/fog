"""Train frozen-agent reconstruction probes on authoritative live battle observations."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fog.config import load_config
from fog.constants.schema import N_EV_UNITS, N_NATURES, N_TERA_TYPES
from fog.constants.tables import TABLES
from fog.encoding.dims import DIMS
from fog.encoding.entities import EV_ID_FIELDS, NATURE_ID_FIELD
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
from fog.eval.encoder_information_probe import (
    HistoryProbe,
    SharedEntityProbe,
    SharedVectorProbe,
    _accumulate,
    _finalize,
    _last_history,
    _sources,
)
from fog.models.three_encoder import CurrentStateEncoder, PrivateStateEncoder


class TensorObservationDataset:
    def __init__(self, path: Path) -> None:
        self.data = torch.load(path, map_location="cpu", weights_only=False)
        self.n = int(self.data["entities"].shape[0])

    def batches(self, batch_size: int, *, shuffle: bool, seed: int, device) -> object:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(self.n, generator=generator) if shuffle else torch.arange(self.n)
        for start in range(0, self.n, batch_size):
            idx = order[start : start + batch_size]
            yield {key: value[idx].to(device) for key, value in self.data.items()}


def _mode(target: torch.Tensor, classes: int) -> int:
    valid = target[(target >= 0) & (target < classes)].reshape(-1)
    return int(torch.bincount(valid, minlength=classes).argmax()) if valid.numel() else 0


def _entity_targets(ids: torch.Tensor, *, private: bool) -> dict[str, tuple[torch.Tensor, int]]:
    out = {
        "species": (ids[..., 0], TABLES.n_species),
        "item": (ids[..., 1], TABLES.n_items),
        "ability": (ids[..., 2], TABLES.n_abilities),
        "nature": (ids[..., NATURE_ID_FIELD], N_NATURES),
    }
    for index in range(DIMS.n_move_slots):
        out[f"move{index}"] = (ids[..., 3 + index], TABLES.n_moves)
    for index, column in enumerate(EV_ID_FIELDS):
        out[f"ev{index}"] = (ids[..., column], N_EV_UNITS)
    return out


def _history_targets(ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, int]]:
    species_fields = (0, 1, 3, 5, 6, 7)
    out = {
        f"species{index}": (ids[..., column], TABLES.n_species)
        for index, column in enumerate(species_fields)
    }
    out.update({
        "my_move": (ids[..., H_MY_MOVE], TABLES.n_moves),
        "opp_move": (ids[..., H_OPP_MOVE], TABLES.n_moves),
        "my_item": (ids[..., H_MY_ITEM], TABLES.n_items),
        "opp_item": (ids[..., H_OPP_ITEM], TABLES.n_items),
        "my_ability": (ids[..., H_MY_ABILITY], TABLES.n_abilities),
        "opp_ability": (ids[..., H_OPP_ABILITY], TABLES.n_abilities),
        "my_tera": (ids[..., 12], N_TERA_TYPES + 1),
        "opp_tera": (ids[..., 13], N_TERA_TYPES + 1),
        "first_actor": (ids[..., H_FIRST_ACTOR], 3),
        "my_switch_phase": (ids[..., H_MY_SWITCH_PHASE], 3),
        "opp_switch_phase": (ids[..., H_OPP_SWITCH_PHASE], 3),
    })
    return out


def _train_modes(train: TensorObservationDataset, agent) -> dict[str, dict[str, int]]:
    data = train.data
    public_ids = data["entity_ids"].clone()
    own = slice(0, 6)
    public_ids[:, own, 1] = agent.backbone.embeds.item.num_embeddings - 1
    public_ids[:, own, 2] = agent.backbone.embeds.ability.num_embeddings - 1
    public_ids[:, own, 3 : 3 + DIMS.n_move_slots] = agent.backbone.embeds.move.num_embeddings - 1
    public_ids[:, own, NATURE_ID_FIELD] = agent.backbone.embeds.nature.num_embeddings - 1
    public_ids[:, own, list(EV_ID_FIELDS)] = agent.backbone.embeds.ev.num_embeddings - 1
    _, history_ids, _ = _last_history(data)
    groups = {
        "public": _entity_targets(public_ids, private=False),
        "private": _entity_targets(data["entity_ids"][:, :6], private=True),
        "history": _history_targets(history_ids),
    }
    return {
        group: {name: _mode(target, classes) for name, (target, classes) in targets.items()}
        for group, targets in groups.items()
    }


def _fixed_majority(metrics: dict[str, dict[str, float]], batch: dict, agent, modes) -> None:
    _, public_ids = CurrentStateEncoder.public_inputs(
        batch["entities"], batch["entity_ids"], agent.backbone.embeds
    )
    _, history_ids, _ = _last_history(batch)
    targets = {
        "public": _entity_targets(public_ids, private=False),
        "private": _entity_targets(batch["entity_ids"][:, :6], private=True),
        "history": _history_targets(history_ids),
    }
    for group, specs in targets.items():
        for metric_group, item in metrics.items():
            if not metric_group.startswith(group + "/"):
                continue
            for name, (target, classes) in specs.items():
                valid = (target >= 0) & (target < classes)
                item[f"{name}_majority_correct"] = float(
                    ((target == modes[group][name]) & valid).sum()
                )


def run(args: argparse.Namespace) -> dict:
    from fog.models.agent import FogAgent

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    agent.eval(); agent.requires_grad_(False)
    sizes = {"species": TABLES.n_species, "item": TABLES.n_items, "ability": TABLES.n_abilities, "move": TABLES.n_moves}
    policy_dim = agent.backbone.state_dim + agent.backbone.dyn_dim + (4 if agent.use_posterior_features else 0)
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
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr)
    datasets = {
        name: TensorObservationDataset(args.data / f"{name}.pt")
        for name in ("train", "team_holdout", "battle_holdout")
    }
    modes = _train_modes(datasets["train"], agent)

    def process(batch: dict, add_baseline: bool) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
        sources, compact = _sources(agent, batch)
        public_raw, public_ids = CurrentStateEncoder.public_inputs(batch["entities"], batch["entity_ids"], agent.backbone.embeds)
        private_raw = PrivateStateEncoder._private_raw(batch["entities"])
        private_ids = batch["entity_ids"][:, :6]
        hist_raw, hist_ids, hist_state = _last_history(batch)
        losses: list[torch.Tensor] = []; metrics: dict[str, dict[str, float]] = {}
        for name in public_sources:
            loss, item = probes["public"].loss_metrics(name, sources[name], public_raw, public_ids)
            losses.append(loss); metrics[f"public/{name}"] = item
        for name in private_sources:
            loss, item = probes["private"].loss_metrics(name, sources[name], private_raw, private_ids)
            losses.append(loss); metrics[f"private/{name}"] = item
        history_values = {"history_pooled": sources["history_pooled"], "fused_public": compact["fused"], "policy_input": compact["policy"]}
        for name, values in history_values.items():
            loss, item = probes["history"].loss_metrics(name, values, hist_raw, hist_ids, hist_state)
            losses.append(loss); metrics[f"history/{name}"] = item
        vector_values = {"public_pooled": sources["public_pooled"], "fused_public": compact["fused"], "policy_input": compact["policy"]}
        for name, values in vector_values.items():
            field_pred = probes["field"](name, values)
            field_loss = F.mse_loss(field_pred, batch["field"]); losses.append(field_loss)
            metrics[f"field/{name}"] = {"mse": float(field_loss.detach()), "zero_mse": float(batch["field"].square().mean())}
            legal_pred = probes["legality"](name, values)
            legal_loss = F.binary_cross_entropy_with_logits(legal_pred, batch["legality_mask"].float()); losses.append(legal_loss)
            legal_bits = (legal_pred > 0) == batch["legality_mask"]
            metrics[f"legality/{name}"] = {"bce": float(legal_loss.detach()), "bit_accuracy": float(legal_bits.float().mean()), "row_accuracy": float(legal_bits.all(-1).float().mean())}
        if add_baseline:
            _fixed_majority(metrics, batch, agent, modes)
        return torch.stack(losses).mean(), metrics

    def evaluate(dataset: TensorObservationDataset) -> tuple[float, dict]:
        probes.eval(); total_loss = 0.0; n_batches = 0
        totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float)); counts = defaultdict(int)
        with torch.no_grad():
            for batch in dataset.batches(args.batch_size, shuffle=False, seed=args.seed, device=device):
                loss, metrics = process(batch, True); total_loss += float(loss); n_batches += 1
                for group, item in metrics.items():
                    _accumulate(totals[group], item); counts[group] += 1
        return total_loss / max(1, n_batches), {group: _finalize(values, counts[group]) for group, values in totals.items()}

    rows = []; best_score = float("inf"); best_state = None; stale = 0
    for epoch in range(args.epochs):
        probes.train(); running = 0.0; batches = 0
        for batch in datasets["train"].batches(args.batch_size, shuffle=True, seed=args.seed + epoch, device=device):
            optimizer.zero_grad(set_to_none=True); loss, _ = process(batch, False); loss.backward()
            torch.nn.utils.clip_grad_norm_(probes.parameters(), 5.0); optimizer.step()
            running += float(loss.detach()); batches += 1
        team_loss, _ = evaluate(datasets["team_holdout"])
        battle_loss, _ = evaluate(datasets["battle_holdout"])
        score = (team_loss + battle_loss) / 2
        row = {"epoch": epoch + 1, "train_loss": running / max(1, batches), "team_holdout_loss": team_loss, "battle_holdout_loss": battle_loss}
        rows.append(row); print(f"[live-encoder-probe] {row}", flush=True)
        if score < best_score - args.min_delta:
            best_score = score; best_state = copy.deepcopy(probes.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("live encoder probe did not produce a finite best state")
    probes.load_state_dict(best_state)
    team_loss, team_report = evaluate(datasets["team_holdout"])
    battle_loss, battle_report = evaluate(datasets["battle_holdout"])
    agent_changed = any(parameter.grad is not None for parameter in agent.parameters())
    report = {
        "status": "complete", "checkpoint": str(args.checkpoint), "data": str(args.data),
        "agent_frozen": not agent_changed and not any(parameter.requires_grad for parameter in agent.parameters()),
        "fixed_majority_source": "training split only", "epochs": rows,
        "selected_holdout_loss": {"team": team_loss, "battle": battle_loss},
        "team_holdout": team_report, "battle_holdout": battle_report,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(best_state, args.out / "probe_best.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261051)
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
