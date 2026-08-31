"""Fixed-corpus replay-censored overfit gate for the general OU belief model.

This is intentionally *not* an eight-team classifier.  Collection runs the ordinary direct
Showdown observation path, and training receives only public-prefix observations plus facts that
the completed public protocol eventually revealed.  Pool candidates and complete pastes are used
only by the evaluator to decide whether a particular arbitrary query has one possible answer.

The gate answers a narrow implementation question before expensive training: can the general
queryable autoregressive model repeatedly fit a small censored replay corpus until a truly
identifiable hidden target receives at least 98% probability?
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fog.constants.schema import NO_ITEM, SET_EV_COLS, SET_NATURE_COL, SET_TERA_COL
from fog.constants.tables import TABLES
from fog.models.belief import SLOT_ATTRS

_SCALAR_COLUMNS = {
    "species": 0, "item": 1, "ability": 2,
    "nature": SET_NATURE_COL, "tera": SET_TERA_COL,
}
_MOVE_COLUMNS = (3, 4, 5, 6)
_EV_COLUMNS = SET_EV_COLS
_REQUIRED_IDENTIFIABLE_FAMILIES = ("item", "ability", "move", "tera")
_GATE_PROBABILITY = 0.98
_GATE_CONFIDENT_RATE = 0.95
_GATE_MINIMUM_QUERIES = 32
_MODEL_KEYS = (
    "entities", "entity_ids", "field", "history", "history_ids", "history_state",
    "history_len", "active_idx", "legality_mask", "opp_revealed",
    "belief_targets", "belief_valid", "turn",
)


def _load_compatible(agent, path: str | Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    return agent.load_compatible_state_dict(state)


def collect_corpus(
    cfg: dict,
    *,
    games: int,
    seed: int,
    actor_ckpt: str | Path,
    device: torch.device,
) -> tuple[dict, dict[str, Any]]:
    """Collect both public perspectives of fixed direct-Showdown games."""
    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import bind_fog_player, play_local_battle
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import load_pool, to_packed
    from fog.models.agent import FogAgent
    from fog.training.replay_supervision import oracle_censored_truth
    from fog.training.rollout import RolloutBuffer

    pool = load_pool(cfg["env"]["team_pool"])
    if len(pool) != 8:
        raise ValueError(f"the controlled gate expects exactly 8 teams, found {len(pool)}")
    packed = [to_packed(paste) for paste in pool]
    truth = [truth_tokens_from_paste(paste, TABLES) for paste in pool]

    actor = FogAgent.from_tables(cfg).to(device)
    compatibility = _load_compatible(actor, actor_ckpt, device)
    actor.eval()
    buffer = RolloutBuffer(gamma=1.0, lam=1.0, beta=0.0)
    p1 = bind_fog_player(FogPlayer(actor, temperature=1.35))
    p2 = bind_fog_player(FogPlayer(actor, temperature=1.35))
    for adapter in (p1, p2):
        adapter.attach_buffer(buffer)
        adapter.record_mirror = False

    current: dict[str, np.ndarray | None] = {"p1": None, "p2": None}
    p1.truth_provider = lambda battle: oracle_censored_truth(
        battle, current["p2"], TABLES
    )
    p2.truth_provider = lambda battle: oracle_censored_truth(
        battle, current["p1"], TABLES
    )

    rng = random.Random(seed)
    results = []
    with ShowdownDirectClient(
        **({"showdown_root": cfg["env"]["showdown_root"]}
           if cfg.get("env", {}).get("showdown_root") else {})
    ) as client:
        for game in range(games):
            p1_index = game % len(pool)
            p2_index = (5 * game + 1 + game // len(pool)) % len(pool)
            if p2_index == p1_index:
                p2_index = (p2_index + 1) % len(pool)
            current["p1"], current["p2"] = truth[p1_index], truth[p2_index]
            result = play_local_battle(
                client,
                {
                    "formatid": cfg.get("env", {}).get("battle_format", "gen9ou"),
                    "seed": f"sodium,{rng.getrandbits(256):064x}",
                    "strictChoices": True,
                    "p1": {"name": f"ReplayFitA{game}", "team": packed[p1_index]},
                    "p2": {"name": f"ReplayFitB{game}", "team": packed[p2_index]},
                },
                p1,
                p2,
                seed=rng.randrange(2**31),
                battle_tag=f"battle-replay-overfit-{game}",
            )
            results.append({
                "game": game,
                "p1_team": p1_index,
                "p2_team": p2_index,
                "turns": result.turns,
                "winner": result.winner,
            })

    batch = buffer.to_batch("cpu")
    batch.pop("preview", None)
    summary = {
        "games": games,
        "perspectives": 2 * games,
        "rows": int(batch["actions"].shape[0]),
        "hidden_targets": {
            family: int(mask.sum()) for family, mask in batch["belief_valid"].items()
        },
        "actor_compatibility": {
            "loaded": len(compatibility["loaded"]),
            "initialized": len(compatibility["initialized"]),
        },
        "schedule": results,
    }
    return batch, summary


def _index_batch(batch: dict, indices: torch.Tensor, device: torch.device) -> dict:
    from fog.training.ppo import _index_nested

    return {
        key: _index_nested(batch[key], indices).to(device)
        if isinstance(batch[key], torch.Tensor)
        else {
            name: (
                value[indices].to(device)
                if isinstance(value, torch.Tensor)
                else {
                    sub_name: sub_value[indices].to(device)
                    for sub_name, sub_value in value.items()
                }
            )
            for name, value in batch[key].items()
        }
        for key in _MODEL_KEYS
        if key in batch
    }


def _candidate_consistent(candidate: np.ndarray, evidence: np.ndarray, no_item: int) -> bool:
    for slot in range(6):
        for column in (0, 1, 2, 7, 8):
            value = int(evidence[slot, column])
            if value < 0 or (column == 1 and value == no_item):
                continue
            if int(candidate[slot, column]) != value:
                return False
        known_moves = {int(move) for move in evidence[slot, _MOVE_COLUMNS] if move >= 0}
        candidate_moves = {int(move) for move in candidate[slot, _MOVE_COLUMNS] if move >= 0}
        if not known_moves.issubset(candidate_moves):
            return False
    return True


def _identifiable_mask(
    public: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
    *,
    family: str,
    slot: int,
    subslot: int | None,
) -> np.ndarray:
    """Rows where all pool candidates consistent with every other known fact agree on this query."""
    no_item = TABLES.item_index(NO_ITEM)
    result = np.zeros(len(public), dtype=bool)
    column = (
        _MOVE_COLUMNS[subslot]
        if family == "move"
        else (_EV_COLUMNS[subslot] if family == "ev" else _SCALAR_COLUMNS[family])
    )
    for row in range(len(public)):
        target = int(labels[row, slot, column])
        if target < 0:
            continue
        evidence = np.asarray(public[row]).copy()
        # Add all recoverable future facts as generated assumptions, except the addressed target.
        for s in range(6):
            for c in range(len(SLOT_ATTRS)):
                if labels[row, s, c] >= 0:
                    original_public_hidden = evidence[s, c] < 0 or (
                        c == 1 and evidence[s, c] == no_item
                    )
                    if original_public_hidden:
                        evidence[s, c] = labels[row, s, c]
        evidence[slot, column] = -1
        consistent = [
            candidate for candidate in candidates
            if _candidate_consistent(candidate, evidence, no_item)
        ]
        if not consistent:
            continue
        if family == "move":
            known = {int(move) for move in evidence[slot, _MOVE_COLUMNS] if move >= 0}
            possible = {
                int(move)
                for candidate in consistent
                for move in candidate[slot, _MOVE_COLUMNS]
                if int(move) >= 0 and int(move) not in known
            }
        else:
            possible = {int(candidate[slot, column]) for candidate in consistent}
        result[row] = possible == {target}
    return result


def _move_context(labels: torch.Tensor, public: torch.Tensor, slot: int, subslot: int) -> torch.Tensor:
    """Pack all other known moves as an unordered assumption set for ``query_field``."""
    context = labels.clone()
    for row in range(len(context)):
        target = int(labels[row, slot, _MOVE_COLUMNS[subslot]])
        public_moves = [
            int(move) for move in public[row, slot, _MOVE_COLUMNS] if int(move) >= 0
        ]
        assumptions = [
            int(move)
            for index, move in enumerate(labels[row, slot, _MOVE_COLUMNS])
            if index != subslot and int(move) >= 0 and int(move) not in public_moves
        ]
        context[row, slot, list(_MOVE_COLUMNS)] = -1
        start = len(public_moves)
        for offset, move in enumerate(assumptions[: 4 - start]):
            context[row, slot, _MOVE_COLUMNS[start + offset]] = move
        # ``target`` is intentionally absent even when another storage order would place it first.
        assert target not in {
            int(move) for move in context[row, slot, _MOVE_COLUMNS] if int(move) >= 0
        }
    return context


@torch.no_grad()
def evaluate_queries(
    agent,
    batch: dict,
    candidates: np.ndarray,
    *,
    device: torch.device,
    max_rows: int = 512,
) -> dict[str, Any]:
    """Direct arbitrary-query calibration, with pool enumeration used only for identifiability."""
    from fog.models.self_belief import tokens_from_labels

    agent.eval()
    n = min(max_rows, int(batch["actions"].shape[0]))
    indices = torch.arange(n)
    model_batch = _index_batch(batch, indices, device)
    contexts = agent.encode_contexts(model_batch)
    belief_context = agent.belief_context(contexts)
    labels = tokens_from_labels(model_batch["belief_targets"])
    public = model_batch["opp_revealed"]["tokens"]
    valid_by_family = model_batch["belief_valid"]
    public_np = public.cpu().numpy()
    labels_np = labels.cpu().numpy()

    all_probability: list[torch.Tensor] = []
    identifiable_probability: list[torch.Tensor] = []
    family_stats: dict[str, dict[str, float]] = {}
    groups = [
        (family, slot, None)
        for family in _SCALAR_COLUMNS
        for slot in range(6)
    ] + [
        ("move", slot, subslot)
        for slot in range(6)
        for subslot in range(4)
    ] + [
        ("ev", slot, subslot)
        for slot in range(6)
        for subslot in range(6)
    ]
    for family, slot, subslot in groups:
        if family == "move":
            target = labels[:, slot, _MOVE_COLUMNS[subslot]]
            valid = valid_by_family["move"][:, slot, subslot]
            context = _move_context(labels, public, slot, int(subslot))
        elif family == "ev":
            column = _EV_COLUMNS[subslot]
            target = labels[:, slot, column]
            valid = valid_by_family["ev"][:, slot, subslot]
            context = labels.clone()
            context[:, slot, column] = -1
        else:
            column = _SCALAR_COLUMNS[family]
            target = labels[:, slot, column]
            valid = valid_by_family[family][:, slot]
            context = labels.clone()
            context[:, slot, column] = -1
        if not bool(valid.any()):
            continue
        logits = agent.belief.query_field(
            belief_context,
            model_batch["opp_revealed"],
            slot=slot,
            field=family,
            context_values=context,
            subslot=subslot if family == "ev" else None,
        )
        probability = torch.softmax(logits, dim=-1).gather(
            1, target.clamp(min=0, max=logits.shape[-1] - 1).unsqueeze(1)
        ).squeeze(1)
        all_probability.append(probability[valid])
        identifiable = torch.as_tensor(
            _identifiable_mask(
                public_np,
                labels_np,
                candidates,
                family=family,
                slot=slot,
                subslot=subslot,
            ),
            dtype=torch.bool,
            device=device,
        ) & valid
        if bool(identifiable.any()):
            values = probability[identifiable]
            identifiable_probability.append(values)
            stats = family_stats.setdefault(
                family,
                {"sum": 0.0, "count": 0.0, "p98": 0.0, "min": float("inf")},
            )
            stats["sum"] += float(values.sum())
            stats["count"] += float(values.numel())
            stats["p98"] += float((values >= 0.98).sum())
            stats["min"] = min(stats["min"], float(values.min()))

    def summarize(parts: list[torch.Tensor]) -> dict[str, float]:
        if not parts:
            return {"count": 0, "mean_true_probability": 0.0, "confident98_rate": 0.0}
        values = torch.cat(parts)
        return {
            "count": int(values.numel()),
            "mean_true_probability": float(values.mean()),
            "confident98_rate": float((values >= 0.98).float().mean()),
            "min_true_probability": float(values.min()),
        }

    by_family = {
        family: {
            "count": int(stats["count"]),
            "mean_true_probability": stats["sum"] / max(1.0, stats["count"]),
            "confident98_rate": stats["p98"] / max(1.0, stats["count"]),
            "min_true_probability": stats["min"],
        }
        for family, stats in family_stats.items()
    }
    return {
        "all_recoverable_leave_one_out": summarize(all_probability),
        "identifiable_leave_one_out": summarize(identifiable_probability),
        "identifiable_by_family": by_family,
    }


def _passes_identifiable_gate(evaluation: dict[str, Any]) -> bool:
    """Require every replay-observable family to pass; easy fields cannot hide a weak one."""
    aggregate = evaluation["identifiable_leave_one_out"]
    if not (
        aggregate["mean_true_probability"] >= _GATE_PROBABILITY
        and aggregate["confident98_rate"] >= _GATE_CONFIDENT_RATE
        and aggregate["count"] >= _GATE_MINIMUM_QUERIES
    ):
        return False
    by_family = evaluation["identifiable_by_family"]
    for family in _REQUIRED_IDENTIFIABLE_FAMILIES:
        stats = by_family.get(family)
        if stats is None or not (
            stats["mean_true_probability"] >= _GATE_PROBABILITY
            and stats["confident98_rate"] >= _GATE_CONFIDENT_RATE
            and stats["count"] >= _GATE_MINIMUM_QUERIES
        ):
            return False
    return True


def train_overfit(
    cfg: dict,
    batch: dict,
    *,
    init_ckpt: str | Path,
    pool: list[str],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, Any]:
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.models.agent import FogAgent
    from fog.training.losses import belief_loss, belief_output_loss

    candidates = np.stack([truth_tokens_from_paste(paste, TABLES) for paste in pool])
    agent = FogAgent.from_tables(cfg).to(device)
    compatibility = _load_compatible(agent, init_ckpt, device)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    for module in (agent.backbone, agent.belief):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in agent.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    rng = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 991)
    n = int(batch["actions"].shape[0])
    series = []
    initial = evaluate_queries(agent, batch, candidates, device=device)
    series.append({"epoch": 0, **initial})
    best_probability = initial["identifiable_leave_one_out"]["mean_true_probability"]
    best_path = out_dir / "checkpoints" / "belief_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), best_path)

    for epoch in range(1, epochs + 1):
        agent.train()
        permutation = torch.randperm(n, generator=rng)
        running_marginal = running_chain = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size]
            minibatch = _index_batch(batch, indices, device)
            output = agent(minibatch)
            marginal = belief_loss(
                output.belief.logits,
                minibatch["belief_targets"],
                minibatch["belief_valid"],
            )
            chain = belief_output_loss(
                output.belief_chain,
                minibatch["belief_targets"],
                minibatch["belief_valid"],
            )
            loss = marginal + 2.0 * chain
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in agent.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            running_marginal += float(marginal.detach())
            running_chain += float(chain.detach())
            steps += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            evaluation = evaluate_queries(agent, batch, candidates, device=device)
            identified = evaluation["identifiable_leave_one_out"]
            row = {
                "epoch": epoch,
                "marginal_loss": running_marginal / max(1, steps),
                "chain_loss": running_chain / max(1, steps),
                **evaluation,
            }
            series.append(row)
            probability = identified["mean_true_probability"]
            print(
                f"[replay-overfit] epoch={epoch:03d} marginal={row['marginal_loss']:.4f} "
                f"chain={row['chain_loss']:.4f} identifiable_p={probability:.4f} "
                f"p>=.98={identified['confident98_rate']:.3f} n={identified['count']}",
                flush=True,
            )
            if probability > best_probability:
                best_probability = probability
                torch.save(agent.state_dict(), best_path)
            if _passes_identifiable_gate(evaluation):
                break

    final_path = out_dir / "checkpoints" / "belief_final.pt"
    torch.save(agent.state_dict(), final_path)
    return {
        "passed": _passes_identifiable_gate(series[-1]),
        "gate": {
            "mean_true_probability": _GATE_PROBABILITY,
            "confident98_rate": _GATE_CONFIDENT_RATE,
            "minimum_queries": _GATE_MINIMUM_QUERIES,
            "required_families": list(_REQUIRED_IDENTIFIABLE_FAMILIES),
            "scope": "aggregate_and_each_required_family",
        },
        "compatibility": {
            "loaded": len(compatibility["loaded"]),
            "initialized": len(compatibility["initialized"]),
        },
        "series": series,
        "best_checkpoint": str(best_path.resolve()),
        "final_checkpoint": str(final_path.resolve()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from fog.config import load_config
    from fog.env.teams import load_pool
    from fog.training.preflight import require_preflight

    cfg = load_config(args.config)
    require_preflight(cfg)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch, collection = collect_corpus(
        cfg,
        games=args.games,
        seed=args.seed,
        actor_ckpt=args.actor_ckpt,
        device=device,
    )
    corpus_path = out_dir / "replay_corpus.pt"
    torch.save(batch, corpus_path)
    report = {
        "design": "general arbitrary-query set-AR; no pool identity/candidates in training",
        "backend": "showdown_direct",
        "supervision": "public replay-censored original-set facts",
        "device": str(device),
        "collection": collection,
        "training": train_overfit(
            cfg,
            batch,
            init_ckpt=args.init_ckpt,
            pool=load_pool(cfg["env"]["team_pool"]),
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        ),
        "corpus": str(corpus_path.resolve()),
    }
    report_path = out_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[replay-overfit] report -> {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/set_ar_overfit8.yaml")
    parser.add_argument("--actor-ckpt", required=True)
    parser.add_argument("--init-ckpt", required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/set_ar_overfit8_replay_fixed")
    args = parser.parse_args()
    report = run(args)
    raise SystemExit(0 if report["training"]["passed"] else 1)


if __name__ == "__main__":
    main()
