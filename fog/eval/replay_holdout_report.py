"""Score a behaviour-cloning checkpoint on every row of a replay manifest.

    python -m fog.eval.replay_holdout_report --config configs/replay_blueprint_cpu.yaml \
        --checkpoint runs/blueprint/g1000/checkpoints/bc_best.pt \
        --manifest runs/blueprint/split/holdout_manifest.parquet --out report.json

``--untrained`` scores the freshly initialised model instead (seeded as ``fog.training.bc``
seeds it), which gives a learning curve its zero point.

Policy metrics use the rows the training loss would supervise (``_policy_supervision_mask`` with
the config's ``loss`` section). Value metrics use every row: the model value ``v`` in [-1, 1] and
the outcome (+1 win, -1 loss, 0 draw) are mapped to win probabilities ``(x + 1) / 2``.

Game phase: a shard holds one game, with each player's rows as a separate block, so row position
is not game time. The turn is read from the observation's field vector (stored as
``min(turn, 50) / 50``); a row's phase is ``(turn - 1) * 3 // last_turn`` where ``last_turn`` is
the largest stored turn in its game. Turns past 50 are stored as 50, so in games longer than 50
turns the rows after turn 50 all count as late.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

SWITCH_ACTION_START = 8
PHASES = ("early", "middle", "late")
TURN_CAP = 50


def _turn_column() -> int:
    from fog.encoding.dims import DIMS

    return (DIMS.field_weather + DIMS.field_terrain + DIMS.field_hazards
            + DIMS.field_screens + DIMS.field_trickroom)


def game_phase(turn: np.ndarray, game: np.ndarray) -> np.ndarray:
    """Phase 0/1/2 (first, middle, last third of the game's turns) for every row."""
    turn = np.asarray(turn, dtype=np.int64)
    game = np.asarray(game, dtype=np.int64)
    last = np.zeros(int(game.max()) + 1 if len(game) else 0, dtype=np.int64)
    np.maximum.at(last, game, turn)
    last_turn = np.maximum(last[game], 1)
    return np.clip((np.maximum(turn, 1) - 1) * 3 // last_turn, 0, 2)


def value_calibration(value: np.ndarray, outcome: np.ndarray, bins: int = 10) -> dict:
    """Brier score, MSE, equal-width reliability table and ECE for value predictions.

    ``value`` and ``outcome`` are on the [-1, 1] scale. Probabilities are ``(x + 1) / 2`` clipped to
    [0, 1]; ``p == 1`` falls in the last bin. Empty bins are reported with count 0 and null means
    and contribute nothing to the ECE.
    """
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    outcome = np.asarray(outcome, dtype=np.float64).reshape(-1)
    if value.shape != outcome.shape:
        raise ValueError("value and outcome must have the same length")
    if bins < 1:
        raise ValueError("bins must be positive")
    total = len(value)
    if total == 0:
        return {"rows": 0, "value_mse": None, "brier": None, "ece": None, "reliability": []}
    p = np.clip((value + 1.0) / 2.0, 0.0, 1.0)
    target = np.clip((outcome + 1.0) / 2.0, 0.0, 1.0)
    index = np.minimum((p * bins).astype(np.int64), bins - 1)
    table = []
    ece = 0.0
    for b in range(bins):
        selected = index == b
        count = int(selected.sum())
        row = {"bin": b, "lo": b / bins, "hi": (b + 1) / bins, "count": count,
               "mean_predicted": None, "observed_win_rate": None}
        if count:
            mean_predicted = float(p[selected].mean())
            observed = float(target[selected].mean())
            row["mean_predicted"] = mean_predicted
            row["observed_win_rate"] = observed
            ece += count / total * abs(mean_predicted - observed)
        table.append(row)
    return {
        "rows": total,
        "value_mse": float(np.mean((value - outcome) ** 2)),
        "brier": float(np.mean((p - target) ** 2)),
        "ece": float(ece),
        "reliability": table,
    }


def policy_summary(nll: np.ndarray, correct: np.ndarray, legal: np.ndarray,
                   action: np.ndarray) -> dict:
    """Top-1, NLL and chance-normalised lift, overall and split into move and switch rows."""
    nll = np.asarray(nll, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    legal = np.asarray(legal, dtype=np.float64)
    action = np.asarray(action)

    def summarise(selected: np.ndarray) -> dict:
        count = int(selected.sum())
        if not count:
            return {"rows": 0, "top1": None, "nll": None, "avg_legal_actions": None,
                    "top1_lift": None}
        top1 = float(correct[selected].mean())
        avg_legal = float(legal[selected].mean())
        return {"rows": count, "top1": top1, "nll": float(nll[selected].mean()),
                "avg_legal_actions": avg_legal, "top1_lift": top1 * avg_legal}

    everything = np.ones(len(nll), dtype=bool)
    summary = summarise(everything)
    summary["move"] = summarise(action < SWITCH_ACTION_START)
    summary["switch"] = summarise(action >= SWITCH_ACTION_START)
    return summary


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def score(cfg: dict, manifest: Path, checkpoint: Path | None, *, batch_size: int = 256) -> dict:
    import polars as pl
    import torch

    from fog.models.agent import FogAgent
    from fog.training.bc import _policy_supervision_mask
    from fog.training.dataset import StreamingReplayDataset

    started = time.perf_counter()
    torch.manual_seed(int(cfg.get("seed", 0)))
    agent = FogAgent.from_tables(cfg)
    if checkpoint is not None:
        agent.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    agent.eval()
    loss_cfg = cfg.get("loss", {})

    table = pl.read_parquet(manifest)
    rows_col = next(c for c in ("n_rows", "rows") if c in table.columns)
    counts = table[rows_col].to_list()
    game_of_row = np.repeat(np.arange(len(counts)), counts)
    turn_column = _turn_column()

    dataset = StreamingReplayDataset(manifest, shard_buffer=64)
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ("action", "nll", "correct", "legal", "policy", "value", "outcome",
                            "turn")
    }
    with torch.no_grad():
        for batch in dataset.batches(batch_size, shuffle=False, seed=0):
            out = agent(batch)
            action = batch["action"]
            nll = -out.log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            policy = _policy_supervision_mask(batch, loss_cfg) & torch.isfinite(nll)
            parts["action"].append(action.numpy())
            parts["nll"].append(nll.float().numpy())
            parts["correct"].append((out.log_probs.argmax(-1) == action).numpy())
            parts["legal"].append(batch["legality_mask"].float().sum(-1).numpy())
            parts["policy"].append(policy.numpy())
            parts["value"].append(out.value.float().reshape(-1).numpy())
            parts["outcome"].append(batch["outcome"].float().reshape(-1).numpy())
            parts["turn"].append(
                np.rint(batch["field"][:, turn_column].numpy() * TURN_CAP).astype(np.int64)
            )
    data = {key: np.concatenate(value) for key, value in parts.items()}
    rows = len(data["action"])
    if rows != len(game_of_row):
        raise RuntimeError(f"manifest lists {len(game_of_row)} rows but shards held {rows}")
    policy = data["policy"]

    report = {
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "untrained": checkpoint is None,
        "manifest": str(manifest),
        "games": len(table),
        "rows": rows,
        "policy_rows": int(policy.sum()),
        "commit": _commit(),
        "policy": policy_summary(data["nll"][policy], data["correct"][policy],
                                 data["legal"][policy], data["action"][policy]),
        "value": value_calibration(data["value"], data["outcome"]),
        "value_by_phase": {},
        "phase_definition": "(turn - 1) * 3 // last turn of the game; turn read from field, "
                            "capped at 50",
        "rows_at_turn_cap": int((data["turn"] >= TURN_CAP).sum()),
        "constant_half_brier_reference": 0.25,
    }
    phase = game_phase(data["turn"], game_of_row)
    for index, name in enumerate(PHASES):
        selected = phase == index
        calibration = value_calibration(data["value"][selected], data["outcome"][selected])
        report["value_by_phase"][name] = {
            "rows": calibration["rows"], "brier": calibration["brier"],
            "value_mse": calibration["value_mse"], "ece": calibration["ece"],
        }
    report["seconds"] = time.perf_counter() - started
    return report


def main() -> None:
    from fog.config import load_config_with_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--untrained", action="store_true",
                        help="score the freshly initialised model instead of a checkpoint")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = parser.parse_args()
    if args.untrained == (args.checkpoint is not None):
        parser.error("pass exactly one of --checkpoint or --untrained")
    cfg = load_config_with_args(args.config, args.overrides)
    report = score(cfg, args.manifest, args.checkpoint, batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    p, v = report["policy"], report["value"]
    print(
        f"rows={report['rows']} policy_rows={report['policy_rows']} top1={p['top1']:.4f} "
        f"lift={p['top1_lift']:.3f} nll={p['nll']:.4f} brier={v['brier']:.4f} "
        f"ece={v['ece']:.4f} seconds={report['seconds']:.1f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
