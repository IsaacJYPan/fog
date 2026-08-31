"""Create and verify a rule-guided model without modifying the retained starting checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _deep_copy_json(value):
    return json.loads(json.dumps(value))


def main() -> None:
    import torch

    from fog.config import load_config
    from fog.models.agent import FogAgent
    from fog.training.heuristic_distill import collect_rows
    from fog.training.rollout import _stack_frags

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--verify-games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    base_cfg = _deep_copy_json(load_config(args.config))
    cfg = _deep_copy_json(base_cfg)
    cfg.setdefault("model", {})["use_action_features"] = True
    cfg["model"]["simple_rule_prior_strength"] = float(args.prior_strength)
    cfg.setdefault("log", {})["dir"] = str(args.out)
    (args.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    starting_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FogAgent.from_tables(cfg).cpu()
    compatibility = model.load_compatible_state_dict(starting_state)
    model.eval()
    guided_state = model.state_dict()
    changed_starting_tensors = [
        name for name, value in starting_state.items()
        if name not in guided_state or not torch.equal(value.cpu(), guided_state[name].cpu())
    ]
    if changed_starting_tensors:
        raise RuntimeError(
            "starting checkpoint tensors changed while adding the rule guide: "
            f"{changed_starting_tensors[:8]}"
        )
    new_tensors = sorted(set(guided_state) - set(starting_state))
    expected_prefix = "policy_value.action_scorer."
    if not new_tensors or any(not name.startswith(expected_prefix) for name in new_tensors):
        raise RuntimeError(f"unexpected new tensors: {new_tensors}")

    checkpoint_path = args.out / "rule_guided_start.pt"
    torch.save(guided_state, checkpoint_path)

    rows = collect_rows(
        base_cfg,
        args.checkpoint,
        games=args.verify_games,
        workers=args.workers,
        actor_threads=2,
        seed=args.seed,
    )
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    verify_model = FogAgent.from_tables(cfg).to(device)
    verify_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    verify_model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(rows), 512):
            part = rows[start : start + 512]
            batch = _stack_frags([row["frag"] for row in part], device=device)
            predictions.extend(verify_model(batch).log_probs.argmax(-1).cpu().tolist())
    targets = [int(row["action"]) for row in rows]
    matches = sum(a == b for a, b in zip(predictions, targets, strict=True))
    legal = sum(bool(row["frag"]["legality_mask"][prediction]) for row, prediction in zip(rows, predictions, strict=True))
    if matches != len(rows) or legal != len(rows):
        raise RuntimeError(
            f"rule-guided parity failed: matches={matches}/{len(rows)} legal={legal}/{len(rows)}"
        )

    report = {
        "status": "pass",
        "starting_checkpoint": str(args.checkpoint),
        "checkpoint": str(checkpoint_path),
        "prior_strength": args.prior_strength,
        "parameter_count": model.n_params(),
        "starting_tensors_byte_identical": True,
        "new_tensors": new_tensors,
        "compatibility": compatibility,
        "verification_games": args.verify_games,
        "verification_rows": len(rows),
        "rule_action_matches": matches,
        "rule_action_match_rate": matches / len(rows) if rows else None,
        "all_predictions_legal": legal == len(rows),
        "validation_teams_used": False,
        "seed": args.seed,
        "next_step": "Use this as the safe starting behavior; train only evidence-backed residual changes and reject any checkpoint that falls below the rule player.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
