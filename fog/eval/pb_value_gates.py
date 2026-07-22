"""Measured Phase-G4 gates for a trained public-belief value model.

The trainer's held-out error answers only whether the model predicts unseen game outcomes.  This
module checks the design-specific claims on the same game-level holdout: whether candidate/belief
inputs beat a separately trained public-only control, whether changing a real belief changes the
value table, whether real candidate permutations preserve values, and whether the played public
leaf is at least as predictive as the pre-action root board.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(probability))
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = ((probability >= lower) & (probability < upper))
        if index == bins - 1:
            mask |= probability == upper
        if mask.any():
            value += float(mask.sum()) / total * abs(
                float(probability[mask].mean()) - float(target[mask].mean())
            )
    return value


def main() -> None:
    import torch

    from fog.config import load_config_with_args
    from fog.env.engine_runner import _stack_obs
    from fog.models.agent import FogAgent
    from fog.models.pb_value import PBValueNet
    from fog.training.pb_value_train import build_pb_value_rows
    from fog.training.value_finetune import _split_by_game

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--ablation-report", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config_with_args(args.config)
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    agent.eval()
    net = PBValueNet.from_tables(
        state_dim=int(agent.backbone.state_dim), dyn_dim=int(agent.backbone.dyn_dim)
    ).to(device)
    net.load_state_dict(torch.load(args.net, map_location=device))
    net.eval()

    rows = build_pb_value_rows(args.data)
    games = np.asarray([row["game"] for row in rows], dtype=np.int64)
    _, holdout = _split_by_game(games, args.holdout, args.seed)
    selected = [rows[int(index)] for index in holdout]

    with torch.no_grad():
        leaf_s, leaf_c = agent.head_inputs(_stack_obs([row["obs"] for row in selected], device))
        root_s, root_c = agent.head_inputs(
            _stack_obs([row["root_obs"] for row in selected], device)
        )

    actual_predictions: list[float] = []
    uniform_predictions: list[float] = []
    root_predictions: list[float] = []
    responsiveness_true: list[float] = []
    responsiveness_table: list[float] = []
    pair_spread: list[float] = []
    permutation_errors: list[float] = []
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for local_index, row in enumerate(selected):
            own = torch.as_tensor(row["own_tokens"], dtype=torch.long, device=device)
            opp = torch.as_tensor(row["opp_tokens"], dtype=torch.long, device=device)
            belief_np = np.asarray(row["belief"], dtype=np.float32)
            support = belief_np > 0.0
            uniform_np = support.astype(np.float32)
            uniform_np /= uniform_np.sum()
            belief = torch.as_tensor(belief_np[None], device=device)
            uniform = torch.as_tensor(uniform_np[None], device=device)

            value = net(
                leaf_s[local_index:local_index + 1],
                leaf_c[local_index:local_index + 1], own, opp, belief,
            )[0]
            value_uniform = net(
                leaf_s[local_index:local_index + 1],
                leaf_c[local_index:local_index + 1], own, opp, uniform,
            )[0]
            value_root = net(
                root_s[local_index:local_index + 1],
                root_c[local_index:local_index + 1], own, opp, belief,
            )[0]
            k, j = int(row["k_star"]), int(row["j_star"])
            actual_predictions.append(float(value[k, j]))
            uniform_predictions.append(float(value_uniform[k, j]))
            root_predictions.append(float(value_root[k, j]))
            responsiveness_true.append(float(abs(value[k, j] - value_uniform[k, j])))
            responsiveness_table.append(float((value - value_uniform).abs().mean()))
            pair_spread.append(float(value.std(unbiased=False)))

            if len(permutation_errors) < 64:
                own_order = torch.as_tensor(rng.permutation(len(own)), device=device)
                opp_order = torch.as_tensor(rng.permutation(len(opp)), device=device)
                permuted = net(
                    leaf_s[local_index:local_index + 1],
                    leaf_c[local_index:local_index + 1],
                    own[own_order], opp[opp_order], belief[:, own_order][:, :, opp_order],
                )[0]
                expected = value[own_order][:, opp_order]
                permutation_errors.append(float((permuted - expected).abs().max()))

    target = np.asarray([row["outcome"] for row in selected], dtype=np.float64)
    actual = np.asarray(actual_predictions)
    uniform = np.asarray(uniform_predictions)
    root = np.asarray(root_predictions)
    one_sided = np.asarray([bool(row["one_sided"]) for row in selected])
    probability = np.clip((actual + 1.0) / 2.0, 0.0, 1.0)
    binary_target = (target + 1.0) / 2.0
    ablation = json.loads(Path(args.ablation_report).read_text(encoding="utf-8"))

    leaf_mse = float(np.mean((actual - target) ** 2))
    root_mse = float(np.mean((root - target) ** 2))
    report = {
        "n_holdout_rows": len(selected),
        "n_holdout_games": int(len(np.unique(games[holdout]))),
        "leaf_holdout_mse": leaf_mse,
        "support_uniform_input_mse": float(np.mean((uniform - target) ** 2)),
        "root_board_with_leaf_belief_mse": root_mse,
        "public_only_retrained_holdout_mse": ablation["holdout_mse_after"],
        "outcome_probability_ece_10_bins": _ece(probability, binary_target),
        "outcome_probability_brier": float(np.mean((probability - binary_target) ** 2)),
        "belief_responsiveness_true_cell_mean_abs": float(np.mean(responsiveness_true)),
        "belief_responsiveness_table_mean_abs": float(np.mean(responsiveness_table)),
        "belief_responsiveness_fraction_true_cell_gt_1e_3": float(
            np.mean(np.asarray(responsiveness_true) > 1e-3)
        ),
        "pair_value_std_mean": float(np.mean(pair_spread)),
        "real_particle_permutation_max_abs_error": max(permutation_errors, default=0.0),
        "pending_replacement_rows": int(one_sided.sum()),
        "pending_replacement_mse": (
            float(np.mean((actual[one_sided] - target[one_sided]) ** 2))
            if one_sided.any() else None
        ),
        "ordinary_rows_mse": (
            float(np.mean((actual[~one_sided] - target[~one_sided]) ** 2))
            if (~one_sided).any() else None
        ),
        "aware_query_calibration": (
            "not_run: logged queried beliefs have no matched counterfactual leaf/outcome target; "
            "the iterative loop is therefore not being skipped"
        ),
        "gates": {
            "beats_predict_mean": bool(
                leaf_mse < ablation["holdout_mse_predict_mean_baseline"]
            ),
            "beats_retrained_public_only": bool(
                leaf_mse < ablation["holdout_mse_after"]
            ),
            "responds_to_real_belief": bool(np.mean(responsiveness_table) > 1e-4),
            "particle_permutation_equivariant": bool(
                max(permutation_errors, default=0.0) < 1e-5
            ),
            "leaf_not_worse_than_pre_action_root": bool(leaf_mse <= root_mse),
        },
        "config": vars(args),
    }
    report["gates"]["all_binding_pass"] = all(report["gates"].values())
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
