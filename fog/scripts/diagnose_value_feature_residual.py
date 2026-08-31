"""Test whether frozen critic features can fit reliable multi-seed continuation values.

This diagnostic reads two disjoint-team reports produced by
``fog.eval.value_continuation_target_control``.  It never loads or changes an agent.  Ridge
strength is selected using grouped cross-validation on the training report only; the heldout-team
report is evaluated once after selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _target(state: dict[str, Any]) -> float:
    outcomes = [
        float(row["outcome"])
        for row in state["continuations"]
        if row.get("outcome") is not None
    ]
    if not outcomes:
        raise ValueError("state has no valid continuation outcomes")
    return float(np.mean(outcomes))


def rows_from_report(report: dict[str, Any]) -> dict[str, Any]:
    states = report.get("states") or []
    if not states:
        raise ValueError("value target report has no states")
    features = np.asarray([state["value_feature"] for state in states], dtype=np.float64)
    baseline = np.asarray([state["value_prediction"] for state in states], dtype=np.float64)
    target = np.asarray([_target(state) for state in states], dtype=np.float64)
    groups = np.asarray([
        "-".join(map(str, sorted((state["learner_team_index"], state["opponent_team_index"]))))
        for state in states
    ])
    if features.ndim != 2 or features.shape[0] != baseline.size:
        raise ValueError("invalid frozen value feature matrix")
    progress = None
    if all(state.get("progress_score") is not None for state in states):
        progress = np.asarray([state["progress_score"] for state in states], dtype=np.float64)
    return {
        "features": features,
        "baseline": baseline,
        "target": target,
        "groups": groups,
        "progress": progress,
    }


def _standardize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(0)
    scale = matrix.std(0)
    scale[scale < 1e-8] = 1.0
    return (matrix - mean) / scale, mean, scale


def _ridge(matrix: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
    design = np.concatenate([matrix, np.ones((matrix.shape[0], 1))], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(strength)
    penalty[-1, -1] = 0.0
    system = design.T @ design + penalty
    rhs = design.T @ target
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def _predict_residual(
    features: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (features - mean) / scale
    residual = standardized @ weights[:-1] + weights[-1]
    return np.clip(baseline + residual, -1.0, 1.0)


def select_ridge_strength(
    features: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    strengths: list[float],
) -> tuple[float, dict[str, float]]:
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("grouped ridge selection needs at least two matchup groups")
    scores: dict[str, float] = {}
    for strength in strengths:
        squared = 0.0
        count = 0
        for group in unique:
            validation = groups == group
            training = ~validation
            train_x, mean, scale = _standardize_fit(features[training])
            weights = _ridge(train_x, target[training] - baseline[training], strength)
            prediction = _predict_residual(
                features[validation], baseline[validation], weights, mean, scale
            )
            squared += float(np.square(prediction - target[validation]).sum())
            count += int(validation.sum())
        scores[f"{strength:g}"] = squared / count
    selected = min(strengths, key=lambda value: (scores[f"{value:g}"], value))
    return float(selected), scores


def _correlation(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if prediction.size < 2 or prediction.std() < 1e-12 or target.std() < 1e-12:
        return None
    return float(np.corrcoef(prediction, target)[0, 1])


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    nonzero = target != 0
    return {
        "rows": int(target.size),
        "mse": float(np.square(error).mean()),
        "mae": float(np.abs(error).mean()),
        "correlation": _correlation(prediction, target),
        "sign_accuracy": float((np.sign(prediction) == np.sign(target)).mean()),
        "sign_accuracy_nonzero": (
            float((np.sign(prediction[nonzero]) == np.sign(target[nonzero])).mean())
            if nonzero.any() else None
        ),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
    }


def affine_logit_calibration(
    train_baseline: np.ndarray, train_target: np.ndarray, evaluation_baseline: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    train_logit = np.arctanh(np.clip(train_baseline, -0.999, 0.999))
    eval_logit = np.arctanh(np.clip(evaluation_baseline, -0.999, 0.999))
    design = np.stack([train_logit, np.ones_like(train_logit)], axis=1)
    coefficients = np.linalg.lstsq(design, train_target, rcond=None)[0]
    prediction = np.clip(coefficients[0] * eval_logit + coefficients[1], -1.0, 1.0)
    return prediction, {"slope": float(coefficients[0]), "intercept": float(coefficients[1])}


def affine_calibration(
    train_feature: np.ndarray, train_target: np.ndarray, evaluation_feature: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    design = np.stack([train_feature, np.ones_like(train_feature)], axis=1)
    coefficients = np.linalg.lstsq(design, train_target, rcond=None)[0]
    prediction = np.clip(
        coefficients[0] * evaluation_feature + coefficients[1], -1.0, 1.0
    )
    return prediction, {"slope": float(coefficients[0]), "intercept": float(coefficients[1])}


def joint_logit_progress_calibration(
    train_baseline: np.ndarray,
    train_progress: np.ndarray,
    train_target: np.ndarray,
    evaluation_baseline: np.ndarray,
    evaluation_progress: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    train_logit = np.arctanh(np.clip(train_baseline, -0.999, 0.999))
    evaluation_logit = np.arctanh(np.clip(evaluation_baseline, -0.999, 0.999))
    design = np.stack([train_logit, train_progress, np.ones_like(train_logit)], axis=1)
    coefficients = np.linalg.lstsq(design, train_target, rcond=None)[0]
    prediction = np.clip(
        coefficients[0] * evaluation_logit
        + coefficients[1] * evaluation_progress
        + coefficients[2],
        -1.0,
        1.0,
    )
    return prediction, {
        "logit_slope": float(coefficients[0]),
        "progress_slope": float(coefficients[1]),
        "intercept": float(coefficients[2]),
    }


def diagnose(train_report: dict[str, Any], holdout_report: dict[str, Any]) -> dict[str, Any]:
    train_teams = set(train_report["selected_team_indices"])
    holdout_teams = set(holdout_report["selected_team_indices"])
    if train_teams & holdout_teams:
        raise ValueError("training and heldout value-target teams overlap")
    train = rows_from_report(train_report)
    holdout = rows_from_report(holdout_report)
    strengths = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
    selected, cv = select_ridge_strength(
        train["features"], train["baseline"], train["target"], train["groups"], strengths
    )
    standardized, mean, scale = _standardize_fit(train["features"])
    weights = _ridge(standardized, train["target"] - train["baseline"], selected)
    feature_prediction = _predict_residual(
        holdout["features"], holdout["baseline"], weights, mean, scale
    )
    logit_prediction, logit_coefficients = affine_logit_calibration(
        train["baseline"], train["target"], holdout["baseline"]
    )
    constant_prediction = np.full_like(holdout["target"], train["target"].mean())
    heldout = {
        "baseline": metrics(holdout["baseline"], holdout["target"]),
        "constant_train_mean": metrics(constant_prediction, holdout["target"]),
        "affine_logit": metrics(logit_prediction, holdout["target"]),
        "feature_residual": metrics(feature_prediction, holdout["target"]),
    }
    progress_calibration = None
    joint_calibration = None
    if train["progress"] is not None and holdout["progress"] is not None:
        progress_prediction, progress_calibration = affine_calibration(
            train["progress"], train["target"], holdout["progress"]
        )
        joint_prediction, joint_calibration = joint_logit_progress_calibration(
            train["baseline"], train["progress"], train["target"],
            holdout["baseline"], holdout["progress"],
        )
        heldout["affine_public_progress"] = metrics(progress_prediction, holdout["target"])
        heldout["affine_logit_plus_progress"] = metrics(joint_prediction, holdout["target"])
    return {
        "status": "complete",
        "diagnostic_only": True,
        "agent_checkpoint_changed": False,
        "train_rows": int(train["target"].size),
        "holdout_rows": int(holdout["target"].size),
        "train_matchup_groups": int(np.unique(train["groups"]).size),
        "holdout_matchup_groups": int(np.unique(holdout["groups"]).size),
        "train_team_count": len(train_teams),
        "holdout_team_count": len(holdout_teams),
        "team_overlap": 0,
        "selected_ridge_strength": selected,
        "ridge_cv_mse": cv,
        "residual_weight_norm": float(np.linalg.norm(weights[:-1])),
        "residual_intercept": float(weights[-1]),
        "logit_calibration": logit_coefficients,
        "progress_calibration": progress_calibration,
        "joint_logit_progress_calibration": joint_calibration,
        "heldout": heldout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--holdout-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train = json.loads(args.train_report.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout_report.read_text(encoding="utf-8"))
    report = diagnose(train, holdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
