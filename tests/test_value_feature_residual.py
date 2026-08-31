import numpy as np

from fog.scripts.diagnose_value_feature_residual import (
    affine_calibration,
    joint_logit_progress_calibration,
    metrics,
    select_ridge_strength,
)


def test_grouped_ridge_selects_a_finite_model_and_improves_signal():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(24, 4))
    baseline = np.zeros(24)
    target = np.clip(features[:, 0] * 0.5, -1.0, 1.0)
    groups = np.asarray([f"g{index // 4}" for index in range(24)])
    selected, scores = select_ridge_strength(
        features, baseline, target, groups, [0.01, 1.0, 100.0]
    )
    assert selected in {0.01, 1.0, 100.0}
    assert all(np.isfinite(value) for value in scores.values())
    assert min(scores.values()) < float(np.square(target).mean())


def test_metrics_reports_sign_accuracy_without_dividing_on_zero_targets():
    result = metrics(np.asarray([0.5, -0.5, 0.0]), np.asarray([1.0, -1.0, 0.0]))
    assert result["sign_accuracy"] == 1.0
    assert result["sign_accuracy_nonzero"] == 1.0


def test_progress_calibrations_recover_linear_signal():
    progress = np.asarray([-1.0, -0.5, 0.5, 1.0])
    target = 0.8 * progress - 0.1
    prediction, coefficients = affine_calibration(progress, target, progress)
    assert np.allclose(prediction, target)
    np.testing.assert_allclose(coefficients["slope"], 0.8, atol=1e-10)

    baseline = np.tanh(np.asarray([-0.4, -0.2, 0.2, 0.4]))
    combined_target = 0.25 * np.arctanh(baseline) + 0.5 * progress
    combined, joint = joint_logit_progress_calibration(
        baseline, progress, combined_target, baseline, progress
    )
    assert np.allclose(combined, combined_target)
    assert np.isfinite(list(joint.values())).all()
