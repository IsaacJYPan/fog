from __future__ import annotations

import pytest

from fog.training.dagger_heuristic_distill import (
    _collection_opponent_selector,
    _policy_output_parameters,
    action_family,
    conditional_family_log_probs,
    consensus_rows,
    correction_weight,
    summarize_corrections,
    summarize_policy_shift,
    summarize_sampling_probabilities,
    summarize_teacher_agreement,
    within_family_eligible,
)


def test_policy_output_parameters_follow_the_active_policy_route() -> None:
    pytest.importorskip("torch")
    from fog.models.heads import PolicyValueHead

    legacy = PolicyValueHead(state_dim=8, dyn_dim=8)
    assert {id(parameter) for parameter in _policy_output_parameters(legacy)} == {
        id(parameter) for parameter in legacy.policy[-1].parameters()
    }

    typed = PolicyValueHead(
        state_dim=8,
        dyn_dim=8,
        action_context_dim=5,
        shared_state_dim=16,
        shared_token_dim=8,
        action_query_policy=True,
        action_query_heads=2,
    )
    selected = _policy_output_parameters(typed)
    assert {id(parameter) for parameter in selected} == {
        id(parameter) for parameter in typed.action_query_scorer[-1].parameters()
    }
    for parameter in typed.parameters():
        parameter.requires_grad_(False)
    for parameter in selected:
        parameter.requires_grad_(True)
    assert all(parameter.requires_grad for parameter in selected)
    assert not any(parameter.requires_grad for parameter in typed.policy.parameters())


def test_summarize_teacher_agreement_uses_identical_rows() -> None:
    rows = [
        {"target": 0, "simple_target": 0, "maxdamage_target": 1},
        {"target": 8, "simple_target": 9, "maxdamage_target": 0},
    ]
    report = summarize_teacher_agreement(rows)
    assert report["rows"] == 2
    assert report["checkpoint_vs_simple"]["exact_agreement"] == pytest.approx(0.5)
    assert report["checkpoint_vs_simple"]["family_agreement"] == pytest.approx(1.0)
    assert report["checkpoint_vs_maxdamage"]["family_agreement"] == pytest.approx(0.5)


def test_consensus_rows_selects_exact_checkpoint_and_scripted_agreement() -> None:
    rows = [
        {"target": 0, "simple_target": 0, "maxdamage_target": 1},
        {"target": 8, "simple_target": 9, "maxdamage_target": 8},
    ]
    assert consensus_rows(rows, "simple") == [rows[0]]
    assert consensus_rows(rows, "maxdamage") == [rows[1]]
    with pytest.raises(ValueError, match="unsupported"):
        consensus_rows(rows, "unknown")


def test_collection_opponents_are_balanced_within_each_worker() -> None:
    workers = 4
    for worker_id in range(workers):
        assigned = [worker_id + workers * index for index in range(8)]
        selectors = [_collection_opponent_selector(game, workers) for game in assigned]
        assert selectors == [0, 1, 2, 3, 0, 1, 2, 3]


def test_collection_opponent_selector_rejects_invalid_workers() -> None:
    with pytest.raises(ValueError, match="positive"):
        _collection_opponent_selector(0, 0)


def test_correction_weight_upweights_disagreement():
    assert correction_weight(2, 2, disagreement=2.5) == 1.0
    assert correction_weight(2, 7, disagreement=2.5) == 2.5


def test_correction_weight_rejects_downweighting_disagreements():
    with pytest.raises(ValueError, match="at least one"):
        correction_weight(1, 2, disagreement=0.5)


@pytest.mark.parametrize(
    ("action", "family"),
    [(0, "ordinary_move"), (4, "tera_move"), (8, "switch"), (13, "forced")],
)
def test_action_family(action, family):
    assert action_family(action) == family


def test_summarize_corrections_reports_family_confusion_by_opponent():
    rows = [
        {"game": 0, "team": 2, "opponent": "simpleheuristics", "executed": 1, "target": 5},
        {"game": 0, "team": 2, "opponent": "simpleheuristics", "executed": 9, "target": 9},
        {"game": 1, "team": 3, "opponent": "maxdamage", "executed": 13, "target": 2},
    ]

    report = summarize_corrections(rows, team_count=171)

    assert report["games"] == 2
    assert report["teams_covered"] == 2
    assert report["overall"]["disagreements"] == 2
    assert report["overall"]["family_confusion"]["ordinary_move"]["tera_move"] == 1
    assert report["by_opponent"]["simpleheuristics"]["disagreement_rate"] == 0.5


def test_within_family_eligible_excludes_cross_family_and_tera():
    assert within_family_eligible(0, 3)
    assert within_family_eligible(8, 12)
    assert not within_family_eligible(0, 8)
    assert not within_family_eligible(4, 4)


def test_conditional_family_log_probs_normalizes_only_target_family():
    torch = pytest.importorskip("torch")
    logits = torch.arange(28, dtype=torch.float32).reshape(2, 14)
    log_probs = torch.log_softmax(logits, dim=-1)
    conditional, mask = conditional_family_log_probs(log_probs, torch.tensor([2, 10]))

    assert mask[0, :4].all() and not mask[0, 4:].any()
    assert mask[1, 8:13].all() and not mask[1, :8].any() and not mask[1, 13]
    assert torch.allclose(conditional[0, :4].exp().sum(), torch.tensor(1.0))
    assert torch.allclose(conditional[1, 8:13].exp().sum(), torch.tensor(1.0))


def test_conditional_family_log_probs_preserves_illegal_negative_infinity():
    torch = pytest.importorskip("torch")
    log_probs = torch.tensor([[-1.0, float("-inf"), -2.0, -3.0] + [float("-inf")] * 10])
    conditional, mask = conditional_family_log_probs(log_probs, torch.tensor([2]))

    finite_mask = mask & torch.isfinite(conditional)
    safe = torch.where(finite_mask, conditional, torch.zeros_like(conditional))
    assert torch.isfinite(safe).all()
    assert torch.isneginf(conditional[0, 1])


def test_summarize_policy_shift_tracks_toward_away_and_other_changes():
    rows = [
        {"executed": 0, "target": 1, "opponent": "simpleheuristics", "outcome": 1.0},
        {"executed": 8, "target": 8, "opponent": "maxdamage", "outcome": -1.0},
        {"executed": 2, "target": 3, "opponent": "donor", "outcome": 0.0},
    ]

    report = summarize_policy_shift(rows, [1, 9, 0])

    overall = report["overall"]
    assert overall["changed"] == 3
    assert overall["changed_toward_teacher"] == 1
    assert overall["changed_away_from_teacher"] == 1
    assert overall["changed_other"] == 1
    assert overall["teacher_agreement_before_rate"] == pytest.approx(1 / 3)
    assert overall["teacher_agreement_after_rate"] == pytest.approx(1 / 3)
    assert report["by_outcome"]["win"]["changed_toward_teacher"] == 1


def test_summarize_policy_shift_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        summarize_policy_shift([], [0])


def test_sampling_probability_summary_measures_suppressed_tera_exploration():
    torch = pytest.importorskip("torch")
    probabilities = torch.tensor(
        [
            [0.80, 0.10, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
            + [0.0] * 6,
            [0.50, 0.25, 0.125, 0.125] + [0.0] * 10,
        ],
        dtype=torch.float32,
    )
    legality = probabilities > 0
    log_probs = probabilities.log()

    report = summarize_sampling_probabilities(log_probs, legality, [1.0, 2.0], [2.0])

    assert report["temperature_1"]["tera_legal_rows"] == 1
    assert report["temperature_1"]["expected_tera_move_rate"] == pytest.approx(0.02)
    assert (
        report["temperature_2"]["expected_tera_move_rate"]
        > report["temperature_1"]["expected_tera_move_rate"]
    )
    assert (
        report["temperature_2"]["expected_non_greedy_rate"]
        > report["temperature_1"]["expected_non_greedy_rate"]
    )
    assert (
        report["tera_bonus_2"]["expected_tera_move_rate"]
        > report["temperature_1"]["expected_tera_move_rate"]
    )
