import pytest

from fog.pretrained.protocol import (
    claim_confirmation,
    claim_strong_evaluation,
    foulplay_fingerprint,
    charged_gpu_seconds,
    record_gpu_overrun,
    reserve_gpu,
)
import json


def test_failed_runs_still_consume_phase_budget(tmp_path):
    path = tmp_path / "budget.json"
    reserve_gpu(path, run_id="a", phase="pilot", condition="ordinary", seconds=3600, family="f")
    reserve_gpu(path, run_id="a", phase="pilot", condition="ordinary", seconds=3600, family="f")
    reserve_gpu(path, run_id="b", phase="pilot", condition="joint", seconds=3600, family="f")
    with pytest.raises(ValueError, match="Aggregate GPU phase"):
        reserve_gpu(
            path, run_id="retry", phase="pilot", condition="joint", seconds=600, family="changed"
        )
    with pytest.raises(ValueError, match="duplicate"):
        reserve_gpu(
            path, run_id="duplicate", phase="pilot", condition="ordinary", seconds=3600, family="f"
        )


def test_overruns_and_recovery_keep_original_attempt_budget(tmp_path):
    path = tmp_path / "budget.json"
    reserve_gpu(
        path,
        run_id="ordinary",
        phase="pilot",
        condition="ordinary",
        seconds=3600,
        family="original",
    )
    reserve_gpu(
        path, run_id="joint", phase="pilot", condition="joint", seconds=3600, family="original"
    )
    original = json.loads(path.read_text())["reservations"]
    options = dict(
        run_id="ordinary",
        extra_seconds=23004,
        evidence={"source": "account quota delta; conservative upper bound"},
    )
    record_gpu_overrun(path, **options)
    record_gpu_overrun(path, **options)
    assert json.loads(path.read_text())["reservations"] == original
    with pytest.raises(ValueError, match="cannot change"):
        record_gpu_overrun(path, **dict(options, extra_seconds=1))
    reserve_gpu(
        path,
        run_id="recovery-a",
        phase="recovery",
        condition="ordinary",
        seconds=1800,
        family="fixed",
    )
    reserve_gpu(
        path, run_id="recovery-b", phase="recovery", condition="joint", seconds=1800, family="fixed"
    )
    assert charged_gpu_seconds(json.loads(path.read_text())) == 33804
    # Even with a fresh weekly quota, this attempt cannot use its old full main/refinement plan.
    reserve_gpu(
        path, run_id="main-a", phase="main", condition="ordinary", seconds=14400, family="main"
    )
    reserve_gpu(
        path, run_id="main-b", phase="main", condition="joint", seconds=14400, family="main"
    )
    with pytest.raises(ValueError, match="exceeds 18 GPU hours"):
        reserve_gpu(
            path,
            run_id="refinement-a",
            phase="refinement",
            condition="ordinary",
            seconds=3600,
            family="refined",
        )


def test_confirmation_cannot_reuse_cohort_for_a_new_model(tmp_path):
    path = tmp_path / "confirmations.json"
    options = dict(attempt=1, candidate_sha256="model-a", pool_sha256="pool", seed=42, games=1024)
    claim_confirmation(path, **options)
    claim_confirmation(path, **options)
    with pytest.raises(ValueError, match="already spent"):
        claim_confirmation(path, **dict(options, candidate_sha256="model-b"))
    claim_confirmation(path, **dict(options, attempt=2, candidate_sha256="model-b"))
    with pytest.raises(ValueError, match="Only two"):
        claim_confirmation(path, **dict(options, attempt=3))
    with pytest.raises(ValueError, match="pool cannot change"):
        claim_confirmation(
            path, **dict(options, attempt=2, candidate_sha256="model-b", pool_sha256="other")
        )


def test_strong_comparison_uses_same_finalist_and_schedule(tmp_path):
    path = tmp_path / "confirmations.json"
    claim_confirmation(
        path, attempt=1, candidate_sha256="new", pool_sha256="pool", seed=42, games=1024
    )
    options = dict(
        attempt=1,
        finalist_sha256="new",
        evaluated_sha256="new",
        original_sha256="old",
        pool_sha256="pool",
        seed=43,
        games=256,
    )
    claim_strong_evaluation(path, **options)
    claim_strong_evaluation(path, **dict(options, evaluated_sha256="old"))
    with pytest.raises(ValueError, match="immutable matched schedule"):
        claim_strong_evaluation(path, **dict(options, seed=44))
    with pytest.raises(ValueError, match="Only the declared finalist"):
        claim_strong_evaluation(path, **dict(options, evaluated_sha256="other"))
    with pytest.raises(ValueError, match="immutable matched schedule"):
        claim_strong_evaluation(path, **dict(options, opponent_sha256="new-priors"))


def test_strong_opponent_fingerprint_covers_untracked_priors(tmp_path):
    code = tmp_path / "fp/search/standard_battles.py"
    code.parent.mkdir(parents=True)
    code.write_text("pinned code")
    prior = tmp_path / "fp/data/smogon_stats_cache/gen9ou-0.json"
    prior.parent.mkdir(parents=True)
    prior.write_text('{"sets": 1}')
    before = foulplay_fingerprint(tmp_path)
    prior.write_text('{"sets": 2}')
    assert foulplay_fingerprint(tmp_path) != before
