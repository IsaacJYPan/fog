"""Separate exploratory training retention from final playing-strength confirmation."""

from __future__ import annotations

from dataclasses import asdict

from fog.eval.stats import paired_block_interval

SCRIPTED_OPPONENTS = ("simpleheuristics", "maxdamage")


def continuation_review(scores, *, failures=0, unresolved=0, seed=0, alpha=0.05):
    """Retain uncertain candidates; recommend rollback only for observed regression.

    Scores must come from complete crossed games against the declared reference.
    This exploratory review is neither a non-inferiority guarantee nor permission
    to claim improved playing strength. Repeated development looks are not an
    anytime-valid test.
    """
    if failures < 0 or unresolved < 0:
        raise ValueError("Failure and unresolved counts cannot be negative")
    incomplete = (
        failures
        or unresolved
        or len(scores) < 256
        or len(scores) % 4
        or any(score is None for score in scores)
    )
    interval = None
    if not incomplete:
        interval = paired_block_interval(scores, alpha=alpha, seed=seed)
    regression = interval is not None and interval.high < 0.5
    improvement = interval is not None and interval.low > 0.5
    decision = "diagnose" if incomplete else "rollback" if regression else "retain"
    return {
        "decision": decision,
        "vs_reference": asdict(interval) if interval else None,
        "significant_regression_on_development": regression,
        "improvement_supported_on_development": improvement,
        "failures": failures,
        "unresolved": unresolved,
        "games": len(scores),
        "alpha": alpha,
        "strength_claim_confirmed": False,
        "automatic_weight_change": False,
        "interpretation": (
            "Diagnose incomplete or failed evaluation before changing checkpoints."
            if incomplete
            else "The exploratory upper confidence bound is below 50%; recommend rollback."
            if regression
            else "Retain for development; absence of significant regression does not prove non-inferiority or final improvement."
        ),
    }


def confirmation_gate(scores, *, attempt, failures=0, unresolved=0, strong_differences=None):
    if attempt not in (1, 2):
        raise ValueError("Only two predeclared confirmation attempts are permitted")
    if len(scores) < 1024 or len(scores) % 4:
        raise ValueError("Confirmation requires at least 1024 complete seat/team-balanced games")
    interval = paired_block_interval(scores, alpha=0.025, block_size=4, seed=20261006 + attempt)
    strong = None
    if strong_differences is not None:
        if len(strong_differences) < 256 or len(strong_differences) % 4:
            raise ValueError(
                "Strong-opponent regression check requires at least 256 matched games per arm"
            )
        strong = paired_block_interval(
            strong_differences,
            alpha=0.025,
            block_size=4,
            bounds=(-1.0, 1.0),
            seed=20261008 + attempt,
        )
    passed = (
        not failures
        and not unresolved
        and interval.low > 0.5
        and strong is not None
        and strong.low > -0.05
    )
    return {
        "passed": passed,
        "attempt": attempt,
        "planned_attempts": 2,
        "vs_kakuna": asdict(interval),
        "strong_opponent_difference": asdict(strong) if strong else None,
        "material_regression_margin": 0.05,
        "failures": failures,
        "unresolved": unresolved,
        "alpha_per_attempt": 0.025,
        "interval": "two-sided percentile bootstrap over four-game blocks",
        "rated_confirmation_required": passed,
        "automatic_promotion": False,
    }


def matched_update_budget(pilot_seconds, *, hours_per_arm):
    if len(pilot_seconds) != 2 or min(pilot_seconds) <= 0:
        raise ValueError("Both matched pilots must complete and report measured update throughput")
    # Allow checkpoint/export overhead; allocate by the slower condition.
    updates = int(hours_per_arm * 3600 * 0.85 / max(pilot_seconds))
    if updates < 1:
        raise ValueError("Pilot throughput leaves no complete matched update")
    return updates
