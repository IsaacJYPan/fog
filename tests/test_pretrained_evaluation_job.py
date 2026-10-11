from copy import deepcopy
import json

import pytest

from fog.scripts import pretrained_evaluation_job as job


def recipe():
    return {
        "version": 1,
        "stage": "development",
        "games": 256,
        "smoke_games": 8,
        "seed": 12,
        "matched": {"models": ["ordinary-sha", "joint-sha"]},
        "conditions": {
            arm: {
                "native": arm + ".pt",
                "reference": arm + "-prob.pt",
                "native_sha256": arm + "-sha",
            }
            for arm in ("ordinary", "joint")
        },
    }


@pytest.mark.parametrize(
    "change",
    [
        {"stage": "confirmation"},
        {"games": 252},
        {"smoke_games": 4},
        {"games": 258},
        {"matched": {"models": ["wrong", "joint-sha"]}},
        {"belief": {"games": 32, "draws": 32, "fresh": "heldout"}},
        {"pilot_only": "yes"},
    ],
)
def test_recipe_rejects_cohort_budget_and_candidate_mismatches(change):
    value = recipe()
    job.validate_recipe(value)
    value.update(change)
    with pytest.raises(ValueError):
        job.validate_recipe(value)


def test_failed_remote_contract_prevents_any_game_or_success_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        job,
        "sha256",
        lambda path: job.KAKUNA_34_SHA256 if path.name == "kakuna34.pt" else path.stem + "-sha",
    )
    calls = []

    def execute(command, *_args, **_kwargs):
        calls.append(command)
        assert command[2] == "fog.scripts.pretrained_contract"
        destination = tmp_path / command[command.index("--out") + 1]
        destination.mkdir(parents=True)
        (destination / "contract.json").write_text(json.dumps({"passed": False}))

    monkeypatch.setattr(job, "execute", execute)
    out = tmp_path / "results"
    with pytest.raises(ValueError, match="probability contract failed"):
        job.run_job(recipe(), "native-python", out, 60)
    assert len(calls) == 1
    assert json.loads((out / "STATE.json").read_text())["status"] == "failed"
    assert not (out / "RESULTS.json").exists()
    assert not (out / "full").exists()


def test_smoke_failure_prevents_full_comparison(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        job,
        "sha256",
        lambda path: job.KAKUNA_34_SHA256 if path.name == "kakuna34.pt" else path.stem + "-sha",
    )
    calls = []

    def execute(command, *_args, **_kwargs):
        calls.append(command)
        destination = tmp_path / command[command.index("--out") + 1]
        destination.mkdir(parents=True)
        if command[2] == "fog.scripts.pretrained_contract":
            checkpoint = tmp_path / command[command.index("--checkpoint") + 1]
            (destination / "contract.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "weights_unchanged": True,
                        "checkpoint_sha256": job.sha256(checkpoint),
                    }
                )
            )
        else:
            assert command[2] == "fog.scripts.pretrained_evaluate"
            assert command[command.index("--games") + 1] == "8"
            (destination / "RESULTS.json").write_text(
                json.dumps({"failures": 1, "unresolved": 1, "rows": [None] * 8})
            )

    monkeypatch.setattr(job, "execute", execute)
    with pytest.raises(ValueError, match="Evaluation failures must be diagnosed"):
        job.run_job(deepcopy(recipe()), "native-python", tmp_path / "results", 60)
    assert len(calls) == 5
    assert not (tmp_path / "results/RESULTS.json").exists()
    assert not (tmp_path / "results/full").exists()
