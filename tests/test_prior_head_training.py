"""Complete matched resume and frozen-policy artifact isolation for prior heads."""

import json

import numpy as np
import pytest
import torch

from fog.constants.tables import TABLES
from fog.pretrained.frozen_belief import (
    DATA_FORMAT,
    DATA_VERSION,
    FrozenBeliefData,
    digest,
    export_heads,
    initialize_residual,
    load_heads,
)
from fog.pretrained.joint import BeliefHeads
from fog.pretrained.kakuna import KAKUNA_34_SHA256
from fog.scripts.train_prior_heads import run
from fog.training.replay_candidates import vocabulary_digest


def inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    torch.manual_seed(8)
    species = [
        TABLES.species_index(name)
        for name in (
            "garchomp",
            "clefable",
            "corviknight",
            "dragapult",
            "gholdengo",
            "ironvaliant",
            "greattusk",
            "dragonite",
            "rillaboom",
            "glimmora",
            "skarmory",
            "tinglu",
        )
    ]
    teams = []
    for offset in [0, 6]:
        team = torch.zeros(6, 15, dtype=torch.long)
        for i, sp in enumerate(species[offset : offset + 6]):
            team[i, 0] = sp
            team[i, 1] = TABLES.item_index("leftovers")
            team[i, 2] = int(np.flatnonzero(TABLES.legal_abilities_mask(sp))[0])
            team[i, 3:7] = torch.from_numpy(np.flatnonzero(TABLES.learnset_mask(sp))[:4].copy())
        teams.append(team)
    records = []
    for split, team in zip(["train", "development"], teams):
        for pov in [0, 1]:
            name = f"{split}/{pov}.pt"
            path = data / name
            path.parent.mkdir(exist_ok=True)
            truth = team.expand(3, -1, -1).clone()
            reveal = torch.full_like(truth, -1)
            reveal[:, :, 0] = truth[:, :, 0]
            torch.save(
                {
                    "latent": torch.randn(3, 900),
                    "labels": {
                        "reveals": reveal,
                        "truth": truth,
                        "verified": torch.ones_like(truth, dtype=torch.bool),
                        "outcomes": torch.tensor([1.0, -1.0, 1.0]),
                        "outcome_valid": torch.ones(3, dtype=torch.bool),
                    },
                },
                path,
            )
            records.append(
                {
                    "path": name,
                    "split": split,
                    "game": split,
                    "positions": 3,
                    "pov_seed": pov,
                    "sha256": digest(path),
                    "compositions": [split],
                    "roster_groups": [split],
                }
            )
    m = {
        "format": DATA_FORMAT,
        "version": DATA_VERSION,
        "complete": True,
        "local_trial": True,
        "native_sha256": KAKUNA_34_SHA256,
        "native_dim": 900,
        "complete_chronological_history": True,
        "both_perspectives": True,
        "private_labels_separate_from_latents": True,
        "records": records,
        "counts": {
            s: {"games": 1, "perspectives": 2, "positions": 6} for s in ["train", "development"]
        },
    }
    (data / "manifest.json").write_text(json.dumps(m))
    bank = tmp_path / "bank.npz"
    # All donor rosters differ from the query roster; retrieval cannot copy it.
    tokens = torch.cat(teams).numpy()
    np.savez_compressed(
        bank,
        tokens=tokens,
        rosters=np.tile(np.arange(100, 106), (12, 1)),
        replay_ids=np.array(["source"] * 12),
        log_hashes=np.array(["x"] * 12),
        timestamps=np.ones(12),
        ratings=np.zeros(12),
        metadata=np.asarray(
            json.dumps(
                {"format": "gen9ou", "schema_version": 1, "vocabulary_sha256": vocabulary_digest()}
            )
        ),
    )
    heads = tmp_path / "warm.pt"
    model = BeliefHeads()
    torch.save(
        {"version": 1, "native_sha256": KAKUNA_34_SHA256, "heads": model.state_dict()}, heads
    )
    return {
        "data": str(data),
        "data_sha256": digest(data / "manifest.json"),
        "prior": str(bank),
        "prior_sha256": digest(bank),
        "initial_heads": str(heads),
        "initial_heads_sha256": digest(heads),
        "native_sha256": KAKUNA_34_SHA256,
        "seed": 90,
        "updates": 2,
        "effective_batch": 2,
        "microbatch": 1,
        "chain_queries": 2,
        "save_every": 1,
        "warmup_updates": 1,
        "learning_rate": 2e-4,
        "max_donors": 2,
        "smoothing": 0.001,
        "training_seconds": 300,
        "evaluation_games": 1,
        "allow_local_data": True,
    }


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def test_actual_complete_adam_and_rng_resume_across_matched_arms(tmp_path):
    torch.set_num_threads(1)
    cfg = inputs(tmp_path)
    full = run(cfg, tmp_path / "full")
    half = run(cfg, tmp_path / "split", stop_after_steps=3)
    assert not half["complete"]
    resumed = run(cfg, tmp_path / "split", resume=tmp_path / "split/latest.state.pt")
    assert resumed["complete"] and full["complete"]
    a = torch.load(tmp_path / "full/latest.state.pt", weights_only=False)
    b = torch.load(tmp_path / "split/latest.state.pt", weights_only=False)
    for key in [
        "model",
        "initial",
        "optimizer",
        "rng",
        "sampler",
        "route_rng",
        "phase",
        "completed",
        "steps",
        "histories",
        "evaluations",
    ]:
        assert equal(a[key], b[key]), key
    assert full["matched_query_routes"] and full["outcome_heads_identical"]
    assert not full["native_policy_changed"] and not full["native_optimizer_present"]
    for phase, mode in enumerate(["facts", "native"]):
        checkpoint = torch.load(tmp_path / f"full/{mode}_full.state.pt", weights_only=False)
        heads = torch.load(tmp_path / f"full/{mode}_heads_final.pt", weights_only=True)
        assert checkpoint["phase"] == phase and checkpoint["completed"] == cfg["updates"]
        assert checkpoint["steps"] == cfg["updates"] * (phase + 1)
        assert checkpoint["optimizer"]["state"] and checkpoint["rng"]
        assert equal(checkpoint["model"], heads["heads"])
        assert not any(key.startswith("native.") for key in checkpoint["model"])


def test_resume_into_new_output_keeps_the_complete_metric_journal(tmp_path):
    torch.set_num_threads(1)
    cfg = inputs(tmp_path)
    run(cfg, tmp_path / "interrupted", stop_after_steps=1)
    source = tmp_path / "interrupted/latest.state.pt"
    resumed = run(cfg, tmp_path / "new-output", resume=source)
    assert resumed["complete"]
    checkpoint = torch.load(tmp_path / "new-output/latest.state.pt", weights_only=False)
    rows = [json.loads(line) for line in (tmp_path / "new-output/metrics.jsonl").read_text().splitlines()]
    assert rows == checkpoint["histories"] and len(rows) == 4


def test_private_labels_cannot_change_retrieval_and_hashes_guard_data(tmp_path):
    cfg = inputs(tmp_path)
    with pytest.raises(ValueError, match="Local trial"):
        FrozenBeliefData(cfg["data"], cfg["data_sha256"])
    record = tmp_path / "data/train/0.pt"
    record.write_bytes(record.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed frozen"):
        FrozenBeliefData(cfg["data"], cfg["data_sha256"], allow_trial=True)


def test_prior_heads_cannot_silently_load_against_another_prior_or_native(tmp_path):
    cfg = inputs(tmp_path)
    model = initialize_residual(
        cfg["initial_heads"],
        expected_sha256=cfg["initial_heads_sha256"],
        native_sha256=cfg["native_sha256"],
    )
    assert all(not name.startswith("native.") for name, _ in model.named_parameters())
    out = tmp_path / "head.pt"
    export_heads(model, out, config=cfg, mode="native", update=1, identity="test")
    for native, prior in [("wrong", cfg["prior_sha256"]), (cfg["native_sha256"], "wrong")]:
        with pytest.raises(ValueError, match="identity"):
            load_heads(model, out, native_sha256=native, prior_sha256=prior)
    loaded = load_heads(
        model, out, native_sha256=cfg["native_sha256"], prior_sha256=cfg["prior_sha256"]
    )
    assert not loaded["native_weights_included"]


def test_fresh_schedule_keeps_sixteen_roster_groups_independent():
    from fog.scripts.prior_belief_contexts import schedule

    games = schedule(16, 20261020)
    assert len(games) == 64
    seen = set()
    for group in range(16):
        block = games[group * 4 : group * 4 + 4]
        members = {g.p1_team for g in block} | {g.p2_team for g in block}
        assert len(members) == 2 and not (members & seen)
        seen |= members
        assert sum(g.p1_is_a for g in block) == 2
        assert len({g.showdown_seed for g in block}) == 4


def test_loading_auxiliary_heads_cannot_change_the_native_policy(tmp_path):
    from torch import nn
    from fog.pretrained.joint import JointKakuna

    cfg = inputs(tmp_path)
    trained = initialize_residual(
        cfg["initial_heads"],
        expected_sha256=cfg["initial_heads_sha256"],
        native_sha256=cfg["native_sha256"],
    )
    path = tmp_path / "head.pt"
    export_heads(trained, path, config=cfg, mode="native", update=1, identity="test")
    native = nn.Module()
    native.traj_encoder = nn.Module()
    native.traj_encoder.emb_dim = 900
    native.actor = nn.Linear(900, 13)
    joined = JointKakuna(native)
    latent = torch.randn(3, 900)
    before = native.actor(latent).detach().clone()
    frozen = {k: v.clone() for k, v in native.state_dict().items()}
    load_heads(joined, path, native_sha256=cfg["native_sha256"], prior_sha256=cfg["prior_sha256"])
    assert torch.equal(native.actor(latent), before)
    assert all(torch.equal(v, native.state_dict()[k]) for k, v in frozen.items())


def test_independent_process_conditions_match_the_sequential_reference(tmp_path):
    import os
    from pathlib import Path
    import sys
    from fog.pretrained.paired_belief import merge_conditions
    from fog.scripts.prior_heads_kaggle_worker import execute_pair

    torch.set_num_threads(1)
    cfg = dict(inputs(tmp_path), execution="parallel_conditions")
    reference = tmp_path / "reference"
    run(cfg, reference)
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(cfg))
    out = tmp_path / "parallel"
    commands = [
        (mode, [sys.executable, "-m", "fog.scripts.train_prior_heads",
                "--config", str(config), "--out", str(out / mode),
                "--device", "cpu", "--condition", mode], device)
        for device, mode in enumerate(["facts", "native"])
    ]
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    execute_pair(commands, environment, tmp_path / "logs", 120)
    merged = merge_conditions(cfg, out)
    assert merged["steps"] == 4 and merged["arm_updates"] == {"facts": 2, "native": 2}
    assert merged["matched_query_routes"] and merged["outcome_heads_identical"]
    for mode in ("facts", "native"):
        before = torch.load(reference / f"{mode}_heads_final.pt", weights_only=True)
        after = torch.load(out / f"{mode}_heads_final.pt", weights_only=True)
        assert equal(before, after)
    # Cross-arm audits reject a changed fitted head, even if its own metadata pass.
    state_path = out / "native/native_full.state.pt"
    state = torch.load(state_path, weights_only=False)
    first = next(key for key in state["model"] if key.startswith("outcome."))
    state["model"][first] += 0.01
    torch.save(state, state_path)
    with pytest.raises(ValueError, match="full-state/exposure"):
        merge_conditions(cfg, out)
    head_path = out / "native/native_heads_final.pt"
    head = torch.load(head_path, weights_only=True)
    head["heads"][first] = state["model"][first]
    torch.save(head, head_path)
    with pytest.raises(ValueError, match="Outcome heads differ"):
        merge_conditions(cfg, out)


def test_condition_specific_resume_cannot_switch_arms(tmp_path):
    cfg = inputs(tmp_path)
    run(cfg, tmp_path / "native", condition="native", stop_after_steps=1)
    state = tmp_path / "native/latest.state.pt"
    with pytest.raises(ValueError, match="identity/schema/recipe"):
        run(cfg, tmp_path / "wrong", condition="facts", resume=state)
    assert run(cfg, tmp_path / "native", condition="native", resume=state)["complete"]


def test_a_failed_condition_promptly_cancels_its_sibling(tmp_path):
    import os
    import sys
    import time
    from fog.scripts.prior_heads_kaggle_worker import execute_pair

    marker = tmp_path / "should-not-finish"
    commands = [
        ("facts", [sys.executable, "-c", "raise RuntimeError('test failure')"], 0),
        ("native", [sys.executable, "-c",
                    "import time; from pathlib import Path; time.sleep(30); "
                    f"Path({str(marker)!r}).write_text('finished')"], 1),
    ]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exit code|sibling failure"):
        execute_pair(commands, dict(os.environ), tmp_path / "logs", 90)
    assert time.monotonic() - started < 15 and not marker.exists()


def test_standalone_cuda_timing_uses_the_main_numerical_settings(tmp_path, monkeypatch):
    """Reach each entrypoint's data boundary without needing local CUDA hardware."""
    from fog.scripts import prior_heads_check, train_prior_heads

    class AtDataBoundary(Exception):
        pass

    observed = []

    def data(*_args, **_kwargs):
        observed.append((torch.are_deterministic_algorithms_enabled(),
                         torch.backends.cuda.matmul.allow_tf32,
                         torch.backends.cudnn.allow_tf32, torch.get_num_threads()))
        raise AtDataBoundary

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prior_heads_check, "FrozenBeliefData", data)
    monkeypatch.setattr(train_prior_heads, "FrozenBeliefData", data)
    cfg = {key: 1 for key in ("updates", "effective_batch", "microbatch", "chain_queries",
                             "save_every", "warmup_updates", "cpu_threads")}
    cfg.update(data="not-read", data_sha256="not-read")
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.get_num_threads())
    try:
        for entry in (lambda: prior_heads_check.benchmark(cfg, tmp_path / "timing", device="cuda"),
                      lambda: train_prior_heads.run(cfg, tmp_path / "main", device="cuda")):
            # Model a fresh --benchmark-only process, rather than inheriting a proof's flags.
            torch.use_deterministic_algorithms(False)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with pytest.raises(AtDataBoundary):
                entry()
        assert observed == [(True, False, False, 1)] * 2
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cuda.matmul.allow_tf32 = before[1]
        torch.backends.cudnn.allow_tf32 = before[2]
        torch.set_num_threads(before[3])
