import json
from pathlib import Path

from fog.scripts.checkpoint_cleanup import build_plan, execute_plan


def test_cleanup_keeps_named_referenced_and_two_recent(tmp_path: Path):
    run = tmp_path / "runs" / "example" / "checkpoints"
    run.mkdir(parents=True)
    for name in ("ppo_10.pt", "ppo_20.pt", "ppo_30.pt", "ppo_40.pt", "ppo_final.pt"):
        (run / name).write_bytes(b"model")
    config = tmp_path / "configs" / "job.yaml"
    config.parent.mkdir()
    config.write_text(
        "ppo:\n  init_ckpt: runs/example/checkpoints/ppo_10.pt\n", encoding="utf-8"
    )

    plan = build_plan(tmp_path, keep_recent=2, min_age_hours=0)
    candidates = {Path(item["path"]).name for item in plan["candidates"]}
    assert candidates == {"ppo_20.pt"}
    assert "ppo_final.pt" in {Path(path).name for path in plan["protected"]}


def test_execute_refuses_nothing_outside_plan_and_removes_candidates(tmp_path: Path):
    run = tmp_path / "runs" / "example" / "checkpoints"
    run.mkdir(parents=True)
    for name in ("bc_epoch_1.pt", "bc_epoch_2.pt"):
        (run / name).write_bytes(b"model")
    plan = build_plan(tmp_path, keep_recent=1, min_age_hours=-1)
    result = execute_plan(plan, tmp_path)
    assert result["removed_count"] == 1
    assert not (run / "bc_epoch_1.pt").exists()
    assert (run / "bc_epoch_2.pt").exists()


def test_cleanup_reports_do_not_self_protect_candidates(tmp_path: Path):
    run = tmp_path / "runs" / "example" / "checkpoints"
    run.mkdir(parents=True)
    old = run / "ppo_1.pt"
    recent = run / "ppo_2.pt"
    old.write_bytes(b"old")
    recent.write_bytes(b"recent")
    report = tmp_path / "runs" / "checkpoint_cleanup" / "preview.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"candidate": str(old)}), encoding="utf-8")

    plan = build_plan(tmp_path, keep_recent=1, min_age_hours=-1)

    assert str(old) in {item["path"] for item in plan["candidates"]}
