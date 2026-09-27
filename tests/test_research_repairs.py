from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fog.eval.showdown_direct_selection import MatchResult
from fog.eval.stats import paired_block_interval
from fog.scripts.automated_training import summarize_evaluation
from fog.scripts.scrape_replays import scrape
from fog.team_identity import team_group_key
from fog.training.ppo import ppo_update


def test_actual_typed_policy_has_identity_ratios_with_dropout_and_zero_lr():
    from tests.test_shared_latent import _batch, _typed_contextual_agent

    torch.manual_seed(27)
    agent = _typed_contextual_agent()
    for module in agent.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.25
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.25
    batch = _batch()
    with torch.no_grad():
        output = agent(batch)
    batch.update(actions=torch.tensor([0, 1]), advantages=torch.tensor([-1., 1.]),
                 returns=output.value.detach(), behavior_log_probs=output.log_probs.detach(),
                 old_log_probs=output.log_probs[torch.arange(2), torch.tensor([0, 1])].detach())
    agent.train()  # update must reconcile the mode without switching off autograd
    result = ppo_update(agent, torch.optim.AdamW(agent.parameters(), lr=0), batch,
                        {"ppo": {"epochs": 1, "minibatch_size": 2, "clip": 0.05}})
    assert result["clip_frac"] == 0
    assert abs(result["policy_kl"]) < 1e-6
    assert result["value_loss"] < 1e-10
    assert result["optimizer_steps"] == 1
    assert agent.training


def test_ppo_kl_stop_does_not_update_or_drop_auxiliary_request():
    from tests.test_ppo_update import TinyPolicyAgent, make_ppo_batch

    class CheckedAgent(TinyPolicyAgent):
        def forward(self, batch):
            assert not self.training
            assert batch["compute_belief_chain"]
            return super().forward(batch)

    agent = CheckedAgent()
    batch = make_ppo_batch()
    batch["old_log_probs"].fill_(-10)
    before = [parameter.clone() for parameter in agent.parameters()]
    metrics = ppo_update(agent, torch.optim.Adam(agent.parameters(), lr=.1), batch,
                         {"ppo": {"target_kl": .001}})
    assert metrics["stopped_for_kl"] == 1
    assert metrics["optimizer_steps"] == 0
    assert all(torch.equal(old, new) for old, new in zip(before, agent.parameters()))


def _row(label, scores):
    return MatchResult(label, *label.split(" vs "), requested_games=len(scores),
                       completed_games=len(scores), a_wins=scores.count(1.),
                       b_wins=scores.count(0.), draws=scores.count(.5),
                       game_scores=scores).as_dict()


def test_promotion_requires_completed_matched_evidence_and_real_improvement():
    truncated = MatchResult("parent vs candidate", "parent", "candidate", 64,
                            completed_games=64, b_wins=1, draws=63, truncated=63,
                            game_scores=[0.] + [None] * 63).as_dict()
    assert not summarize_evaluation({"status": "complete", "rows": [truncated]})[
        "recommend_viewer_update"]
    # Small point estimates remain diagnostic, even when all games are completed.
    report = {"status": "complete", "rows": [_row("parent vs candidate", [0.] * 34 + [1.] * 30)]}
    assert not summarize_evaluation(report)["recommend_viewer_update"]
    # Strong independent evidence, paired identical anchors, and a 12-look error allocation.
    report["rows"] = [_row("parent vs candidate", [0.] * 1024)]
    for opponent in ("maxdamage", "simpleheuristics"):
        report["rows"] += [_row(f"{name} vs {opponent}", [0., 1.] * 512)
                           for name in ("parent", "candidate")]
    settings = {"opponents": ["maxdamage", "simpleheuristics"], "promotion_alpha": .05 / 12}
    assert not summarize_evaluation(report, settings)["recommend_viewer_update"]
    settings["promotion_pool_role"] = "test"
    result = summarize_evaluation(report, settings)
    assert result["recommend_viewer_update"]
    report["rows"].pop()
    assert not summarize_evaluation(report, settings)["recommend_viewer_update"]


def test_draws_count_half_but_truncations_are_not_draw_scores():
    row = _row("parent vs candidate", [0., .5, 1., .5] * 32)
    summary = summarize_evaluation({"status": "complete", "rows": [row]})
    assert summary["candidate_head_to_head_win_rate"] == .5
    assert not summary["recommend_viewer_update"]
    interval = paired_block_interval([0.] * 128, bounds=(-1., 1.))
    assert interval.low < 0 < interval.high


def test_team_identity_groups_order_and_names_but_preserves_hidden_set_differences():
    first = "A (Gholdengo) @ Leftovers\nAbility: Good as Gold\n- Recover\n- Shadow Ball"
    second = "B (Dragapult) @ Choice Specs\nAbility: Infiltrator\n- Shadow Ball\n- Draco Meteor"
    paste = first + "\n\n" + second
    permuted = second.replace("B (", "Nickname (") + "\n\n" + first.replace(
        "- Recover\n- Shadow Ball", "- Shadow Ball\n- Recover")
    assert team_group_key(paste) == team_group_key(permuted)
    changed = paste.replace("Leftovers", "Air Balloon")
    assert team_group_key(paste) != team_group_key(changed)
    assert team_group_key(paste, mode="roster") == team_group_key(changed, mode="roster")


def test_scrape_resume_drains_pending_page_and_retries_transport_errors(tmp_path, monkeypatch):
    import fog.scripts.scrape_replays as scraper

    entries = [{"id": f"gen9ou-{i}", "uploadtime": 100 - i, "rating": 1800} for i in range(3)]
    monkeypatch.setattr(scraper, "search_page", lambda fmt, before: entries if before is None else [])
    monkeypatch.setattr(scraper, "fetch_replay", lambda rid: {"id": rid, "rating": 1800, "log": "x"})
    assert scrape("gen9ou", 1500, 1, tmp_path, delay=0) == 1
    monkeypatch.setattr(scraper, "fetch_replay", lambda _: None)
    assert scrape("gen9ou", 1500, 3, tmp_path, delay=0) == 1
    assert not (tmp_path / "_rejected.txt").exists()
    assert len(json.loads((tmp_path / "_cursor.json").read_text())["pending"]) == 2
    monkeypatch.setattr(scraper, "fetch_replay", lambda rid: {"id": rid, "rating": 1800, "log": "x"})
    assert scrape("gen9ou", 1500, 3, tmp_path, delay=0) == 3


def test_block_interval_resamples_whole_blocks():
    scores = np.tile([0., 0., 1., 1.], 32)
    first = paired_block_interval(scores)
    second = paired_block_interval(scores.reshape(-1, 4)[:, ::-1].ravel())
    assert first == second
    with pytest.raises(ValueError):
        paired_block_interval([0., 1.] * 5)


def test_promotion_design_cannot_expand_looks_after_observing_results():
    from fog.scripts.automated_training import _promotion_design

    job = {"training": {"blocks": 12}, "evaluation": {"promotion_family_alpha": .05}}
    state = {}
    assert _promotion_design(job, state)["looks"] == 12
    assert _promotion_design(job, state)["family_alpha"] == .05
    job["training"]["blocks"] = 24
    with pytest.raises(ValueError, match="promotion design changed"):
        _promotion_design(job, state)
