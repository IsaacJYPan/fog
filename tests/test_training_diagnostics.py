import pytest
import torch

from fog.training.diagnostics import gradient_comparison, tera_bonus_at


def test_tera_schedule_endpoints_resume_and_validation():
    cfg = {"rollout_tera_logit_bonus_initial": 1.5, "rollout_tera_logit_bonus_final": 0,
           "rollout_tera_logit_bonus_decay_decisions": 60000}
    assert tera_bonus_at(cfg, 0, 80000) == 1.5
    assert tera_bonus_at(cfg, 20000, 80000) == 1.
    assert tera_bonus_at(cfg, 80000, 80000) == 0.
    assert tera_bonus_at({"rollout_tera_logit_bonus": .7}, 20000, 80000) == .7
    with pytest.raises(ValueError):
        tera_bonus_at({"rollout_tera_logit_bonus_initial": float("nan")}, 0, 80000)


def test_gradient_measurement_uses_policy_only_and_does_not_accumulate():
    parameter = torch.nn.Parameter(torch.tensor([2., 3.]))
    unused = torch.nn.Parameter(torch.tensor(1.))
    policy = 3 * parameter[0] + 4 * parameter[1]
    replay = 12 * parameter[0] - 9 * parameter[1]
    result = gradient_comparison(policy, replay, [parameter, unused], 1/30)
    assert result["policy_gradient_norm"] == 5
    assert result["replay_unweighted_gradient_norm"] == 15
    assert result["replay_weighted_gradient_ratio"] == pytest.approx(.1)
    assert result["replay_policy_gradient_cosine"] == 0
    assert parameter.grad is unused.grad is None
    (policy + replay/30).backward()
    assert parameter.grad.tolist() == pytest.approx([3.4, 3.7])


def test_small_test_pool_screen_cannot_promote_even_with_all_wins():
    from fog.eval.showdown_direct_selection import MatchResult
    from fog.scripts.automated_training import summarize_evaluation

    row = MatchResult("parent vs candidate", "parent", "candidate", 256,
                      completed_games=256, b_wins=256, game_scores=[0.] * 256).as_dict()
    result = summarize_evaluation({"status": "complete", "rows": [row]},
                                  {"promotion_pool_role": "test", "promotion_head_min_games": 128})
    assert not result["recommend_viewer_update"]


def test_opponent_draws_cannot_shift_game_schedule_and_resume_preserves_stream():
    import random
    from types import SimpleNamespace

    from fog.training.diagnostics import direct_rng_streams

    def setup():
        return SimpleNamespace(env_cfg={"independent_opponent_rng": True}, seed=7,
            rng=random.Random(7), resume_backend_rng_state=None, backend_rng=None)

    a, b = setup(), setup()
    ga, oa = direct_rng_streams(a)
    gb, ob = direct_rng_streams(b)
    for _ in range(99):
        oa.random()
    ob.random()
    assert [ga.getrandbits(256) for _ in range(10)] == [gb.getrandbits(256) for _ in range(10)]
    resumed = setup()
    resumed.resume_backend_rng_state = ga.getstate()
    gr, _ = direct_rng_streams(resumed)
    assert ga.getrandbits(256) == gr.getrandbits(256)
