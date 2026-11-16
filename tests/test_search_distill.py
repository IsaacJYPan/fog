"""Search distillation: schedule, storage, loss, and one small live collection."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from fog.encoding.action_features import ACTION_FEATURE_DIM
from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.training import search_distill as sd


def test_schedule_is_a_function_of_seed_and_game_and_shards_partition_it():
    plans = [sd.game_plan(game, 7, 600) for game in range(16)]
    assert plans == [sd.game_plan(game, 7, 600) for game in range(16)]
    assert plans != [sd.game_plan(game, 8, 600) for game in range(16)]
    assert all(plan["p1_team"] != plan["p2_team"] for plan in plans)
    assert [plan["opponent"] for plan in plans[:4]] == list(sd.OPPONENT_CYCLE)
    assert [plan["seats"] for plan in plans[:8]] == [
        ("p1", "p2"), ("p1", "p2"), ("p1",), ("p1",), ("p1", "p2"), ("p1", "p2"), ("p2",), ("p2",)
    ]


def test_a_game_is_claimed_once_and_never_after_it_is_stored(tmp_path):
    assert sd.claim_game(tmp_path, 3) is True
    assert sd.claim_game(tmp_path, 3) is False          # another process holds it
    assert sd.claim_game(tmp_path, 4) is True
    sd.game_path(tmp_path, 5).write_bytes(b"stored")
    assert sd.claim_game(tmp_path, 5) is False          # finished games are never replayed


def test_policy_gradient_is_softmax_minus_target_and_rows_without_target_add_nothing():
    logits = torch.tensor([[0.3, -0.2, 0.9, 0.1], [0.5, 0.4, -0.3, 0.2], [0.0, 0.1, 0.2, 0.3]],
                          requires_grad=True)
    legal = torch.tensor([[True, True, True, False], [True, True, False, False],
                          [True, True, True, True]])
    log_probs = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
    targets = {
        "policy": torch.tensor([[0.7, 0.0, 0.3, 0.0], [0.25, 0.75, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]]),
        "has_policy": torch.tensor([True, True, False]),
        "teacher_value": torch.tensor([0.5, float("nan"), float("nan")]),
        "outcome": torch.tensor([1.0, -1.0, 1.0]),
    }
    value_logit = torch.tensor([0.2, -0.1, 0.0], requires_grad=True)
    total, metrics = sd.distill_loss(log_probs, value_logit.tanh(), value_logit, targets,
                                     value_mix=0.5, value_coef=0.0)
    total.backward()
    probabilities = log_probs.exp().detach()
    expected = (probabilities - targets["policy"]) / 2
    expected[2] = 0.0
    torch.testing.assert_close(logits.grad, expected, atol=1e-6, rtol=0)
    assert metrics["policy_rows"] == 2 and metrics["rows"] == 3
    assert metrics["policy_kl"] >= 0

    # Value target: half outcome, half search where the search has a value; outcome elsewhere.
    value_logit.grad = None
    total, _ = sd.distill_loss(log_probs.detach(), value_logit.tanh(), value_logit, targets,
                               value_mix=0.5, value_coef=1.0)
    total.backward()
    target = (torch.tensor([0.75, -1.0, 1.0]) + 1) / 2
    expected_value_gradient = 2 * (torch.sigmoid(2 * value_logit.detach()) - target) / 3
    torch.testing.assert_close(value_logit.grad, expected_value_gradient, atol=1e-6, rtol=0)


def _fragment(generator: np.random.Generator, turns: int) -> dict:
    return {
        "entities": generator.random((12, ENTITY_RAW_DIM), dtype=np.float32),
        "entity_ids": generator.integers(0, 5, (12, N_ID_FIELDS)),
        "field": generator.random(DIMS.field_dim, dtype=np.float32),
        "history": generator.random((turns, DIMS.turn_dim), dtype=np.float32),
        "history_ids": generator.integers(0, 5, (turns, DIMS.hist_id_fields)),
        "history_state": generator.random((turns, DIMS.hist_state), dtype=np.float32),
        "history_len": np.int64(turns),
        "active_idx": np.asarray([0, 6]),
        "legality_mask": generator.random(14) > 0.3,
        "action_features": generator.random((14, 3), dtype=np.float32),
        "teacher_policy": np.full(14, 1 / 14, dtype=np.float32),
        "teacher_value": np.float32(0.25),
        "teacher_target": np.bool_(True),
        "teacher_action": np.int64(0),
        "executed_by_teacher": np.bool_(True),
        "network_log_probs": np.zeros(14, dtype=np.float32),
        "check_repeat_action": np.int64(-1),
        "check_default_action": np.int64(-1),
        "turn": np.int64(turns + 1),
    }


def test_stored_rows_come_back_as_the_live_path_would_stack_them(tmp_path):
    from fog.training.rollout import _stack_frags

    generator = np.random.default_rng(3)
    first = [_fragment(generator, turns) for turns in (0, 1, 4)]
    second = [_fragment(generator, turns) for turns in (2, 7)]
    arrays = sd.merge_seats([sd.pack_rows(first, [1, 2, 3], 1.0, 0),
                             sd.pack_rows(second, [4, 5], -1.0, 1)])
    sd.save_game(sd.game_path(tmp_path, 0), arrays, {"game": 0})
    assert not list(tmp_path.glob("*.tmp"))
    data = sd.DistillData([tmp_path], holdout_fraction=0.0, seed=0)
    for rows in ([0, 1, 2, 3, 4], [3], [0, 4], [0]):
        inputs, targets = data.batch(np.asarray(rows))
        fragments = [(first + second)[row] for row in rows]
        expected = _stack_frags([{key: frag[key] for key in sd.MODEL_KEYS} for frag in fragments])
        assert set(inputs) == set(sd.MODEL_KEYS)
        for key in sd.MODEL_KEYS:
            assert inputs[key].shape == expected[key].shape, key
            torch.testing.assert_close(inputs[key].double(), expected[key].double(),
                                       atol=0, rtol=0, msg=key)
        assert targets["outcome"].tolist() == [1.0 if row < 3 else -1.0 for row in rows]


def test_shared_history_gives_back_every_row_and_stores_a_growing_history_once():
    generator = np.random.default_rng(11)
    turns = generator.random((9, DIMS.turn_dim), dtype=np.float32)
    ids = generator.integers(0, 5, (9, DIMS.hist_id_fields))
    state = generator.random((9, DIMS.hist_state), dtype=np.float32)
    frags = []
    # A seat whose history grows, then slides (a window), then is rewritten.
    for start, end in ((0, 0), (0, 1), (0, 3), (0, 3), (1, 5), (2, 9)):
        frag = _fragment(generator, 0)
        frag.update(history=turns[start:end], history_ids=ids[start:end],
                    history_state=state[start:end], history_len=np.int64(end - start))
        frags.append(frag)
    frags.append(_fragment(generator, 4))               # unrelated history: stored whole
    packed = sd.merge_seats([sd.pack_rows(frags, list(range(7)), 1.0, 0),
                             sd.pack_rows(frags[:3], [0, 1, 2], -1.0, 1)])
    shared = sd.share_history(packed)
    width = packed["history"].shape[1]
    for row in range(10):
        length = int(packed["history_rows"][row])
        start, end = int(shared["history_start"][row]), int(shared["history_end"][row])
        assert end - start == length
        for key in ("history", "history_ids", "history_state"):
            np.testing.assert_array_equal(shared[key][start:end],
                                          packed[key][row][width - length:])
    # Seat 0 stores turns 0 to 8 once and the unrelated four; seat 1 stores turns 0 to 2.
    assert len(shared["history"]) == 9 + 4 + 3


def _collect_args(out: Path, **changes) -> Namespace:
    values = dict(config="configs/search_distill_cpu.yaml", checkpoint=None,
                  pool=Path("data/teams/replay_v1/train"), out=out, games=3, beta=1.0,
                  temperature=1.0, seed=5, teacher_k=2, teacher_budget_ms=20, max_turns=300,
                  agreement_every=0, check_budget_ms=40, workers=1, shards=1, shard=0,
                  max_seconds=0.0)
    values.update(changes)
    return Namespace(**values)


def test_live_collection_labels_every_decision_and_resumes(tmp_path):
    if not Path("server/pokemon-showdown/dist/sim/battle.js").exists():
        pytest.skip("pinned Showdown build not present")
    pytest.importorskip("poke_engine")
    grad_enabled = torch.is_grad_enabled()
    threads = torch.get_num_threads()
    try:
        out = tmp_path / "teacher"
        sd.collect_shard(_collect_args(out))
        files = sorted(out.glob("game_*.npz"))
        assert [path.name for path in files] == [f"game_{game:07d}.npz" for game in range(3)]
        stamps = [path.stat().st_mtime_ns for path in files]
        assert sd.collect_shard(_collect_args(out)) == 0          # nothing left to play
        assert stamps == [path.stat().st_mtime_ns for path in files]

        data = sd.DistillData([out], holdout_fraction=0.0, seed=0)
        rows = np.arange(len(data.index))
        assert len(rows) >= 30
        inputs, targets = data.batch(rows)
        stacked = {key: np.concatenate([game[key] for game in data.games])
                   for key in ("teacher_action", "executed", "executed_by_teacher",
                               "network_log_probs", "seat")}
        has_policy = targets["has_policy"]
        assert bool(has_policy.any())
        totals = targets["policy"][has_policy].sum(-1)
        torch.testing.assert_close(totals, torch.ones_like(totals), atol=1e-5, rtol=0)
        assert float((targets["policy"] * ~inputs["legality_mask"]).sum()) == 0.0
        best = targets["policy"].argmax(-1)[has_policy].numpy()
        assert (best == stacked["teacher_action"][has_policy.numpy()]).all()
        # beta = 1: the teacher's action is always the one played.
        assert stacked["executed_by_teacher"].all()
        assert (stacked["executed"] == stacked["teacher_action"]).all()
        # Games 0 and 1 are mirrors with both sides recorded; game 2 records one side.
        assert set(data.games[0]["seat"].tolist()) == {0, 1}
        assert set(data.games[2]["seat"].tolist()) == {0}

        # A stored row gives the network the same input it had in play.
        from fog.config import load_config

        agent = sd.build_agent(load_config("configs/search_distill_cpu.yaml"), None, 5)
        with torch.no_grad():
            replayed = agent(inputs, need_belief=False).log_probs
        live = torch.as_tensor(stacked["network_log_probs"])
        finite = torch.isfinite(live)
        assert bool((torch.isfinite(replayed) == finite).all())
        torch.testing.assert_close(replayed[finite], live[finite], atol=1e-5, rtol=0)

        # beta = 0: the network plays every move and is still labelled.
        own = tmp_path / "network"
        sd.collect_shard(_collect_args(own, games=1, beta=0.0))
        arrays, meta = sd.load_game(sd.game_path(own, 0))
        assert not arrays["executed_by_teacher"].any()
        assert arrays["teacher_target"].any() and meta["beta"] == 0.0
        summary = sd.summarize(own)
        assert summary["games"] == 1 and summary["teacher_executed_rows"] == 0
        assert 0 < summary["teacher_value_rows"] <= summary["rows"]
        assert 0.0 <= summary["teacher_value_outcome_mse"] <= 4.0
        assert summary["teacher_visits_per_search"] > 0

        # A time limit that has already passed starts no game and leaves no claim behind.
        late = tmp_path / "late"
        assert sd.collect_shard(_collect_args(late, max_seconds=1e-9)) == 0
        assert not list(late.iterdir())
    finally:
        torch.set_grad_enabled(grad_enabled)
        torch.set_num_threads(threads)


def test_target_power_one_is_the_identity_and_two_sharpens_without_moving_the_peak():
    policy = torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert sd.sharpen(policy, 1.0) is policy
    sharp = sd.sharpen(policy, 2.0)
    torch.testing.assert_close(sharp[0], torch.tensor([0.25, 0.09, 0.04, 0.0]) / 0.38)
    assert sharp[0].argmax() == policy[0].argmax() and float(sharp[1].sum()) == 0.0
    with pytest.raises(ValueError):
        sd.sharpen(policy, 0.0)
    log_probs = torch.log_softmax(torch.zeros(2, 4), dim=-1)
    targets = {"policy": policy, "has_policy": torch.tensor([True, False]),
               "teacher_value": torch.tensor([0.0, 0.0]), "outcome": torch.tensor([1.0, -1.0])}
    value = torch.zeros(2)
    plain = sd.distill_loss(log_probs, value, None, targets)[1]
    same = sd.distill_loss(log_probs, value, None, targets, target_power=1.0)[1]
    assert plain == same and plain["value_brier"] == pytest.approx(0.25)


def test_training_stops_before_an_epoch_it_has_no_time_for_and_a_rerun_finishes(tmp_path):
    generator = np.random.default_rng(5)
    for game in range(3):
        frags = [_fragment(generator, turns) for turns in (0, 1, 2)]
        for frag in frags:
            legal = frag["legality_mask"].astype(np.float32)
            frag["teacher_policy"] = legal / legal.sum()
            frag["action_features"] = np.zeros((14, ACTION_FEATURE_DIM), dtype=np.float32)
        sd.save_game(sd.game_path(tmp_path / "rows", game),
                     sd.merge_seats([sd.pack_rows(frags, [0, 1, 2], 1.0, 0)]), {"game": game})
    threads = torch.get_num_threads()
    args = Namespace(config="configs/search_distill_cpu.yaml", data=[tmp_path / "rows"],
                     out=tmp_path / "net", init=None, epochs=2, batch_size=4, lr=3e-4,
                     lr_final=3e-5, holdout_fraction=0.34, value_mix=0.5, target_power=1.0,
                     seed=3, threads=1, device="cpu", limit_games=None, max_seconds=1e-9)
    try:
        stopped = sd.train(args)
        assert [row["epoch"] for row in stopped["history"]] == [0]
        assert not (tmp_path / "net" / "policy_final.pt").exists()
        args.max_seconds = 0.0
        finished = sd.train(args)
        assert [row["epoch"] for row in finished["history"]] == [0, 1, 2]
        assert (tmp_path / "net" / "policy_final.pt").exists()
        assert np.isfinite(finished["history"][-1]["holdout_policy_ce"])
    finally:
        torch.set_num_threads(threads)
