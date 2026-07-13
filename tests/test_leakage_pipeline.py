"""End-to-end tests for the §7 leakage pipeline: mirror recording -> RolloutBuffer -> ppo_update
self-belief training, FrozenProbe freezing, moving-mask ΔI (incl. the final-step measurement), and
the team-preview bandit."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.models.agent import FogAgent
from fog.models.probe import FrozenProbe, TrainableProbe
from fog.training.ppo import ppo_update
from fog.training.rollout import RolloutBuffer

# FogAgent() placeholder cardinalities (models/belief.py defaults)
SIZES = {"species": 320, "item": 64, "ability": 160, "move": 720, "spread": 17, "tera": 19}


def _my_view(rng) -> dict:
    tokens = np.full((6, 9), -1, dtype=np.int64)
    tokens[:, 0] = rng.integers(0, SIZES["species"], 6)   # species public
    tokens[0, 3] = rng.integers(0, SIZES["move"])         # one revealed move
    labels = {
        "species": tokens[:, 0].copy(),
        "item": rng.integers(0, SIZES["item"], 6),
        "ability": rng.integers(0, SIZES["ability"], 6),
        "move": rng.integers(0, SIZES["move"], (6, 4)),
        "spread": rng.integers(0, SIZES["spread"], 6),
        "tera": rng.integers(0, SIZES["tera"], 6),
    }
    labels = {k: v.astype(np.int64) for k, v in labels.items()}
    masks = {
        "species": np.ones((6, SIZES["species"]), dtype=bool),
        "ability": np.ones((6, SIZES["ability"]), dtype=bool),
        "move": np.ones((6, SIZES["move"]), dtype=bool),
        "spread": np.ones((6, SIZES["spread"]), dtype=bool),
    }
    return {"tokens": tokens, "masks": masks, "labels": labels}


def _frag(rng, t: int = 3) -> dict:
    opp_tokens = np.full((6, 9), -1, dtype=np.int64)
    opp_tokens[:, 0] = rng.integers(0, SIZES["species"], 6)
    return {
        "entities": rng.normal(size=(12, ENTITY_RAW_DIM)).astype(np.float32),
        "entity_ids": np.zeros((12, N_ID_FIELDS), dtype=np.int64),
        "field": rng.normal(size=DIMS.field_dim).astype(np.float32),
        "history": rng.normal(size=(t, DIMS.turn_dim)).astype(np.float32),
        "active_idx": np.array([0, 6], dtype=np.int64),
        "legality_mask": np.array([True] * 4 + [False] * 10),
        "opp_revealed": {"tokens": opp_tokens, "masks": {}},
        "turn": np.int64(t),
        "opp_view": {
            "entities": rng.normal(size=(12, ENTITY_RAW_DIM)).astype(np.float32),
            "entity_ids": np.zeros((12, N_ID_FIELDS), dtype=np.int64),
            "field": rng.normal(size=DIMS.field_dim).astype(np.float32),
            "active_idx": np.array([0, 6], dtype=np.int64),
            "my_view": _my_view(rng),
        },
    }


def _filled_buffer(rng, games=3, decisions=4) -> RolloutBuffer:
    buf = RolloutBuffer()
    for _ in range(games):
        g = buf.start_game()
        for _ in range(decisions):
            buf.add_decision(g, _frag(rng), action=int(rng.integers(0, 4)),
                             log_prob=float(np.log(0.25)), value=0.0)
        truth = np.zeros((6, 9), dtype=np.int64)
        buf.end_game(g, outcome=float(rng.choice([-1.0, 1.0])), belief_targets=truth)
    return buf


def test_to_batch_emits_self_belief_targets_and_aliases_history():
    rng = np.random.default_rng(0)
    batch = _filled_buffer(rng).to_batch()
    assert "opp_view" in batch and "my_view" in batch["opp_view"]
    assert "self_belief_targets" in batch and "self_belief_valid" in batch
    assert batch["self_belief_targets"]["move"].shape == (12, 6, 4)
    # BELIEF_PLAN §4.3: only attrs the opponent still can't see are leakage targets. _my_view makes
    # species public (preview), so species is a delta row -> invalid; item/tera stay hidden -> valid.
    assert not batch["self_belief_valid"]["species"].any()
    assert batch["self_belief_valid"]["item"].all()
    assert batch["self_belief_valid"]["tera"].all()
    # history is aliased (same tensor object), not duplicated
    assert batch["opp_view"]["history"] is batch["history"]


def test_ppo_update_trains_self_belief_and_it_decreases():
    """The M3-critical property: the self-belief predictor MUST receive gradient in self-play PPO."""
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    agent = FogAgent()
    batch = _filled_buffer(rng, games=4, decisions=4).to_batch()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)

    before = [p.detach().clone() for p in agent.self_belief.parameters()]
    cfg = {"ppo": {"minibatch_size": 16, "epochs": 1}, "loss": {"self_belief_coef": 1.0}}
    m0 = ppo_update(agent, opt, dict(batch), cfg)
    assert "self_belief_loss" in m0 and np.isfinite(m0["self_belief_loss"])
    after = list(agent.self_belief.parameters())
    assert any((a - b).abs().sum() > 0 for a, b in zip(after, before)), \
        "self-belief predictor received no gradient — the §7 measurement would be untrained noise"

    losses = [m0["self_belief_loss"]]
    for _ in range(4):
        losses.append(ppo_update(agent, opt, dict(batch), cfg)["self_belief_loss"])
    assert losses[-1] < losses[0]


def test_frozen_probe_is_fully_frozen_and_measures():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    agent = FogAgent()
    probe = FrozenProbe.from_agent(agent)
    frozen_before = [p.detach().clone() for p in probe.parameters()]

    batch = _filled_buffer(rng, games=2, decisions=3).to_batch()
    opp_view = batch["opp_view"]
    mirror = {k: opp_view[k] for k in ("entities", "entity_ids", "field", "history", "active_idx")}
    ce = probe.leakage_ce(mirror, opp_view["my_view"])
    assert ce.shape == (6,) and torch.isfinite(ce).all() and (ce >= 0).all()
    per_fam = probe.leakage_ce(mirror, opp_view["my_view"], per_family=True)
    assert set(per_fam) == {"species", "item", "ability", "move", "spread", "tera"}
    # weighted ΔI knob: zero weights -> zero CE; family sums add up
    zero = probe.leakage_ce(mirror, opp_view["my_view"],
                            family_weights={k: 0.0 for k in per_fam})
    assert torch.allclose(zero, torch.zeros_like(zero))
    total = probe.leakage_ce(mirror, opp_view["my_view"])
    assert torch.allclose(total, sum(per_fam.values()))

    # training the agent must NOT move the probe
    opt = torch.optim.Adam(agent.parameters(), lr=1e-2)
    ppo_update(agent, opt, dict(batch), {"ppo": {"minibatch_size": 8, "epochs": 1}})
    for p0, p1 in zip(frozen_before, probe.parameters()):
        assert torch.equal(p0, p1)


def test_trainable_probe_learns_to_read():
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    agent = FogAgent()
    probe = TrainableProbe.from_agent(agent)
    batch = _filled_buffer(rng, games=2, decisions=4).to_batch()
    opp_view = batch["opp_view"]
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    l0 = float(probe.loss(opp_view).detach())
    for _ in range(15):
        opt.zero_grad()
        loss = probe.loss(opp_view)
        loss.backward()
        opt.step()
    assert float(loss.detach()) < l0


def test_moving_mask_delta_with_final_measurement():
    """ΔI comes from differencing recorded CEs; final_leak_ce closes the last step so end-of-game
    reveals are penalized (previously silently zero)."""
    rng = np.random.default_rng(4)
    buf = RolloutBuffer(gamma=1.0, lam=1.0, beta=0.1)
    g = buf.start_game()
    for ce in (5.0, 3.0, 4.0):  # leaked 2 nats, then became LESS readable (no reward for that)
        buf.add_decision(g, _frag(rng), action=0, log_prob=-1.0, value=0.0, leak_ce=ce)
    buf.end_game(g, outcome=1.0, final_leak_ce=1.0)  # last step leaked 3 nats post-battle
    batch = buf.to_batch()
    # rewards: [-0.1*2, -0.1*0, -0.1*3 + 1] ; gamma=lam=1 -> returns are suffix sums
    assert torch.allclose(batch["returns"], torch.tensor([0.5, 0.7, 0.7]), atol=1e-6)


def test_hard_reveal_contributes_surprisal():
    """A hard reveal drops out of the next CE sum, so its full surprisal lands in ΔI at time t —
    the moving-mask semantics (training/leakage.py docstring)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    agent = FogAgent()
    probe = FrozenProbe.from_agent(agent)
    batch = _filled_buffer(rng, games=1, decisions=2).to_batch()
    opp_view = batch["opp_view"]
    mirror = {k: opp_view[k][:1] for k in ("entities", "entity_ids", "field", "history", "active_idx")}
    mv_before = {
        "tokens": opp_view["my_view"]["tokens"][:1].clone(),
        "masks": {k: v[:1] for k, v in opp_view["my_view"]["masks"].items()},
        "labels": {k: v[:1] for k, v in opp_view["my_view"]["labels"].items()},
    }
    from fog.models.self_belief import still_hidden_from_tokens
    from fog.training.leakage import delta_information

    # (a) Pure mask semantics: same board/tokens, one attr leaves the still-hidden set -> ΔI is
    # EXACTLY that attribute's surprisal under Φ̂ at time t (the "hard reveals count" property).
    sh_before = still_hidden_from_tokens(mv_before["tokens"])
    sh_after = {k: v.clone() for k, v in sh_before.items()}
    sh_after["item"][0, 2] = False                      # slot 2's item revealed during the turn
    only_item = {k: torch.zeros_like(v) for k, v in sh_before.items()}
    only_item["item"][0, 2] = True
    ce_b = probe.leakage_ce(mirror, mv_before, still_hidden_mask=sh_before)
    ce_a = probe.leakage_ce(mirror, mv_before, still_hidden_mask=sh_after)
    surprisal = probe.leakage_ce(mirror, mv_before, still_hidden_mask=only_item)
    assert torch.allclose(ce_b - ce_a, surprisal, atol=1e-5)
    assert surprisal.item() > 0

    # (b) Full moving-mask estimator (token ALSO teacher-forced after the reveal): finite, and equal
    # to the total-CE difference by construction. Conditioning on the reveal may move other attrs'
    # CE either way, so no exact per-family decomposition is asserted here.
    mv_after = {
        "tokens": mv_before["tokens"].clone(),
        "masks": mv_before["masks"],
        "labels": mv_before["labels"],
    }
    mv_after["tokens"][0, 2, 1] = mv_after["labels"]["item"][0, 2]
    obs_b = {**mirror, "my_view": mv_before}
    obs_a = {**mirror, "my_view": mv_after}
    di = delta_information(probe, obs_b, obs_a)
    assert torch.isfinite(di).all()
    expected = probe.leakage_ce(mirror, mv_before) - probe.leakage_ce(mirror, mv_after)
    assert torch.allclose(di, expected, atol=1e-6)


def test_preview_bandit_records_and_trains():
    torch.manual_seed(0)
    rng = np.random.default_rng(6)
    agent = FogAgent()
    buf = RolloutBuffer()
    for outcome in (1.0, -1.0):
        g = buf.start_game()
        frag = _frag(rng)
        buf.add_preview(
            g, {"entities": frag["entities"], "entity_ids": frag["entity_ids"],
                "field": frag["field"]},
            slot=int(rng.integers(0, 6)), log_prob=float(np.log(1 / 6)),
        )
        buf.add_decision(g, frag, action=0, log_prob=-1.0, value=0.0)
        buf.end_game(g, outcome=outcome)
    batch = buf.to_batch()
    assert "preview" in batch and batch["preview"]["slot"].shape == (2,)
    before = [p.detach().clone() for p in agent.preview.parameters()]
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    metrics = ppo_update(agent, opt, batch, {"ppo": {"minibatch_size": 8, "epochs": 1}})
    assert "preview_loss" in metrics
    assert any((a - b).abs().sum() > 0 for a, b in zip(agent.preview.parameters(), before))


def test_fit_and_measure_probes_produces_watchdog_inputs():
    torch.manual_seed(0)
    rng = np.random.default_rng(7)
    agent = FogAgent()
    probe = FrozenProbe.from_agent(agent)
    fresh = TrainableProbe.from_agent(agent)
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    batch = _filled_buffer(rng, games=2, decisions=3).to_batch()
    from fog.training.ppo import _fit_and_measure_probes

    m = _fit_and_measure_probes(fresh, fresh_opt, probe, batch, steps=2)
    for key in ("fresh_probe_loss", "fresh_ce", "frozen_ce"):
        assert key in m and np.isfinite(m[key])


def test_belief_accuracy_vs_turn_metric():
    torch.manual_seed(0)
    rng = np.random.default_rng(8)
    agent = FogAgent()
    batch = _filled_buffer(rng, games=3, decisions=4).to_batch()
    from fog.training.ppo import _belief_accuracy_vs_turn

    m = _belief_accuracy_vs_turn(agent, batch)
    assert any(k.startswith("belief_top1_t") for k in m)


def test_watchdog_from_control_run(tmp_path):
    csv = tmp_path / "metrics.csv"
    csv.write_text(
        "step,win_rate,entropy,avg_game_length,move_entropy\n"
        + "".join(f"{i},0.6,1.2,40.0,1.1\n" for i in range(20)),
        encoding="utf-8",
    )
    from fog.training.leakage import DegenerateSolutionWatchdog

    wd = DegenerateSolutionWatchdog.from_control_run(tmp_path)
    assert wd.baseline_win_rate == pytest.approx(0.6)
    assert wd.baseline_move_entropy == pytest.approx(1.1)   # prefers move_entropy column
    assert wd.baseline_game_length == pytest.approx(40.0)
    assert wd.check_stalling(80.0) and not wd.check_stalling(41.0)


def test_mirror_frame_consistency():
    """The mirror must be encoded in the standard frame: viewer's team = slots 0-5 with is_mine=1,
    viewer's side = first per-side field block."""
    from fog.constants.tables import TABLES
    from fog.encoding.entities import L
    from fog.encoding.mirror import build_opponent_view_np, mirror_field
    from fog.encoding.observation import Observation

    ent = np.zeros((12, ENTITY_RAW_DIM), dtype=np.float32)
    ent[:6, L.is_mine] = 1.0  # my mons flagged mine in the primary frame
    eid = np.zeros((12, N_ID_FIELDS), dtype=np.int64)
    obs = Observation(
        entities=ent, entity_ids=eid, field=np.zeros(DIMS.field_dim, np.float32),
        history=np.zeros((2, DIMS.turn_dim), np.float32), history_len=2,
        legality_mask=np.array([True] + [False] * 13), active_idx=np.array([1, 8]),
        opp_revealed={"tokens": np.full((6, 9), -1, np.int64), "masks": {}},
    )
    my_view = np.full((6, 9), -1, np.int64)
    my_view[:, 0] = 1
    truth = np.zeros((6, 9), np.int64)
    frag, mv = build_opponent_view_np(obs, my_view, truth, TABLES)
    assert frag["entities"].shape == (12, ENTITY_RAW_DIM)
    assert (frag["entities"][:6, L.is_mine] == 1.0).all()   # viewer's team reads mine
    assert (frag["entities"][6:, L.is_mine] == 0.0).all()
    assert frag["active_idx"].tolist() == [2, 7]            # sides swapped
    assert "history" not in frag                            # aliased at batch time
    assert set(mv["masks"]) == {"species", "ability", "move", "spread"}

    # field side-swap: mark my hazards block, expect it in the opponent block after mirroring
    field = np.zeros(DIMS.field_dim, dtype=np.float32)
    o = DIMS.field_weather + DIMS.field_terrain
    field[o] = 1.0             # my stealth rock
    out = mirror_field(field)
    assert out[o] == 0.0 and out[o + 6] == 1.0


def test_mirror_labels_follow_my_view_clamp():
    """Self-belief labels must match the mirror input packing (fix 2026-07-12): a move I revealed
    is clamped into my_view's earliest subslot, so the subslot-0 LABEL must be that move even
    when its paste position on my sheet is later."""
    from fog.constants.tables import TABLES
    from fog.encoding.mirror import build_opponent_view_np
    from fog.encoding.observation import Observation

    obs = Observation(
        entities=np.zeros((12, ENTITY_RAW_DIM), np.float32),
        entity_ids=np.zeros((12, N_ID_FIELDS), np.int64),
        field=np.zeros(DIMS.field_dim, np.float32),
        history=np.zeros((2, DIMS.turn_dim), np.float32), history_len=2,
        legality_mask=np.array([True] + [False] * 13), active_idx=np.array([1, 8]),
        opp_revealed={"tokens": np.full((6, 9), -1, np.int64), "masks": {}},
    )
    my_view = np.full((6, 9), -1, np.int64)
    my_view[:, 0] = 1
    my_view[0, 3] = 12  # opponent saw this move of mine; paste position on my sheet is 2
    truth = np.zeros((6, 9), np.int64)
    truth[:, 0] = 1
    truth[0, 3:7] = [10, 11, 12, 13]
    truth[1, 3:7] = [20, 21, 22, 23]

    _, mv = build_opponent_view_np(obs, my_view, truth, TABLES)
    assert mv["labels"]["move"][0].tolist() == [12, 10, 11, 13]
    assert mv["labels"]["move"][1].tolist() == [20, 21, 22, 23]  # no reveals: paste order kept
