"""Load-bearing tests for the unordered set-autoregressive belief model."""
from __future__ import annotations

import torch

from fog.models.belief import N_ATTRS, N_SLOTS, SLOT_ATTRS, BeliefModel
from fog.training.losses import belief_output_loss

SIZES = dict(n_species=10, n_items=6, n_abilities=5, n_moves=12, n_spreads=4, n_tera=19)
STATE_DIM = 32
D = 16
B = 3
_FAMILY = ("species", "item", "ability", "move", "move", "move", "move", "spread", "tera")


def make_model() -> BeliefModel:
    torch.manual_seed(0)
    return BeliefModel(state_dim=STATE_DIM, d=D, **SIZES).eval()


def full_truth_grid() -> torch.Tensor:
    gen = torch.Generator().manual_seed(4)
    grid = torch.zeros(B, N_SLOTS, len(SLOT_ATTRS), dtype=torch.long)
    highs = [
        SIZES["n_species"], SIZES["n_items"], SIZES["n_abilities"], 0, 0, 0, 0,
        SIZES["n_spreads"], SIZES["n_tera"],
    ]
    for column, high in enumerate(highs):
        if column not in (3, 4, 5, 6):
            grid[:, :, column] = torch.randint(
                0, high, (B, N_SLOTS), generator=gen
            )
    for row in range(B):
        for slot in range(N_SLOTS):
            grid[row, slot, 3:7] = torch.randperm(
                SIZES["n_moves"], generator=gen
            )[:4]
    return grid


def revealed_species(truth: torch.Tensor) -> torch.Tensor:
    tokens = torch.full_like(truth, -1)
    tokens[:, :, 0] = truth[:, :, 0]
    return tokens


def logit_at(output, flat: int) -> torch.Tensor:
    slot, column = divmod(flat, len(SLOT_ATTRS))
    family = _FAMILY[column]
    return (
        output.logits["move"][:, slot, column - 3]
        if family == "move" else output.logits[family][:, slot]
    )


def test_move_reveal_storage_order_is_invariant():
    """Known moves form a set: swapping two revealed move storage cells changes no prediction."""
    model = make_model()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    tokens[:, 0, 3:5] = truth[:, 0, 3:5]
    swapped = tokens.clone()
    swapped[:, 0, 3], swapped[:, 0, 4] = tokens[:, 0, 4], tokens[:, 0, 3]
    c_dyn = torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(8))

    first = model(c_dyn, {"tokens": tokens, "masks": None})
    second = model(c_dyn, {"tokens": swapped, "masks": None})
    for family in ("species", "item", "ability", "spread", "tera"):
        assert torch.allclose(first.logits[family], second.logits[family], atol=1e-6)
    # All unresolved move queries are the same "insert another move" query.
    assert torch.allclose(
        first.logits["move"][:, 0, 2], second.logits["move"][:, 0, 2], atol=1e-6
    )


def test_hidden_move_queries_are_exchangeable():
    model = make_model()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    out = model(torch.zeros(B, STATE_DIM), {"tokens": tokens, "masks": None})
    for subslot in (1, 2, 3):
        assert torch.allclose(
            out.logits["move"][:, :, 0], out.logits["move"][:, :, subslot], atol=1e-6
        )


def test_pokemon_slot_permutation_equivariance():
    """Reordering the six Pokemon only reorders output rows; it cannot change the represented team."""
    model = make_model()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    tokens[:, :, 1] = truth[:, :, 1]
    tokens[:, 1, 8] = truth[:, 1, 8]
    c_dyn = torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(9))
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])

    base = model(c_dyn, {"tokens": tokens, "masks": None})
    moved = model(c_dyn, {"tokens": tokens[:, permutation], "masks": None})
    for family in ("species", "item", "ability", "spread", "tera", "move"):
        assert torch.allclose(
            moved.logits[family], base.logits[family][:, permutation], atol=1e-5
        ), family


def test_all_reveals_are_immediate_context():
    """A late-named Tera reveal changes an item prediction in the same forward."""
    model = make_model()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    c_dyn = torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(10))
    before = model(c_dyn, {"tokens": tokens, "masks": None})
    tokens[:, 0, 8] = truth[:, 0, 8]
    after = model(c_dyn, {"tokens": tokens, "masks": None})
    assert not torch.allclose(before.logits["item"][:, 0], after.logits["item"][:, 0])


def test_training_forward_scores_exactly_one_matched_route_step():
    model = make_model().train()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    out = model(
        torch.zeros(B, STATE_DIM),
        {"tokens": tokens, "masks": None},
        teacher_values=truth,
        generator=torch.Generator().manual_seed(11),
    )
    assert out.loss_valid is not None and out.loss_targets is not None and out.order is not None
    scored = sum(mask.reshape(B, -1).sum(dim=1) for mask in out.loss_valid.values())
    assert torch.equal(scored, torch.ones(B, dtype=torch.long))

    # The route step is a real differentiable conditional under the caller's censoring mask.
    labels = {
        "species": truth[:, :, 0], "item": truth[:, :, 1],
        "ability": truth[:, :, 2], "move": truth[:, :, 3:7],
        "spread": truth[:, :, 7], "tera": truth[:, :, 8],
    }
    valid = {name: torch.ones_like(value, dtype=torch.bool) for name, value in labels.items()}
    loss = belief_output_loss(out, labels, valid)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_teacher_values_after_selected_step_cannot_leak_back():
    """Changing a future route value leaves the selected next-field conditional byte-identical."""
    model = make_model()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    c_dyn = torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(12))
    seed = 13
    base = model(
        c_dyn, {"tokens": tokens, "masks": None}, teacher_values=truth,
        tail_order="canonical", generator=torch.Generator().manual_seed(seed),
    )
    selected = []
    for row in range(B):
        chosen = next(
            flat
            for flat in range(N_ATTRS)
            if bool(logit_at_mask(base.loss_valid, row, flat))
        )
        selected.append(chosen)

    perturbed = truth.clone().reshape(B, N_ATTRS)
    for row, chosen in enumerate(selected):
        route = base.order[row]
        rank = int(torch.nonzero(route == chosen, as_tuple=True)[0][0])
        later = int(route[min(rank + 1, int((route >= 0).sum()) - 1)])
        slot, column = divmod(later, len(SLOT_ATTRS))
        family = _FAMILY[column]
        size = {
            "species": SIZES["n_species"], "item": SIZES["n_items"],
            "ability": SIZES["n_abilities"], "move": SIZES["n_moves"],
            "spread": SIZES["n_spreads"], "tera": SIZES["n_tera"],
        }[family]
        perturbed[row, later] = (perturbed[row, later] + 1) % size
    changed = model(
        c_dyn, {"tokens": tokens, "masks": None},
        teacher_values=perturbed.reshape_as(truth), tail_order="canonical",
        generator=torch.Generator().manual_seed(seed),
    )
    for row, flat in enumerate(selected):
        assert torch.allclose(logit_at(base, flat)[row], logit_at(changed, flat)[row])


def logit_at_mask(valid: dict[str, torch.Tensor], row: int, flat: int) -> torch.Tensor:
    slot, column = divmod(flat, len(SLOT_ATTRS))
    family = _FAMILY[column]
    return (
        valid["move"][row, slot, column - 3]
        if family == "move" else valid[family][row, slot]
    )


def test_random_routes_are_not_a_disguised_fixed_order():
    model = make_model()
    revealed = model._revealed_mask(revealed_species(full_truth_grid()))
    route_a, length_a, log_q_a = model._route(
        revealed, "random", torch.Generator().manual_seed(20)
    )
    route_b, length_b, log_q_b = model._route(
        revealed, "random", torch.Generator().manual_seed(21)
    )
    assert torch.equal(length_a, length_b)
    assert not torch.equal(route_a, route_b)
    assert torch.isfinite(log_q_a).all() and (log_q_a < 0).all()
    assert torch.isfinite(log_q_b).all()


def test_sample_trace_and_reproducibility():
    model = make_model()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    args = (torch.zeros(B, STATE_DIM), {"tokens": tokens, "masks": None})
    first = model.sample_team(
        *args, k=2, generator=torch.Generator().manual_seed(30), return_trace=True
    )
    second = model.sample_team(
        *args, k=2, generator=torch.Generator().manual_seed(30), return_trace=True
    )
    assert torch.equal(first.teams, second.teams)
    assert torch.equal(first.order, second.order)
    assert torch.allclose(first.log_joint_path_prob, second.log_joint_path_prob)
    assert first.teams.shape == (B, 2, 6, 9)
    assert torch.isfinite(first.log_value_prob).all()
    assert torch.isfinite(first.log_route_prob).all()
    assert (first.log_route_prob < 0).all()


def test_path_probability_is_finite_and_route_dependent():
    model = make_model()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    c_dyn = torch.zeros(B, STATE_DIM)
    first = model.path_logprob(
        c_dyn, {"tokens": tokens, "masks": None}, truth,
        generator=torch.Generator().manual_seed(40),
    )
    second = model.path_logprob(
        c_dyn, {"tokens": tokens, "masks": None}, truth,
        generator=torch.Generator().manual_seed(41),
    )
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.allclose(first, second)


def test_revealed_rows_are_exact_deltas():
    model = make_model()
    truth = full_truth_grid()
    tokens = revealed_species(truth)
    out = model(torch.zeros(B, STATE_DIM), {"tokens": tokens, "masks": None})
    for row in range(B):
        for slot in range(N_SLOTS):
            value = int(tokens[row, slot, 0])
            logits = out.logits["species"][row, slot]
            assert logits[value] == 0
            assert torch.isinf(logits).sum() == SIZES["n_species"] - 1


def test_agent_chain_does_not_change_policy_path():
    from tests.test_belief import make_agent_batch
    from fog.models.agent import FogAgent

    torch.manual_seed(50)
    agent = FogAgent()
    for module in agent.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0
    agent.train()
    batch = make_agent_batch()
    truth = full_truth_grid()[:2]
    labels = {
        "species": truth[:, :, 0], "item": truth[:, :, 1],
        "ability": truth[:, :, 2], "move": truth[:, :, 3:7],
        "spread": truth[:, :, 7], "tera": truth[:, :, 8],
    }
    plain = agent(dict(batch))
    chained = agent(dict(batch, belief_targets=labels))
    assert chained.belief_chain is not None
    assert torch.allclose(plain.log_probs, chained.log_probs, atol=1e-6)
    assert torch.allclose(plain.value, chained.value, atol=1e-6)
