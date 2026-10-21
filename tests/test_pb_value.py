"""Public-belief value-network tests on hand-built tensors.

Check candidate permutation equivariance and sensitivity to belief changes without
a simulator or backbone.
"""
import torch

from fog.models.pb_value import PBValueNet, TeamEncoder

SIZES = {
    "species": 32, "item": 12, "ability": 16, "move": 40,
    "nature": 8, "ev": 8, "tera": 6,
}


def _net(d=32):
    torch.manual_seed(0)
    return PBValueNet(state_dim=24, dyn_dim=20, d=d, n_rounds=2, sizes=SIZES).eval()


def _rand_teams(n, gen):
    # concrete teams: every attribute a valid in-range index for its family
    from fog.models.pb_value import _POS_FAMILY, N_ATTRS
    cols = [torch.randint(0, SIZES[_POS_FAMILY[p]], (n, 1), generator=gen) for p in range(N_ATTRS)]
    return torch.cat(cols, dim=1).reshape(n, 6, 15)


def _inputs(L=5, K=3, J=4, seed=1):
    gen = torch.Generator().manual_seed(seed)
    s_board = torch.randn(L, 24, generator=gen)
    c_dyn = torch.randn(L, 20, generator=gen)
    own = _rand_teams(K, gen)
    opp = _rand_teams(J, gen)
    rho = torch.rand(L, K, J, generator=gen)
    rho = rho / rho.sum(dim=(1, 2), keepdim=True)                     # each leaf's belief sums to 1
    return s_board, c_dyn, own, opp, rho


def test_output_shape_and_range():
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=5, K=3, J=4)
    with torch.no_grad():
        v = net(s_board, c_dyn, own, opp, rho)
    assert v.shape == (5, 3, 4)
    assert torch.isfinite(v).all()
    assert (v.abs() < 1.0).all()                                     # tanh-bounded payoff


def test_permutation_equivariance_over_own_particles():
    """Property 5: reorder the K own candidates (and rho's K axis) -> V's k-axis reorders identically,
    values otherwise unchanged. Particle storage order must not matter."""
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=4, K=3, J=4)
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        base = net(s_board, c_dyn, own, opp, rho)
        permuted = net(s_board, c_dyn, own[perm], opp, rho[:, perm, :])
    assert torch.allclose(permuted, base[:, perm, :], atol=1e-5)


def test_permutation_equivariance_over_opp_particles():
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=4, K=3, J=4)
    perm = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        base = net(s_board, c_dyn, own, opp, rho)
        permuted = net(s_board, c_dyn, own, opp[perm], rho[:, :, perm])
    assert torch.allclose(permuted, base[:, :, perm], atol=1e-5)


def test_belief_conditioning_is_wired():
    """Property 3: V must actually respond to rho. Same board and same candidate teams, two different
    beliefs -> the value table must differ. A network that ignored rho would be belief-independent values."""
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=3, K=3, J=4)
    gen = torch.Generator().manual_seed(99)
    rho2 = torch.rand(3, 3, 4, generator=gen)
    rho2 = rho2 / rho2.sum(dim=(1, 2), keepdim=True)
    with torch.no_grad():
        v1 = net(s_board, c_dyn, own, opp, rho)
        v2 = net(s_board, c_dyn, own, opp, rho2)
    assert not torch.allclose(v1, v2, atol=1e-3)


def test_pairs_carry_distinct_values():
    """Property 4: the table is genuinely per-pair, not a broadcast scalar — different (k, j) entries
    take different values under a generic belief."""
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=2, K=3, J=4)
    with torch.no_grad():
        v = net(s_board, c_dyn, own, opp, rho)
    # for each leaf, the K x J table is not constant
    for leaf in range(v.shape[0]):
        assert v[leaf].std() > 1e-4


def test_unk_safe_partial_team():
    """A partially specified candidate team (a -1 / out-of-range entry -> <UNK> row) stays finite."""
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=2, K=3, J=4)
    own = own.clone()
    own[0, 0, 0] = -1                                                # hide one attribute
    own[1, 2, 3] = 10_000                                           # out-of-range -> <UNK>
    with torch.no_grad():
        v = net(s_board, c_dyn, own, opp, rho)
    assert torch.isfinite(v).all()


def test_zero_belief_row_is_finite():
    """An eliminated candidate (a fully-zero rho margin) must not produce NaN messages."""
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=2, K=3, J=4)
    rho = rho.clone()
    rho[:, 0, :] = 0.0                                               # own candidate 0 eliminated
    rho = rho / rho.sum(dim=(1, 2), keepdim=True)
    with torch.no_grad():
        v = net(s_board, c_dyn, own, opp, rho)
    assert torch.isfinite(v).all()


def test_batchable_leaves_match_singletons():
    """Property 7: scoring L leaves in one batch equals scoring each alone (no cross-leaf leakage)."""
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=6, K=3, J=4)
    with torch.no_grad():
        batched = net(s_board, c_dyn, own, opp, rho)
        singles = torch.cat([net(s_board[i:i + 1], c_dyn[i:i + 1], own, opp, rho[i:i + 1])
                             for i in range(6)], dim=0)
    assert torch.allclose(batched, singles, atol=1e-5)


def test_cached_representation_matches_reference_forward():
    """Caching removes only belief-independent work; it cannot change the value definition."""
    net = _net()
    s_board, c_dyn, own, opp, rho = _inputs(L=6, K=3, J=4)
    with torch.no_grad():
        reference = net(s_board, c_dyn, own, opp, rho)
        cache = net.prepare(s_board, c_dyn, own, opp)
        cached = net.forward_cached(cache, rho)
        rho2 = torch.flip(rho, dims=(1, 2))
        rho2 = rho2 / rho2.sum(dim=(1, 2), keepdim=True)
        cached2 = net.forward_cached(cache, rho2)
    assert torch.allclose(reference, cached, atol=1e-6)
    assert not torch.allclose(cached, cached2)


def test_team_encoder_binds_attributes_to_their_pokemon():
    """Swapping items between two species changes the team; it must not be an exact collision."""
    gen = torch.Generator().manual_seed(123)
    encoder = TeamEncoder(32, SIZES).eval()
    team = _rand_teams(1, gen)
    team[0, 0, 0], team[0, 1, 0] = 1, 2                       # distinct species
    team[0, 0, 1], team[0, 1, 1] = 3, 7                       # distinct items
    swapped = team.clone()
    swapped[0, 0, 1], swapped[0, 1, 1] = team[0, 1, 1], team[0, 0, 1]
    with torch.no_grad():
        original_embedding = encoder(team)
        swapped_embedding = encoder(swapped)
    assert not torch.allclose(original_embedding, swapped_embedding, atol=1e-7)


def test_team_encoder_is_invariant_to_whole_pokemon_storage_order():
    """Reordering complete Pokémon rows changes storage only, not the team they describe."""
    gen = torch.Generator().manual_seed(124)
    encoder = TeamEncoder(32, SIZES).eval()
    team = _rand_teams(2, gen)
    order = torch.tensor([5, 2, 0, 4, 1, 3])
    with torch.no_grad():
        original = encoder(team)
        reordered = encoder(team[:, order])
    assert torch.allclose(original, reordered, atol=1e-6)
