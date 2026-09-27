from types import SimpleNamespace

import numpy as np
import pytest

from fog.search.belief_table import JointBeliefTable
from fog.search.public_observation import PublicObservationLikelihood, public_observation_key


def _grid(probabilities, keys):
    probabilities = np.asarray(probabilities, dtype=float)
    return SimpleNamespace(branch_prob=probabilities, leaf_obs=list(range(probabilities.size)),
                           leaf_index=np.arange(probabilities.size).reshape(probabilities.shape),
                           leaf_public_keys=keys)


def test_bayes_uses_chance_likelihood_and_is_identical_for_identical_public_outcomes():
    grid = _grid(np.array([.9, .1, .1, .9]).reshape(1, 2, 1, 1, 2), ["E", "F", "E", "F"])
    model = PublicObservationLikelihood(grid)
    prior = np.array([[.5, .5]])
    post = model.posteriors(prior, np.ones((1, 1)), np.ones((2, 1)))
    np.testing.assert_allclose(post[0], [[.9, .1]])
    np.testing.assert_array_equal(post[0], post[2])
    np.testing.assert_allclose(post[1], [[.1, .9]])
    table = JointBeliefTable(np.log(prior))
    assert table.observation_step(model.likelihood("E", np.ones((1, 1)), np.ones((2, 1))))
    np.testing.assert_allclose(table.normalized(), post[0])


def test_unobserved_selected_move_is_marginalized_out():
    grid = _grid(np.ones((1, 2, 1, 2, 1)), ["KO", "KO", "KO", "KO"])
    sigma2 = np.array([[.99, .01], [.01, .99]])
    prior = np.array([[.5, .5]])
    posterior = PublicObservationLikelihood(grid).posteriors(prior, np.ones((1, 1)), sigma2)
    np.testing.assert_allclose(posterior, np.broadcast_to(prior, (4, 1, 2)))


def test_correlated_prior_does_not_resurrect_impossible_joint_worlds():
    grid = _grid(np.ones((2, 2, 1, 1, 1)), ["same", "different", "different", "same"])
    model = PublicObservationLikelihood(grid)
    prior = np.diag([.4, .6])
    strategy = np.ones((2, 1))
    posterior = model.posteriors(prior, strategy, strategy)
    np.testing.assert_allclose(posterior[[0, 3]], np.stack([prior, prior]))
    assert not posterior[[1, 2]].any(), "impossible hypothetical leaves are not posterior queries"
    np.testing.assert_allclose(model.transposed().posteriors(prior.T, strategy, strategy),
                               posterior.transpose(0, 2, 1))
    log_prior = np.full_like(prior, -np.inf)
    np.log(prior, out=log_prior, where=prior > 0)
    table = JointBeliefTable(log_prior)
    assert not table.observation_step(model.likelihood("different", strategy, strategy))


def test_public_key_censors_exact_hp_and_ignores_private_submitted_choices():
    def battle(hp, selected):
        return SimpleNamespace(
            observations={1: SimpleNamespace(events=[["", "-damage", "p2a: mon", hp]])},
            _pending_events=[], revealed={}, pending_sides=lambda: ["p1", "p2"],
            turn=2, hidden_choice=selected,
        )
    assert public_observation_key(battle("149/300", "thunderbolt")) == public_observation_key(
        battle("199/400", "icebeam"))
    assert public_observation_key(battle("140/300", "thunderbolt")) != public_observation_key(
        battle("149/300", "thunderbolt"))


def test_persistent_filter_refuses_unsupported_observation():
    table = JointBeliefTable(np.log([[.5, .5]]))
    assert not table.observation_step(np.zeros((1, 2)))
    assert table.needs_rejuvenation()
    with pytest.raises(ValueError):
        table.observation_step(np.array([[np.nan, 1.]]))


def test_native_flinch_does_not_reveal_the_prevented_move():
    from tests.test_pb_grid import _battle

    battle = _battle()
    branches, _ = battle.step_branches({"p1": "darkpulse", "p2": "thunderwave"})
    hidden_branches = 0
    for _, leaf in branches:
        moves = [event for observation in leaf.observations.values()
                 for event in observation.events if event[1] == "move"]
        if not any(event[2].startswith("p2") for event in moves):
            hidden_branches += 1
            assert not any("thunderwave" in event for event in moves)
    assert hidden_branches > 0


def test_native_flinch_still_reveals_terastallization():
    from tests.test_pb_grid import _battle

    branches, _ = _battle().step_branches({"p1": "darkpulse", "p2": "thunderwave-tera"})
    for _, leaf in branches:
        events = [event for observation in leaf.observations.values() for event in observation.events]
        assert any(event[1] == "-terastallize" and event[2].startswith("p2") for event in events)
        assert all(event[3] == event[3].lower() for event in events if event[1] == "-terastallize")
