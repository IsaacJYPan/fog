import pytest

from fog.search.native_reference import known_team_action


class Player:
    def __init__(self):
        self.branches = {}
        self.messages = []

    def update(self, message):
        self.messages.append(message)
        tag = message["battle_tag"]
        if message["op"] == "fork":
            self.branches[tag] = message.get("choice")
        if message["op"] == "close":
            self.branches.pop(tag, None)
        if message["op"] == "value":
            return {"reward": 0., "gamma": .9,
                    "value": 5. if self.branches[tag] == "move 2" else 1.}
        if message["op"] == "terminal":
            return {"reward": 100. if self.branches[tag] == "move 2" else -100.}
        if message["op"] == "act" and message["choose"]:
            return {"choice": "switch 2"}
        return {"updated": True}


class Client:
    def __init__(self, fail=False, terminal=False, force=False):
        self.calls = []
        self.fail, self.terminal, self.force = fail, terminal, force

    def sample(self, state, choices, *, seed):
        self.calls.append((choices, seed))
        if self.fail and "move 2" in choices:
            raise ValueError("invalid simulated world")
        own_wait = self.force and None not in choices
        return {"ended": self.terminal, "state": {},
                "requests": {"p1": {"wait": own_wait}, "p2": {}},
                "player_logs": {"p1": [], "p2": []}}


def proposals():
    own = {"choice": "move 1", "native_action": 0, "choices": {0: "move 1", 1: "move 2"},
           "probabilities": [.7, .3] + [0.] * 11}
    opponent = {"choice": "move 4", "native_action": 3, "choices": {2: "move 3", 3: "move 4"},
                "probabilities": [0., 0., .1, .9] + [0.] * 9}
    return own, opponent


def choose(client, player, **kwargs):
    own, opponent = proposals()
    return known_team_action(client, player, state={}, own_tag="own", opponent_tag="other",
                             role="p1", own_proposal=own, opponent_proposal=opponent, seed=5,
                             max_actions=2, **kwargs)


def test_counterfactual_choices_share_chance_and_simultaneous_reply():
    client, player = Client(), Player()
    result = choose(client, player)
    assert result.choice == "move 2" and result.changed and result.complete_grid
    assert result.simulations == 4 and not result.failures and result.privileged_known_teams
    assert client.calls[0][1] == client.calls[1][1] != client.calls[2][1]
    assert all(choices[1] == "move 4" for choices, _ in client.calls)
    assert not player.branches
    # Only cloned records are executed; the live root remains uncommitted.
    assert not any(message["op"] == "commit" for message in player.messages)
    assert result.operation_counts["fork"] == result.simulations == 4
    assert all(message["source_tag"] == "own" for message in player.messages if message["op"] == "fork")


def test_failed_action_branch_retains_policy_instead_of_dropping_bad_samples():
    client, player = Client(fail=True), Player()
    result = choose(client, player)
    assert not result.complete_grid and result.choice == result.policy_choice == "move 1"
    assert result.failures[0]["message"] == "invalid simulated world"
    assert not player.branches


def test_terminal_shaped_reward_is_not_discounted_or_clamped_to_outcome_units():
    player = Player()
    result = choose(Client(terminal=True), player)
    assert result.scores[1]["mean_return"] == 100.
    assert result.scores[0]["mean_return"] == -100.
    assert all(message["source_tag"] == "own" for message in player.messages if message["op"] == "fork")


def test_opponent_only_replacements_resolve_before_the_own_leaf():
    client, player = Client(force=True), Player()
    result = choose(client, player, samples=1)
    assert result.complete_grid and result.forced_steps == 2
    assert client.calls[1][0] == client.calls[3][0] == [None, "switch 2"]
    assert not player.branches
    forks = [message for message in player.messages if message["op"] == "fork"]
    assert result.operation_counts["fork"] == 4
    assert [message["choice"] for message in forks if message["source_tag"] == "other"] == ["move 4"] * 2


def test_delayed_opponent_clone_failure_retains_policy_and_closes_both_branches():
    class FailingPlayer(Player):
        def update(self, message):
            if message["op"] == "fork" and message["source_tag"] == "other":
                raise ValueError("cannot copy opponent history")
            return super().update(message)

    player = FailingPlayer()
    result = choose(Client(force=True), player, samples=1)
    assert not result.complete_grid and result.choice == result.policy_choice == "move 1"
    assert result.failures[0]["message"] == "cannot copy opponent history"
    assert not player.branches


def test_delayed_opponent_fork_keeps_idle_opponent_history_without_fabricating_action():
    own, _ = proposals()
    player, client = Player(), Client()
    # The initial opponent is idle; the simulator later asks it to replace.
    calls = 0
    original_sample = client.sample

    def sample(state, choices, *, seed):
        nonlocal calls
        calls += 1
        branch = original_sample(state, choices, seed=seed)
        branch["requests"]["p1"]["wait"] = calls % 2 == 1
        return branch

    client.sample = sample
    result = known_team_action(client, player, state={}, own_tag="own", opponent_tag="other",
                               role="p1", own_proposal=own, opponent_proposal={"updated": True},
                               seed=5, max_actions=2, samples=1)
    assert result.complete_grid and result.forced_steps == 2
    forks = [message for message in player.messages if message["op"] == "fork" and message["source_tag"] == "other"]
    assert len(forks) == 2 and all(message["choice"] is None for message in forks)
    assert not player.branches


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_multiple_opponent_replacements_reuse_one_isolated_history_per_alternative(role):
    own, opponent = proposals()
    player, client = Player(), Client()
    calls = 0
    original_sample = client.sample

    def sample(state, choices, *, seed):
        nonlocal calls
        calls += 1
        branch = original_sample(state, choices, seed=seed)
        branch["requests"] = {role: {"wait": calls % 3 != 0},
                              "p2" if role == "p1" else "p1": {}}
        return branch

    client.sample = sample
    result = known_team_action(client, player, state={}, own_tag="own", opponent_tag="other",
                               role=role, own_proposal=own, opponent_proposal=opponent,
                               seed=5, max_actions=2, samples=1)
    assert result.complete_grid and result.forced_steps == 4
    assert result.choice == "move 2" and result.scores[1]["mean_return"] == 4.5
    assert result.operation_counts["fork"] == 4
    forks = [message for message in player.messages if message["op"] == "fork" and message["source_tag"] == "other"]
    assert len(forks) == 2
    for choices, _ in (client.calls[i] for i in (1, 2, 4, 5)):
        assert choices == ([None, "switch 2"] if role == "p1" else ["switch 2", None])
    assert not player.branches


def test_soft_budget_is_explicit_and_retains_policy(monkeypatch):
    monkeypatch.setattr("fog.search.native_reference.time.monotonic", lambda: 10.)
    client, player = Client(), Player()
    result = choose(client, player, seconds=1., started=0.)
    assert result.choice == "move 1" and not result.complete_grid
    assert result.failures[0]["type"] == "TimeoutError" and not client.calls


@pytest.mark.parametrize("kwargs", [{"seconds": float("nan")}, {"samples": 0},
                                   {"reply_temperature": -1}, {"prior_weight": float("inf")}])
def test_invalid_reference_budget_is_rejected(kwargs):
    with pytest.raises(ValueError, match="settings"):
        choose(Client(), Player(), **kwargs)
