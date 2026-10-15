import copy
import shutil

import pytest

from fog.env.showdown_direct import DEFAULT_SHOWDOWN_ROOT, ShowdownDirectClient, ShowdownWorkerError
from fog.training.teacher_replay import public_trace


@pytest.mark.skipif(shutil.which("node") is None or not (DEFAULT_SHOWDOWN_ROOT / "dist/sim/index.js").is_file(),
                    reason="requires the pinned local Showdown build")
def test_reseed_preserves_both_player_requests_and_prefix_then_matches_seeded_fork():
    options = {
        "formatid": "gen9customgame@@@!teampreview", "strictChoices": True,
        "seed": "sodium," + "a" * 64,
        **{side: {"name": side, "team": [{"species": "Mew", "ability": "Synchronize",
                                           "item": "Leftovers", "moves": ["Tackle", "Splash"]}]}
           for side in ("p1", "p2")},
    }
    seed = "sodium," + "b" * 64
    with ShowdownDirectClient() as client:
        session, _ = client.create_session(options)
        first = client.step_session(session, ["move 2", "move 2"], include_state=True)
        before = copy.deepcopy(first.state)
        expected = client.sample(before, ["move 1", "move 1"], seed=seed)
        client.reseed_session(session, seed=seed)
        current = client.get_session(session)
        assert current.messages == {"p1": (), "p2": ()} and current.public_messages == ()
        assert current.turn == first.turn and current.requests == first.requests
        actual = client.step_session(session, ["move 1", "move 1"], include_state=True)
        assert public_trace(actual.public_messages) == public_trace(expected["public_log"])
        assert actual.requests == expected["requests"]
        assert before == first.state
        with pytest.raises(ShowdownWorkerError, match="explicit sodium seed"):
            client.reseed_session(session, seed="historical future")
        client.close_session(session)
