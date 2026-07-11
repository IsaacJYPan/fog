"""Full-scale parser fuzz checks (spec §4.3) — placeholders for corpus-scale validation.

The replay parser itself is implemented (fog/scripts/parse_replays.py; 0% quarantine on the first
150 replays) and its components are tested in test_events.py / test_tracker.py / the golden-battle
test. What remains here are the corpus-scale checks from the spec, which need a large scraped
corpus + a live local server and are run as part of the M4 data pass, not CI:

  * every legal request JSON from 1000 random Showdown battles -> nonempty legality mask
  * re-simulate 500 parsed replays through the local server, diff state trajectories
  * quarantine rate 2-5% (fail the build if >5%)
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="corpus-scale check (needs >=500 replays + live server; M4 data pass)")
def test_replay_roundtrip_state_diff():
    ...


@pytest.mark.skip(reason="corpus-scale check (needs a large scraped corpus; M4 data pass)")
def test_quarantine_rate_under_5pct():
    ...
