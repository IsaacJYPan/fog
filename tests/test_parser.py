"""Parser tests (spec §4.3). The parser is the highest-defect-risk component; these are placeholders
that will fuzz once the parser exists.

Planned (from spec):
  * every legal request JSON from 1000 random Showdown battles -> nonempty legality mask
  * re-simulate 500 parsed replays through the local server, diff state trajectories
  * quarantine rate 2-5% (fail the build if >5%)
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="parser not implemented in scaffolding (§4.3 stub)")
def test_replay_roundtrip_state_diff():
    ...


@pytest.mark.skip(reason="parser not implemented in scaffolding (§4.3 stub)")
def test_quarantine_rate_under_5pct():
    ...
