"""The Showdown-truth reconstruction fuzzer (ROADMAP P1.3). Needs poke-engine, Node and the
Showdown build; skipped otherwise."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fog.env.engine_battle import HAVE_ENGINE
from fog.env.showdown_direct import DEFAULT_SHOWDOWN_ROOT

SHOWDOWN_BUILT = (DEFAULT_SHOWDOWN_ROOT / "dist" / "sim" / "index.js").is_file()
pytestmark = pytest.mark.skipif(
    not HAVE_ENGINE or shutil.which("node") is None or not SHOWDOWN_BUILT,
    reason="requires poke-engine, Node and the pinned local Pokemon Showdown build",
)


def test_rebuilt_states_match_showdown_on_public_and_own_fields():
    from fog.scripts.fuzz_recon_showdown import run

    report = run(Path("data/teams/pool_b"), games=6, seed=3, max_turns=40)
    assert report["positions"] > 100 and report["reconstruction_errors"] == 0
    fields = report["mismatch_positions_by_field"]
    # Faults fixed so far must stay fixed: hazards, weather and its timer, terrain, Tera flags,
    # status, own HP / stats / items / PP, and which Pokemon is active.
    fixed = ("stealth_rock", "sticky_web", "spikes", "weather", "terrain", "trick_room",
             "terastallized", "tera_type", "status", "me.hp", "me.stats", "me.item", "me.moves",
             "me.pp_zero", "active", "missing_mon", "rest_sleep", "boosts")
    assert [name for name in fields if any(key in name for key in fixed)] == []
    # Whatever remains is rare; hidden Regenerator healing is unobservable and not counted.
    other = sum(count for name, count in fields.items() if "hidden_regenerator" not in name)
    assert other <= 0.1 * report["positions"]
