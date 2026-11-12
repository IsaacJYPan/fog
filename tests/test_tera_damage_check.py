"""Engine Tera damage against pinned Showdown (ROADMAP P1.2c). Needs Node, the Showdown build and
poke-engine; skipped otherwise."""
from __future__ import annotations

import shutil

import pytest

from fog.env.engine_battle import HAVE_ENGINE
from fog.env.showdown_direct import DEFAULT_SHOWDOWN_ROOT

SHOWDOWN_BUILT = (DEFAULT_SHOWDOWN_ROOT / "dist" / "sim" / "index.js").is_file()
pytestmark = pytest.mark.skipif(
    not HAVE_ENGINE or shutil.which("node") is None or not SHOWDOWN_BUILT,
    reason="requires poke-engine, Node and the pinned local Pokemon Showdown build",
)


def test_engine_tera_damage_matches_showdown_except_stellar():
    from fog.eval.tera_damage_check import run

    report = run()
    stellar = [r for r in report["cases"] if r["attacker_tera"] == "Stellar"]
    others = [r for r in report["cases"] if r["attacker_tera"] != "Stellar"]
    assert len(others) >= 20 and all(r["showdown_exact"] for r in others)
    assert [r["label"] for r in others if not r["agrees"]] == []
    # Same-type bonus ladder on Showdown itself: 1.5x kept, 1.5x new type, 2.0x matching type.
    by_label = {r["label"]: r for r in report["cases"]}
    plain = by_label["no Tera, original type"]["showdown_max"]
    assert by_label["new Tera type, original-type move"]["showdown_max"] == plain
    assert by_label["Tera matches original type"]["showdown_max"] == pytest.approx(
        plain * 2.0 / 1.5, abs=2)
    # Documented gap: the engine applies no Tera Stellar bonus. If this starts agreeing, the
    # engine learned Stellar and the note in the results ledger should be updated.
    assert stellar and not any(r["agrees"] for r in stellar)
