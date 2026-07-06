"""The golden-battle integration test (spec §10): the full encoding stack must produce
BYTE-IDENTICAL tensors for the committed scripted battle across refactors.

Regenerate after an INTENTIONAL encoding change (and say so in the commit message):

    FOG_REGEN_GOLDEN=1 python -m pytest tests/test_golden_battle.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.golden_fixture import golden_arrays

GOLDEN = Path(__file__).parent / "golden" / "golden_battle.npz"


def test_golden_battle_byte_identical():
    arrays = golden_arrays()
    if os.environ.get("FOG_REGEN_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, **arrays)
        if os.environ.get("FOG_REGEN_GOLDEN") != "1":
            pytest.skip("golden file generated on first run — commit tests/golden/ and re-run")
    stored = np.load(GOLDEN)
    assert set(stored.files) == set(arrays), (
        f"golden key set changed: {sorted(set(arrays) ^ set(stored.files))} — regenerate deliberately"
    )
    for key in stored.files:
        assert np.array_equal(stored[key], arrays[key]), (
            f"encoder output changed for {key!r}. If intentional, regenerate with "
            f"FOG_REGEN_GOLDEN=1 and flag it in review; if not, you broke the encoders."
        )


def test_golden_spotchecks():
    """Semantic anchors so the golden file itself can't silently encode a bug."""
    from fog.constants.tables import TABLES
    from fog.encoding.entities import L

    a = golden_arrays()
    ent, ids = a["entities"], a["entity_ids"]

    # my Great Tusk consumed Booster Energy (tracker); Kingambit sits at +2 Atk with brn
    assert ent[0, L.booster] == 1.0
    assert ent[1, L.boosts + 0] == pytest.approx(2 / 6)
    # Dragapult fainted
    assert ent[3, L.fainted] == 1.0 and ent[3, L.hp] == 0.0
    # last_used_move: my Glowking's last move was Sludge Bomb (turn 5)
    assert ids[4, 7] == TABLES.move_index("sludgebomb")
    # opponent Tusk: item knocked/consumed -> <NO_ITEM> is KNOWLEDGE (epistemic flag set)
    assert ent[6, L.epistemic + 0] == 1.0
    # opponent Gholdengo terastallized Fairy -> tera revealed
    ghold_slot = 7
    assert ent[ghold_slot, L.tera_used] == 1.0
    # field: my hazards (SR + 2 spikes) and their reflect/tailwind, Future Sight pending on THEIR side
    o_haz = 6 + 6
    assert a["field"][o_haz] == 1.0 and a["field"][o_haz + 1] == pytest.approx(2 / 3)
    # history: 5 turns, most recent last; turn-5 record has opponent tera flag set
    assert a["history"].shape[0] == 5
    from fog.encoding.events import _OFF_OPP

    assert a["history"][4, _OFF_OPP + 4] == 1.0
    # mirror: my Glowking's revealed moves visible, Zamazenta's item hidden
    mv = a["my_view_tokens"]
    assert mv[4, 3] >= 0          # Glowking revealed at least one move
    assert mv[5, 1] == -1         # Zamazenta item never revealed
    assert a["my_truth_species"][5] == TABLES.species_index("zamazenta")
