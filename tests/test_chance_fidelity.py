"""Phase H instrumentation (fog/eval/chance_fidelity.py): the Damage-instruction matcher finds
the engine's average-roll damage, the crit family share reproduces the engine's 1/16 branch
weighting, and roll reconstruction flags a knockout straddle exactly when HP falls inside the
16-roll span."""
from __future__ import annotations

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

ATTACKER = """Hydreigon @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Dark Pulse
- Flash Cannon
- Fire Blast
"""

DEFENDER = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave
"""


def _battle():
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(ATTACKER, DEFENDER, seed=0, battle_tag="chance-fid-test")


def test_pair_stats_match_damage_and_crit_share():
    from fog.eval.chance_fidelity import pair_damage_stats

    b = _battle()
    draco = next(i for i, m in enumerate(b.active("p1").moves)
                 if str(m.id).lower().replace(" ", "") == "dracometeor")
    m1 = b.engine_action("p1", draco)
    m2 = b.engine_action("p2", int(np.flatnonzero(b.action_mask("p2"))[0]))
    s = pair_damage_stats(b.state, m1, m2)
    assert s["matched"] >= 2, "the crit and non-crit Draco branches both match their base"
    fams = [f for f in s["crit_families"] if f["side"] == "Two"]
    assert fams, "side two takes classified move damage"
    for f in fams:
        assert f["crit_share"] == pytest.approx(1 / 16, abs=1e-6), \
            "probe finding 5: the engine branches crits at 1/16 (Gen 9 real rate is 1/24)"


def test_roll_reconstruction_and_straddle_flag():
    from fog.eval.chance_fidelity import _avg_roll, _rolls

    rolls = _rolls(190)
    assert len(rolls) == 16 and rolls[0] == 161 and rolls[-1] == 190
    assert _avg_roll(190) == 175, "probe fact 2: branch damage = floor(base * 0.925)"
    # a defender at 170 HP straddles (rolls 161..190 span it); at 100 HP it does not
    assert 0 < sum(1 for x in rolls if x >= 170) / 16 < 1
    assert sum(1 for x in rolls if x >= 100) / 16 == 1.0
