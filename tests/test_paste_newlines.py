import numpy as np
import pytest

from fog.constants.tables import TABLES
from fog.env.team_truth import truth_tokens_from_paste
from fog.env.teams import make_paired_teambuilders, make_pool_teambuilder, to_packed
from fog.team_identity import team_group_key


PASTE = """Gholdengo @ Choice Scarf
Ability: Good as Gold
Tera Type: Water
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Recover
- Trick

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Ground
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Stealth Rock
"""


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_network_pastes_preserve_packing_groups_private_truth_and_both_teambuilders(newline):
    network = PASTE.replace("\n", newline)
    packed = to_packed(PASTE)
    assert len(packed.split("]")) == 2
    assert to_packed(network) == packed
    assert team_group_key(network) == team_group_key(PASTE)
    expected = truth_tokens_from_paste(PASTE, TABLES)
    assert expected[0, 0] == TABLES.species_index("gholdengo")
    assert expected[1, 0] == TABLES.species_index("greattusk")
    assert np.all(expected[2:] == -1)
    assert np.array_equal(truth_tokens_from_paste(network, TABLES), expected)
    assert make_pool_teambuilder([network], seed=1).yield_team() == packed
    p1, p2 = make_paired_teambuilders([network], n_games=2, seed=1)
    assert p1.yield_team() == p2.yield_team() == packed
