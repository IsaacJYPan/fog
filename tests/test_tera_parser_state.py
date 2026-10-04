from types import SimpleNamespace

import numpy as np
import pytest

from fog.constants.schema import TYPE_INDEX
from fog.constants.tables import TABLES
from fog.encoding.entities import ENTITY_RAW_DIM, L, _fill_mon_record
from fog.encoding.pokemon_state import is_terastallized
from fog.encoding.reveals import opponent_views_from_battle


def test_real_parser_activation_is_encoded_without_revealing_an_unused_tera_type():
    from poke_env.battle import Pokemon, PokemonType

    mon = Pokemon(gen=9, species='garchomp')
    # Own request metadata can contain the submitted type before public activation.
    mon._terastallized_type = PokemonType.WATER
    row = np.zeros(ENTITY_RAW_DIM, dtype=np.float32)
    _fill_mon_record(row, mon, TABLES)
    assert row[L.tera_used] == 0
    assert row[L.tera_type:L.tera_used].sum() == 0
    battle = SimpleNamespace(opponent_team={'p2: Garchomp': mon}, teampreview_opponent_team=[mon])
    assert opponent_views_from_battle(battle, TABLES)[0].tera is None

    mon.terastallize('Water')
    row.fill(0)
    _fill_mon_record(row, mon, TABLES)
    assert row[L.tera_used] == 1
    assert row[L.tera_type + TYPE_INDEX['water']] == 1
    assert row[L.tera_type:L.tera_used].sum() == 1
    assert opponent_views_from_battle(battle, TABLES)[0].tera == TYPE_INDEX['water']


@pytest.mark.parametrize('value', [False, True])
def test_engine_adapter_flag_remains_supported(value):
    assert is_terastallized(SimpleNamespace(terastallized=value)) == value


def test_modern_flag_takes_precedence_over_an_obsolete_attribute():
    assert not is_terastallized(SimpleNamespace(is_terastallized=False, terastallized=True))
