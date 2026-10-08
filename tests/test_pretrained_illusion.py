"""Run in the isolated native environment; no private team inputs are needed."""

from datetime import datetime

import pytest

pytest.importorskip("metamon")

from metamon.backend.replay_parser.forward import SimProtocol
from metamon.backend.replay_parser.replay_state import ParsedReplay
from metamon.backend.replay_parser.exceptions import ZoroarkException
from fog.pretrained.illusion import install_illusion_family


@pytest.mark.parametrize("species", ["Zorua", "Zorua-Hisui", "Zoroark", "Zoroark-Hisui"])
def test_replace_preserves_physical_condition_and_restores_disguised_reserve(species):
    install_illusion_family()
    replay = ParsedReplay(gameid="illusion-test", time_played=datetime.now(), gen=9)
    protocol = SimProtocol(replay)
    for line in [
        f"|poke|p1|{species}, M|",
        "|poke|p1|Voltorb|",
        "|poke|p2|Pikachu|",
        "|start",
        "|switch|p1a: Voltorb|Voltorb|100/100",
        "|switch|p2a: Pikachu|Pikachu|100/100",
        "|turn|1",
        "|-damage|p1a: Voltorb|37/100",
        f"|replace|p1a: {species}|{species}, M",
    ]:
        protocol.interpret_message(line.split("|")[1:])
    current = replay.turnlist[-1]
    revealed = current.active_pokemon_1[0]
    assert revealed.name == species and revealed.current_hp == 37
    reserve = next(p for p in current.pokemon_1 if p and p.name == "Voltorb")
    assert reserve.current_hp == 100
    assert len(current.replacements_1) == 1


def test_unrelated_species_still_rejects_replace():
    install_illusion_family()
    replay = ParsedReplay(gameid="invalid-replace", time_played=datetime.now(), gen=9)
    protocol = SimProtocol(replay)
    for line in [
        "|poke|p1|Pikachu|",
        "|poke|p1|Voltorb|",
        "|start",
        "|switch|p1a: Voltorb|Voltorb|100/100",
        "|turn|1",
    ]:
        protocol.interpret_message(line.split("|")[1:])
    with pytest.raises(ZoroarkException):
        protocol.interpret_message(["replace", "p1a: Pikachu", "Pikachu"])
