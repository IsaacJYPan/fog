from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fog.constants.tables import TABLES
from fog.training.replay_supervision import oracle_censored_truth, public_replay_truth


def _battle(events, species="Garchomp", nickname="Chomp"):
    mon = SimpleNamespace(
        species=species,
        item=None,
        ability=None,
        moves={"earthquake": object()},
        terastallized=False,
        tera_type=None,
    )
    return SimpleNamespace(
        player_role="p1",
        opponent_team={f"p2: {nickname}": mon},
        teampreview_opponent_team=[SimpleNamespace(species=species)],
        observations={1: SimpleNamespace(events=events)},
        turn=2,
    )


def test_knocked_off_item_is_original_item_label_not_no_item():
    battle = _battle([
        ["", "switch", "p2a: Chomp", "Garchomp, L100", "100/100"],
        ["", "move", "p2a: Chomp", "Earthquake", "p1a: Target"],
        ["", "-enditem", "p2a: Chomp", "Leftovers", "[from] move: Knock Off"],
    ])
    labels = public_replay_truth(battle, TABLES)
    assert labels[0, 1] == TABLES.item_index("leftovers")
    assert labels[0, 3] == TABLES.move_index("earthquake")


def test_trick_received_item_is_not_mislabeled_as_submitted_item():
    battle = _battle([
        ["", "switch", "p2a: Chomp", "Garchomp, L100", "100/100"],
        ["", "-item", "p2a: Chomp", "Choice Scarf", "[from] move: Trick"],
    ])
    labels = public_replay_truth(battle, TABLES)
    assert labels[0, 1] == -1


def test_post_tera_ability_is_not_mislabeled_as_submitted_ability():
    battle = _battle([
        ["", "switch", "p2a: Chomp", "Garchomp, L100", "100/100"],
        ["", "-terastallize", "p2a: Chomp", "Dragon"],
        ["", "-ability", "p2a: Chomp", "Embody Aspect (Teal)", "boost"],
    ])
    labels = public_replay_truth(battle, TABLES)
    assert labels[0, 2] == -1


def test_effect_source_records_owner_item_and_ability():
    battle = _battle([
        ["", "switch", "p2a: Chomp", "Garchomp, L100", "100/100"],
        ["", "-heal", "p2a: Chomp", "100/100", "[from] item: Leftovers"],
        ["", "-damage", "p1a: Target", "50/100", "[from] ability: Rough Skin",
         "[of] p2a: Chomp"],
    ])
    labels = public_replay_truth(battle, TABLES)
    assert labels[0, 1] == TABLES.item_index("leftovers")
    assert labels[0, 2] == TABLES.ability_index("roughskin")


def test_effect_source_uses_primary_healed_mon_not_of_attacker():
    battle = _battle(
        [
            ["", "switch", "p2a: Steam", "Volcanion", "100/100"],
            [
                "",
                "-heal",
                "p2a: Steam",
                "100/100",
                "[from] ability: Water Absorb",
                "[of] p1a: Alomomola",
            ],
        ],
        species="Volcanion",
        nickname="Steam",
    )
    labels = public_replay_truth(battle, TABLES)
    assert labels[0, 2] == TABLES.ability_index("waterabsorb")


def test_skill_swap_records_original_ability_not_received_ability():
    battle = _battle(
        [
            ["", "switch", "p2a: Keys", "Klefki", "100/100"],
            [
                "",
                "-activate",
                "p1a: Deoxys",
                "Skill Swap",
                "Prankster",
                "Pressure",
                "[of] p2a: Keys",
            ],
            ["", "-ability", "p2a: Keys", "Pressure"],
        ],
        species="Klefki",
        nickname="Keys",
    )
    labels = public_replay_truth(battle, TABLES)
    assert labels[0, 2] == TABLES.ability_index("prankster")


def test_smeargle_sketch_expands_the_hard_move_mask():
    smeargle = TABLES.species_index("smeargle")
    assert TABLES.learnset_mask(smeargle)[TABLES.move_index("jawlock")]
    assert TABLES.learnset_mask(smeargle)[TABLES.move_index("sketch")]


def test_oracle_censor_never_exposes_unrevealed_fields():
    battle = _battle([
        ["", "switch", "p2a: Chomp", "Garchomp, L100", "100/100"],
        ["", "move", "p2a: Chomp", "Earthquake", "p1a: Target"],
    ])
    truth = np.full((6, 15), -1, dtype=np.int64)
    truth[0] = [
        TABLES.species_index("garchomp"),
        TABLES.item_index("leftovers"),
        TABLES.ability_index("roughskin"),
        TABLES.move_index("earthquake"),
        TABLES.move_index("swordsdance"),
        TABLES.move_index("scaleshot"),
        TABLES.move_index("firefang"),
        0, 0, 0, 0, 0, 0, 0,
        3,
    ]
    labels = oracle_censored_truth(battle, truth, TABLES)
    assert labels[0, 0] == truth[0, 0]
    assert labels[0, 3] == truth[0, 3]
    assert (labels[0, [1, 2, *range(4, 15)]] == -1).all()


def test_identifiability_oracle_is_evaluation_only_and_requires_one_answer():
    from fog.training.replay_overfit import _identifiable_mask

    candidates = np.full((2, 6, 15), -1, dtype=np.int64)
    candidates[:, 0, 0] = TABLES.species_index("garchomp")
    candidates[0, 0, 1] = TABLES.item_index("leftovers")
    candidates[1, 0, 1] = TABLES.item_index("choiceband")
    candidates[:, 0, 3] = TABLES.move_index("earthquake")
    candidates[0, 0, 4] = TABLES.move_index("swordsdance")
    candidates[1, 0, 4] = TABLES.move_index("firefang")
    public = np.full((1, 6, 15), -1, dtype=np.int64)
    public[0, 0, 0] = TABLES.species_index("garchomp")
    labels = np.full_like(public, -1)
    labels[0, 0, 1] = TABLES.item_index("leftovers")

    # Without the target item, the two candidates disagree: not identifiable.
    assert not _identifiable_mask(
        public, labels, candidates, family="item", slot=0, subslot=None
    )[0]
    # With item supplied as other evidence, the final move is uniquely determined.
    labels[0, 0, 3] = TABLES.move_index("earthquake")
    labels[0, 0, 4] = TABLES.move_index("swordsdance")
    assert _identifiable_mask(
        public, labels, candidates, family="move", slot=0, subslot=1
    )[0]
