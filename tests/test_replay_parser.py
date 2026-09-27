from pathlib import Path

import numpy as np

from fog.constants.tables import TABLES
from fog.scripts.parse_replays import (
    parse_replay,
    parse_replay_policy_versions,
    replay_files,
    rows_to_arrays,
)
from fog.scripts.replay_preflight_audit import _perspective_issues
from fog.training.replay_policy_data import EXACT, INFERRED, complete_private_team
from fog.training.replay_supervision import _preview_species_id


def test_replay_files_excludes_scraper_metadata(tmp_path: Path):
    for name in ("gen9ou-a.json", "gen9ou-b.json", "_cursor.json", "_other.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    assert [path.name for path in replay_files(tmp_path)] == [
        "gen9ou-a.json",
        "gen9ou-b.json",
    ]
    assert [path.name for path in replay_files(tmp_path, 1)] == ["gen9ou-a.json"]


def test_replay_supervision_normalizes_tera_display_forms_to_preview_species():
    assert _preview_species_id("Ogerpon-Cornerstone-Tera, F, tera:Rock") == "ogerponcornerstone"


def test_replay_parser_builds_opposite_public_perspectives():
    log = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|1
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Garchomp, M|
|poke|p2|Corviknight, M|
|teampreview
|start
|switch|p1a: Chomp|Garchomp, M|100/100
|switch|p2a: Bird|Corviknight, M|100/100
|turn|1
|move|p1a: Chomp|Earthquake|p2a: Bird
|-immune|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|-heal|p2a: Bird|100/100
|upkeep
|turn|2
|move|p1a: Chomp|Earthquake|p2a: Bird
|-immune|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|-heal|p2a: Bird|100/100
|win|Alice
|player|p2|"""
    replay = {"log": log, "players": ["Alice", "Bob"], "rating": 1600}
    rows = parse_replay(replay)["rows"]
    p1 = next(row for row in rows if row["role"] == "p1")
    p2 = next(row for row in rows if row["role"] == "p2")
    chomp = TABLES.species_index("garchomp")
    corv = TABLES.species_index("corviknight")

    assert p1["belief"][0, 0] == corv
    assert p2["belief"][0, 0] == chomp
    assert p1["self_belief"][0, 0] == chomp
    assert p2["self_belief"][0, 0] == corv
    assert p1["obs"].opp_revealed["tokens"][0, 0] == corv
    assert p2["obs"].opp_revealed["tokens"][0, 0] == chomp
    assert p1["obs"].entity_ids[0, 0] == chomp
    assert p2["obs"].entity_ids[0, 0] == corv
    assert not np.array_equal(p1["belief"], p2["belief"])
    assert _perspective_issues(rows, replay, "test") == []

    p2["belief"] = p1["belief"].copy()
    messages = _perspective_issues(rows, replay, "test")
    assert any("p2: opponent target species" in message for message in messages)
    assert any("identical perspective targets" in message for message in messages)


def test_replay_policy_labels_require_previously_revealed_move_features():
    """A replay move is safe to imitate only after its public feature row already existed."""
    log = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|1
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Garchomp|
|poke|p2|Corviknight|
|teampreview
|start
|switch|p1a: Chomp|Garchomp|100/100
|switch|p2a: Bird|Corviknight|100/100
|turn|1
|move|p1a: Chomp|Earthquake|p2a: Bird
|-immune|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|turn|2
|move|p1a: Chomp|Earthquake|p2a: Bird
|-immune|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|win|Alice
|player|p2|"""
    rows = parse_replay({"log": log, "players": ["Alice", "Bob"], "rating": 1600})["rows"]
    p1_rows = sorted((row for row in rows if row["role"] == "p1"), key=lambda row: row["turn"])

    assert [row["policy_valid"] for row in p1_rows] == [False, True]
    arrays = rows_to_arrays(rows)
    assert arrays["policy_valid"].dtype == np.bool_
    assert int(arrays["policy_valid"].sum()) == 2
    assert arrays["action_features"].shape[:2] == (len(rows), 14)


def test_private_set_imputation_retains_exact_fields_and_marks_every_field():
    from fog.constants.schema import SET_N_ATTRS

    revealed = np.full((6, SET_N_ATTRS), -1, dtype=np.int64)
    revealed[0, 0] = TABLES.species_index("garchomp")
    revealed[0, 3] = TABLES.move_index("earthquake")
    completed, provenance = complete_private_team(revealed, seed=7)

    assert completed[0, 3] == revealed[0, 3]
    assert provenance[0, 0] == EXACT
    assert provenance[0, 3] == EXACT
    assert np.all(provenance[0, 1:] > 0)
    assert np.any(provenance[0] == INFERRED)
    assert np.all(provenance[1:] == 0)


def test_policy_reconstruction_does_not_copy_a_future_opponent_item_reveal_backward():
    prefix = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|1
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Garchomp|
|poke|p2|Corviknight|
|teampreview
|start
|switch|p1a: Chomp|Garchomp|100/100
|switch|p2a: Bird|Corviknight|100/100
|turn|1
|move|p1a: Chomp|Earthquake|p2a: Bird
|-immune|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|turn|2
"""
    suffix = """|move|p1a: Chomp|Earthquake|p2a: Bird
|-immune|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|win|Alice
|player|p2|"""
    without_reveal = parse_replay({"log": prefix + suffix, "players": ["Alice", "Bob"]})["rows"]
    with_reveal = parse_replay({
        "log": prefix + "|-item|p2a: Bird|Leftovers\n" + suffix,
        "players": ["Alice", "Bob"],
    })["rows"]
    first_a = next(row for row in without_reveal if row["role"] == "p1" and row["turn"] == 1)
    first_b = next(row for row in with_reveal if row["role"] == "p1" and row["turn"] == 1)

    assert first_a["obs"].opp_revealed["tokens"][0, 1] == -1
    assert first_b["obs"].opp_revealed["tokens"][0, 1] == -1
    assert np.array_equal(first_a["obs"].entities, first_b["obs"].entities)


def test_policy_versions_reconstruct_tera_choice_and_action_inputs():
    log = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|1
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Garchomp|
|poke|p2|Corviknight|
|teampreview
|start
|switch|p1a: Chomp|Garchomp|100/100
|switch|p2a: Bird|Corviknight|100/100
|turn|1
|-terastallize|p1a: Chomp|Ground
|move|p1a: Chomp|Earthquake|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|turn|2
|move|p1a: Chomp|Earthquake|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|win|Alice
|player|p2|"""
    result = parse_replay_policy_versions(
        {"log": log, "players": ["Alice", "Bob"]}, replay_id="tera-test", imputations=1
    )
    p1 = next(row for row in result["rows"] if row["role"] == "p1")
    arrays = rows_to_arrays(result["rows"])

    assert 4 <= p1["action"] < 8
    p1_rows = [row for row in result["rows"] if row["role"] == "p1"]
    assert len(p1_rows) == 2
    assert not p1_rows[1]["obs"].legality_mask[4:8].any()
    assert p1_rows[0]["obs"].legality_mask[4:8].any()
    assert arrays["legality_mask"][0].sum() >= 2
    assert np.array_equal(
        arrays["action_features"][..., 0].astype(bool), arrays["legality_mask"]
    )


def test_replay_parser_remaps_illusion_moves_to_the_revealed_preview_slot():
    """Moves used behind Illusion belong to Zoroark, never to its disguise's set."""
    log = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|2
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Arcanine-Hisui, M|
|poke|p1|Zoroark-Hisui, M|
|poke|p2|Corviknight, M|
|teampreview
|start
|switch|p1a: Arcanine|Arcanine-Hisui, M|100/100
|switch|p2a: Bird|Corviknight, M|100/100
|turn|1
|move|p1a: Arcanine|Bitter Malice|p2a: Bird
|replace|p1a: Zoroark|Zoroark-Hisui, M|100/100
|-end|p1a: Zoroark|Illusion
|move|p2a: Bird|Roost|p2a: Bird
|turn|2
|move|p1a: Zoroark|Bitter Malice|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|win|Alice
|player|p2|"""
    rows = parse_replay({"log": log, "players": ["Alice", "Bob"], "rating": 1600})["rows"]
    target = next(row["belief"] for row in rows if row["role"] == "p2")
    arcanine = TABLES.species_index("arcaninehisui")
    zoroark = TABLES.species_index("zoroarkhisui")
    bitter_malice = TABLES.move_index("bittermalice")

    assert not (target[target[:, 0] == arcanine, 3:7] == bitter_malice).any()
    assert (target[target[:, 0] == zoroark, 3:7] == bitter_malice).any()


def test_replay_parser_resolves_an_unreplaced_illusion_from_a_unique_move():
    """A revealing move can identify Illusion even when the replay ends before |replace|."""
    log = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|2
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Arcanine-Hisui, M|
|poke|p1|Zoroark-Hisui, M|
|poke|p2|Corviknight, M|
|teampreview
|start
|switch|p1a: Arcanine|Arcanine-Hisui, M|100/100
|switch|p2a: Bird|Corviknight, M|100/100
|turn|1
|move|p1a: Arcanine|Bitter Malice|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|turn|2
|win|Alice
|player|p2|"""
    rows = parse_replay({"log": log, "players": ["Alice", "Bob"], "rating": 1600})["rows"]
    target = next(row["belief"] for row in rows if row["role"] == "p2")
    arcanine = TABLES.species_index("arcaninehisui")
    zoroark = TABLES.species_index("zoroarkhisui")
    bitter_malice = TABLES.move_index("bittermalice")

    assert not (target[target[:, 0] == arcanine, 3:7] == bitter_malice).any()
    assert (target[target[:, 0] == zoroark, 3:7] == bitter_malice).any()


def test_replay_parser_censors_moves_copied_by_imposter_transform():
    log = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|1
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Alomomola|
|poke|p2|Ditto|
|teampreview
|start
|switch|p1a: Alo|Alomomola|100/100
|switch|p2a: Ditto|Ditto|100/100
|turn|1
|-transform|p2a: Ditto|p1a: Alo|[from] ability: Imposter
|move|p2a: Ditto|Scald|p1a: Alo
|move|p1a: Alo|Wish|p1a: Alo
|turn|2
|move|p2a: Ditto|Scald|p1a: Alo
|move|p1a: Alo|Wish|p1a: Alo
|win|Alice
|player|p2|"""
    rows = parse_replay({"log": log, "players": ["Alice", "Bob"], "rating": 1600})["rows"]
    target = next(row["belief"] for row in rows if row["role"] == "p1")
    ditto = TABLES.species_index("ditto")
    scald = TABLES.move_index("scald")

    assert not (target[target[:, 0] == ditto, 3:7] == scald).any()


def test_replay_parser_transform_copy_cannot_remap_prior_mew_evidence():
    """Copied moves must not redirect a transformed Mew's prior submitted-set labels."""
    log = """|gametype|singles
|player|p1|Alice|1|1600
|player|p2|Bob|2|1600
|teamsize|p1|1
|teamsize|p2|2
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Iron Valiant|
|poke|p2|Mew|
|poke|p2|Ninetales-Alola|
|teampreview
|start
|switch|p1a: Valiant|Iron Valiant|100/100
|switch|p2a: Mew|Mew|100/100
|turn|1
|move|p2a: Mew|Transform|p1a: Valiant
|-transform|p2a: Mew|p1a: Valiant
|move|p1a: Valiant|Moonblast|p2a: Mew
|turn|2
|move|p2a: Mew|Moonblast|p1a: Valiant
|move|p1a: Valiant|Moonblast|p2a: Mew
|win|Alice
|player|p2|"""
    rows = parse_replay({"log": log, "players": ["Alice", "Bob"], "rating": 1600})["rows"]
    target = next(row["belief"] for row in rows if row["role"] == "p1")
    mew = TABLES.species_index("mew")
    ninetales = TABLES.species_index("ninetalesalola")
    transform = TABLES.move_index("transform")

    assert (target[target[:, 0] == mew, 3:7] == transform).any()
    assert not (target[target[:, 0] == ninetales, 3:7] == transform).any()
