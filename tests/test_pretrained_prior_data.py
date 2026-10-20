import copy
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pytest

from fog.constants.schema import NATURE_INDEX, TYPE_INDEX
from fog.constants.tables import TABLES
from fog.pretrained.data import composition
from fog.pretrained.prior_data import build_prior, candidates, validated_row, validate_export
from fog.training.replay_candidates import ReplayCandidateBank


ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "runs/pretrained/team_pools_20261006_v1"


def tusk():
    return dict(
        species="Great Tusk",
        item="Leftovers",
        ability="Protosynthesis",
        nature="Jolly",
        moves=["Earthquake", "Knock Off", "Rapid Spin", "Close Combat"],
        evs={"atk": 252, "def": 4, "spe": 252},
        teraType="Water",
    )


def archive(path, pastes):
    with tarfile.open(path, "w:gz") as stream:
        for i, paste in enumerate(pastes):
            raw = paste.encode()
            member = tarfile.TarInfo(f"gen9ou/{i}.gen9ou_team")
            member.size = len(raw)
            stream.addfile(member, io.BytesIO(raw))
    return path


def test_validator_export_codec_preserves_modeled_categories_without_mutating_input():
    source = tusk()
    before = copy.deepcopy(source)
    row = validated_row(source)
    assert source == before
    assert row[:3].tolist() == [
        TABLES.species_index("greattusk"),
        TABLES.item_index("leftovers"),
        TABLES.ability_index("protosynthesis"),
    ]
    assert row[3:7].tolist() == sorted(
        TABLES.move_index(m) for m in ("earthquake", "knockoff", "rapidspin", "closecombat")
    )
    assert row[7] == NATURE_INDEX["jolly"] and row[14] == TYPE_INDEX["water"]
    assert row[8:14].tolist() == [0, 63, 1, 0, 0, 63]


def test_incomplete_or_invalid_prior_fields_are_rejected_instead_of_invented():
    for changes in [
        dict(moves=["Earthquake"] * 4),
        dict(evs={"hp": 252, "atk": 252, "spe": 252}),
        dict(ability="Levitate"),
    ]:
        with pytest.raises(ValueError):
            validated_row(dict(tusk(), **changes))
    missing = tusk()
    del missing["teraType"]
    with pytest.raises(KeyError):
        validated_row(missing)


def test_reservoir_excludes_protected_rosters_before_validation_and_is_reproducible(tmp_path):
    a = "Pikachu\n- Thunderbolt\n\nCharizard\n- Flamethrower"
    b = a.replace("Thunderbolt", "Surf")
    c = "Great Tusk\n- Earthquake\n\nGholdengo\n- Make It Rain"
    path = archive(tmp_path / "a.tar.gz", [a, b, c])
    rows, count = candidates(path, {composition(a)}, maximum=10, seed=1)
    assert count == 1 and [r["paste"] for r in rows] == [c]
    assert candidates(path, {composition(a)}, maximum=10, seed=1) == (rows, count)


def test_oversized_source_member_fails_closed(tmp_path):
    path = archive(tmp_path / "a.tar.gz", ["x" * (64 * 1024 + 1)])
    with pytest.raises(ValueError, match="Oversized"):
        candidates(path, set(), maximum=2, seed=1)


@pytest.mark.skipif(
    not POOLS.exists() or not shutil.which("node"),
    reason="Local pinned simulator integration fixture",
)
def test_real_validator_prior_matches_existing_codec_and_excludes_held_out_rosters(tmp_path):
    from fog.pretrained.labels import exact_team_tokens

    train = json.loads((POOLS / "train.json").read_text())["teams"][:2]
    protected = json.loads((POOLS / "development.json").read_text())["teams"][:1]
    final = json.loads((POOLS / "final.json").read_text())["teams"][:1]
    answers = validate_export(train, ROOT / "server/pokemon-showdown")
    assert all(row["valid"] for row in answers)
    for paste, answer in zip(train, answers):
        expected = exact_team_tokens(paste, TABLES)
        expected[:, 3:7] = np.sort(expected[:, 3:7], axis=1)
        assert np.array_equal(expected, np.stack([validated_row(mon) for mon in answer["team"]]))
    pools = tmp_path / "pools"
    pools.mkdir()
    for name, teams in [("development", protected), ("final", final)]:
        (pools / (name + ".json")).write_text(json.dumps({"teams": teams}))
    source = archive(tmp_path / "teams.tar.gz", train + protected + final + train[:1])
    result = build_prior(
        source,
        pools,
        tmp_path / "prior",
        ROOT / "server/pokemon-showdown",
        maximum_teams=2,
        minimum_teams=2,
        maximum_candidates=10,
    )
    bank = ReplayCandidateBank(tmp_path / "prior/bank.npz")
    assert (
        len(bank.tokens) == 12
        and bank.metadata["prior_only"]
        and not bank.metadata["private_truth"]
    )
    with np.load(tmp_path / "prior/bank.npz") as data:
        assert not set(data["composition_ids"]).intersection(
            composition(team) for team in protected + final
        )
    assert result["counts"]["accepted_teams"] == 2 and len(result["excluded_compositions"]) == 2
