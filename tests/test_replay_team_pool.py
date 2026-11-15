"""The replay pool's held-out composition, evidence, privacy and legality contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fog.constants.tables import TABLES
from fog.env.teams import SHOWDOWN_DIR, validate_team
from fog.scripts.build_replay_team_pool import (
    ShowdownValidator,
    build_pool,
    preview_composition,
    process_replay,
    revealed_team,
    split_teams,
    tokens_to_paste,
)
from fog.training.replay_policy_data import EXACT, complete_private_team

POOL = Path("data/teams/replay_v1")
HAS_SHOWDOWN = (SHOWDOWN_DIR / "dist/sim/team-validator.js").exists()


def _replay(rid="gen9ou-fixture", *, rating=1800, timestamp=100, move_count=3):
    from fog.env.team_truth import truth_tokens_from_paste

    paste = next(Path("data/teams/pool_a").glob("*.txt")).read_text()
    truth = truth_tokens_from_paste(paste, TABLES)
    species = [TABLES.species["features"][TABLES.species["ids"][int(r[0])]]["name"]
               for r in truth]
    lines = ["|player|p1|private-fixture-one|", "|player|p2|private-fixture-two|"]
    for role in ("p1", "p2"):
        lines.extend(f"|poke|{role}|{s}|" for s in species)
        for slot, (s, row) in enumerate(zip(species, truth)):
            token = f"{role}a: secret-nickname-{slot}"
            lines.append(f"|switch|{token}|{s}|100/100")
            for move in row[3:3 + move_count]:
                lines.append(f"|move|{token}|{TABLES.moves['ids'][int(move)]}|{token}")
            lines.append(f"|-item|{token}|{TABLES.items['ids'][int(row[1])]}")
            lines.append(f"|-ability|{token}|{TABLES.abilities['ids'][int(row[2])]}")
            if slot == 0:
                lines.append(f"|-terastallize|{token}|Water")
    return {"id": rid, "rating": rating, "uploadtime": timestamp, "log": "\n".join(lines),
            "players": ["private-fixture-one", "private-fixture-two"]}


def _team(composition, *, rid="gen9ou-old", revealed=30, side="p1"):
    return {"composition": sorted(composition), "replay_id": rid, "side": side,
            "revealed_fields": {"total": revealed}, "paste": "fixture"}


def _preview(composition, rid="gen9ou-old", timestamp=1):
    return {"composition": sorted(composition), "replay_id": rid, "uploadtime": timestamp}


def test_split_excludes_any_older_preview_and_keeps_most_revealed_legal_team():
    old, new = ["a", "b", "c", "d", "e", "f"], ["a", "b", "c", "d", "e", "g"]
    teams = [_team(old), _team(old, rid="gen9ou-holdout", revealed=40),
             _team(new, rid="gen9ou-holdout", revealed=35)]
    # Older preview matters even if its side was below 1700 or had too few revealed moves.
    previews = [_preview(old), _preview(old, "gen9ou-holdout", 2),
                _preview(new, "gen9ou-holdout", 2)]
    splits, report = split_teams(teams, previews, {"gen9ou-holdout"}, seed=0)
    assert splits["test"] == [dict(teams[-1], first_seen_uploadtime=2,
                                   first_seen_replay_id="gen9ou-holdout")]
    assert splits["train"][0]["revealed_fields"]["total"] == 40
    assert report["unique_valid_compositions"] == 2
    assert report["deduplicated_sides"] == 1


def test_splits_disjoint_seeded_capped_and_independent_of_input_order():
    teams = [_team([str(i)] * 6) for i in range(1200)]
    previews = [_preview(t["composition"]) for t in teams]
    for i in range(200):
        c = [f"new-{i}"] * 6
        teams.append(_team(c, rid="gen9ou-new"))
        previews.append(_preview(c, "gen9ou-new", 2))
    a, report = split_teams(teams, previews, {"gen9ou-new"}, seed=3)
    b, _ = split_teams(teams[::-1], previews[::-1], {"gen9ou-new"}, seed=3)
    c, _ = split_teams(teams, previews, {"gen9ou-new"}, seed=4)
    assert a == b and a != c
    assert {s: len(t) for s, t in a.items()} == {"train": 600, "validation": 150, "test": 150}
    assert report["before_caps"] == {"train": 1020, "validation": 180, "test": 200}
    groups = [{tuple(t["composition"]) for t in values} for values in a.values()]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))


def test_evidence_preserved_and_personal_metadata_not_exported():
    replay = _replay()
    result = process_replay(replay, seed=8, validator=lambda _: [])
    assert len(result["teams"]) == 2
    for team in result["teams"]:
        assert team["revealed_fields"]["moves"] == 18
        assert team["sampled_fields"]["moves"] == 6
        assert team["revealed_fields"]["species"] == 6
        assert team["revealed_fields"]["item"] == 6
        assert team["revealed_fields"]["ability"] == 6
        assert team["revealed_fields"]["tera"] == 1
        assert team["sampled_fields"]["nature"] == 6
        assert team["sampled_fields"]["evs"] == 36
        assert team["revealed_fields"]["total"] + team["sampled_fields"]["total"] == 90
    exported = json.dumps(result)
    assert "private-fixture" not in exported and "secret-nickname" not in exported
    events = [line.split("|") for line in replay["log"].splitlines()]
    revealed = revealed_team(events, "p1")
    completed, provenance = complete_private_team(revealed, seed=8)
    np.testing.assert_array_equal(completed[:, :3], revealed[:, :3])
    for slot in range(6):
        expected = set(revealed[slot, 3:7][revealed[slot, 3:7] >= 0])
        observed = set(completed[slot, 3:7][provenance[slot, 3:7] == EXACT])
        assert expected == observed


@pytest.mark.parametrize("kind", ["rating", "moves", "species", "preview"])
def test_filters_and_previews_are_independent_per_side(kind):
    replay = _replay(rating=1699 if kind == "rating" else 1700,
                     move_count=2 if kind == "moves" else 3)
    if kind == "species":
        replay["log"] = replay["log"].replace("|poke|p1|", "|poke|p1|MissingSpecies", 1)
    if kind == "preview":
        replay["log"] = replay["log"].replace("|poke|p1|", "|ignored|p1|", 1)
    result = process_replay(replay, seed=1, validator=lambda _: [])
    assert len(result["teams"]) == (0 if kind in ("rating", "moves") else 1)
    assert len(result["previews"]) == (2 if kind in ("rating", "moves") else 1)
    if result["teams"]:
        assert result["teams"][0]["side"] == "p2"


def test_invalid_completion_is_dropped_without_retry():
    calls = []

    def reject(paste):
        calls.append(paste)
        return ["illegal team"]

    result = process_replay(_replay(), seed=1, validator=reject)
    assert len(calls) == 2
    assert result["counts"]["validator_rejected"] == 2
    assert result["teams"] == []


def test_preview_composition_is_a_species_multiset():
    events = [["", "poke", "p1", s] for s in ["Garchomp", "Clefable"] * 3]
    assert preview_composition(events, "p1") == tuple(sorted(["garchomp", "clefable"] * 3))


@pytest.mark.skipif(not HAS_SHOWDOWN, reason="pinned Showdown not built")
def test_persistent_validator_agrees_with_cli():
    paste = next(Path("data/teams/pool_a").glob("*.txt")).read_text()
    invalid = paste.replace(" @ ", " @ InvalidItem", 1)
    with ShowdownValidator() as validator:
        assert not validator(paste)
        assert validator(invalid)
        for text in (paste, invalid):
            ok, _ = validate_team(text)
            assert (not validator(text)) == ok


@pytest.mark.skipif(not HAS_SHOWDOWN, reason="pinned Showdown not built")
def test_real_build_reproduces_and_resumes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for replay in (_replay("gen9ou-old", timestamp=1), _replay("gen9ou-new", timestamp=2)):
        (raw / f"{replay['id']}.json").write_text(json.dumps(replay))
    ids = tmp_path / "holdout.txt"
    ids.write_text("gen9ou-new\n")
    a, b, cache = tmp_path / "a", tmp_path / "b", tmp_path / "cache"
    build_pool(raw, ids, a, seed=0, work_dir=cache)
    build_pool(raw, ids, b, seed=0, work_dir=cache)
    assert {str(p.relative_to(a)): p.read_bytes() for p in a.rglob("*") if p.is_file()} == {
        str(p.relative_to(b)): p.read_bytes() for p in b.rglob("*") if p.is_file()}
    build_pool(raw, ids, b, seed=0)
    assert {str(p.relative_to(a)): p.read_bytes() for p in a.rglob("*") if p.is_file()} == {
        str(p.relative_to(b)): p.read_bytes() for p in b.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="checkpoint options changed"):
        build_pool(raw, ids, b, seed=1, work_dir=cache)
    ids.write_text("gen9ou-old\n")
    with pytest.raises(ValueError, match="not the newest"):
        build_pool(raw, ids, b, seed=0, work_dir=cache)


@pytest.mark.skipif(not POOL.exists(), reason="committed replay pool not built")
def test_committed_pool_disjoint_compositions_and_manifest_hashes():
    from fog.env.team_truth import truth_tokens_from_paste

    root = json.loads((POOL / "manifest.json").read_text())
    groups = []
    holdout = set(Path(root["holdout_ids"]).read_text().splitlines())
    for split in ("train", "validation", "test"):
        path = POOL / split / "manifest.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == root["split_manifest_sha256"][split]
        records = json.loads(path.read_text())["teams"]
        assert len(records) == root["split_sizes"][split]
        compositions = set()
        assert len(list((POOL / split).glob("*.txt"))) == len(records)
        for record in records:
            paste = (POOL / split / record["file"]).read_text()
            tokens = truth_tokens_from_paste(paste, TABLES)
            # A canonical species-only paste cannot carry original nicknames or log text.
            tokens[tokens[:, 1] < 0, 1] = TABLES.item_index("<NO_ITEM>")
            assert tokens_to_paste(tokens) == paste
            assert set(record) == {"composition", "file", "first_seen_replay_id",
                                   "first_seen_uploadtime", "rating", "replay_id",
                                   "revealed_fields", "sampled_fields", "side", "uploadtime"}
            actual = sorted(TABLES.species["ids"][int(row[0])] for row in tokens)
            assert actual == record["composition"]
            assert tuple(actual) not in compositions
            compositions.add(tuple(actual))
            assert record["rating"] >= 1700
            assert record["revealed_fields"]["moves"] >= 14
            if split == "test":
                assert record["first_seen_replay_id"] in holdout
        groups.append(compositions)
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))


@pytest.mark.skipif(not POOL.exists() or not Path("data/metamon_gen9ou_r1500").exists(),
                    reason="pool or raw import missing")
def test_test_compositions_absent_from_every_older_raw_preview():
    root = json.loads((POOL / "manifest.json").read_text())
    holdout = set(Path(root["holdout_ids"]).read_text().splitlines())
    records = json.loads((POOL / "test/manifest.json").read_text())["teams"]
    test_compositions = {tuple(r["composition"]) for r in records}
    for path in Path(root["raw"]).glob("gen9ou-*.json"):
        if path.stem in holdout:
            continue
        replay = json.loads(path.read_text())
        events = [line.split("|") for line in replay["log"].splitlines() if line.startswith("|")]
        for role in ("p1", "p2"):
            assert preview_composition(events, role) not in test_compositions, path.stem


@pytest.mark.skipif(not POOL.exists() or not HAS_SHOWDOWN, reason="pool or Showdown missing")
def test_every_committed_team_is_validator_clean():
    with ShowdownValidator() as validator:
        for path in sorted(POOL.glob("*/*.txt")):
            assert validator(path.read_text()) == [], path


def test_scripted_anchor_factories_use_existing_baselines():
    from fog.eval.showdown_direct_selection import load_controller_factory

    for name in ("random", "maxdamage", "simpleheuristics"):
        factory = load_controller_factory("fog.eval.showdown_direct_selection:scripted_controller",
                                          {"name": name})
        assert factory() == name
    with pytest.raises(ValueError, match="unknown scripted"):
        load_controller_factory("fog.eval.showdown_direct_selection:scripted_controller",
                                {"name": "unrecognised"})()
