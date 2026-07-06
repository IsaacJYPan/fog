"""Generative pool (scripts/build_generative_pool.py) — experiment-intent tests.

The pool exists to give M3 real hidden information, so the tests assert the properties the
experiment depends on, not just shapes: every sampled attribute lands inside the belief
vocabulary (a truth token of <OTHER_*>/-1 would silently censor supervision), all teams share
the species core (the oxygen check must pass), and generation is deterministic in the seed
(the committed bank is reproducible from pinned constants).
"""
from __future__ import annotations

import shutil

import pytest

from fog.constants.tables import GENERATED_DIR, Tables

pytestmark = pytest.mark.skipif(
    not (GENERATED_DIR / "species.json").exists(), reason="generated tables not built"
)


@pytest.fixture(scope="module")
def tables() -> Tables:
    return Tables()


@pytest.fixture(scope="module")
def pastes(tables) -> list[str]:
    from fog.scripts.build_generative_pool import generate_pool

    return generate_pool(6, seed=123, tables=tables, validate=False)


def test_deterministic_in_seed(tables):
    from fog.scripts.build_generative_pool import generate_pool

    a = generate_pool(3, seed=42, tables=tables, validate=False)
    b = generate_pool(3, seed=42, tables=tables, validate=False)
    c = generate_pool(3, seed=43, tables=tables, validate=False)
    assert a == b
    assert a != c


def test_truth_tokens_fully_in_vocabulary(tables, pastes):
    """No sampled attribute may fall back to a sentinel/censored token — that would silently
    weaken belief supervision on exactly the pool built to strengthen it."""
    from fog.constants.schema import MAX_SPREADS_PER_SPECIES, OTHER_ITEM, RARE_SPECIES
    from fog.env.team_truth import truth_tokens_from_paste

    rare = tables.species_index(RARE_SPECIES)
    other_item = tables.item_index(OTHER_ITEM)
    for paste in pastes:
        tok = truth_tokens_from_paste(paste, tables)
        assert (tok[:, 0] >= 0).all() and (tok[:, 0] != rare).all()          # species
        assert (tok[:, 1] >= 0).all() and (tok[:, 1] != other_item).all()    # item
        assert (tok[:, 2] >= 0).all()                                        # ability
        assert (tok[:, 3:7] >= 0).all()                                      # 4 moves
        assert (tok[:, 7] >= 0).all() and (tok[:, 7] < MAX_SPREADS_PER_SPECIES).all()  # spread
        assert (tok[:, 8] >= 0).all() and (tok[:, 8] < 18).all()             # tera (a real type)


def test_moves_distinct_and_learnset_legal(tables, pastes):
    from fog.env.team_truth import truth_tokens_from_paste

    learnset = tables._learnset_matrix
    for paste in pastes:
        tok = truth_tokens_from_paste(paste, tables)
        for row in tok:
            moves = [int(m) for m in row[3:7]]
            assert len(set(moves)) == 4
            assert all(learnset[int(row[0]), m] for m in moves)


def test_shared_core_passes_oxygen_check(tables, pastes):
    """All teams share one species multiset -> hidden_info_warning is None (β>0 runs allowed);
    and species-multiset matching is ambiguous by construction (the recording teambuilder, not
    inference, must be the truth source — same contract as POOL-A)."""
    from fog.env.team_truth import PoolTruth, truth_tokens_from_paste

    truth = PoolTruth(pastes, tables)
    assert truth.hidden_info_warning() is None
    species = truth_tokens_from_paste(pastes[0], tables)[:, 0].tolist()
    assert truth.match(species) is None  # ambiguous on >1 identical-core teams


def test_entropy_report_positive(tables):
    from fog.scripts.build_generative_pool import SetSampler, species_core_from_pool

    sampler = SetSampler(species_core_from_pool("data/teams/pool_a"), tables)
    report = sampler.entropy_report()
    assert set(report) == set(sampler.core)
    for attrs in report.values():
        assert attrs["moves_upper"] > 0
        assert attrs["item"] >= 0 and attrs["tera"] >= 0
    # the pool's raison d'être: materially more hidden info than 8 fixed variants (~3 bits)
    assert sum(sum(a.values()) for a in report.values()) > 20


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_generated_team_is_validator_legal(tables):
    from fog.env.teams import SHOWDOWN_DIR, validate_team
    from fog.scripts.build_generative_pool import generate_pool

    if not SHOWDOWN_DIR.exists():
        pytest.skip("pinned Showdown checkout missing")
    paste = generate_pool(1, seed=7, tables=tables, validate=False)[0]
    ok, out = validate_team(paste)
    assert ok, out
