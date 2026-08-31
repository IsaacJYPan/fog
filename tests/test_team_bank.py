from __future__ import annotations

import hashlib
import json

from fog.team_bank import (
    HttpFetcher,
    PasteCandidate,
    SourceSpec,
    canonical_paste_url,
    collect_bank,
    discover_source,
    evidence_metrics,
    normalize_paste,
    obvious_full_team,
    quality_score,
    rmt_title_qualifies,
)


def _team(species: str = "Great Tusk") -> str:
    blocks = []
    for index in range(6):
        blocks.append(
            f"{species}{index} @ Leftovers\n"
            "Ability: Protosynthesis\n"
            "Tera Type: Ground\n"
            "EVs: 252 HP / 4 Atk / 252 Spe\n"
            "Jolly Nature\n"
            "- Earthquake\n"
            "- Protect\n"
        )
    return "\n".join(blocks)


def test_canonical_paste_url_accepts_only_exact_public_ids():
    assert (
        canonical_paste_url("https://pokepast.es/ABCDEF0123456789/raw?x=1")
        == "https://pokepast.es/abcdef0123456789"
    )
    assert canonical_paste_url("https://example.com/abcdef0123456789") is None
    assert canonical_paste_url("https://pokepast.es/not-an-id") is None


def test_rmt_evidence_filter_and_score():
    title = "Peak #31, 1932 ELO, 83.4% GXE - bulky offense"
    assert evidence_metrics(title) == (1932, 31, 83.4)
    assert rmt_title_qualifies(title)
    assert not rmt_title_qualifies("new player, please rate my first team")
    candidate = PasteCandidate(
        paste_url="https://pokepast.es/abcdef0123456789",
        source_name="rmt",
        source_kind="rmt_forum",
        source_url="https://www.smogon.com/forums/threads/example.1/",
        source_title=title,
        replay_urls=["r1", "r2", "r3"],
        base_quality=20,
    )
    score, elo, rank, gxe = quality_score(candidate)
    assert score >= 70
    assert (elo, rank, gxe) == (1932, 31, 83.4)


def test_representative_recent_rmt_can_clear_candidate_threshold():
    title = "1704 ELO - ladder balance"
    assert rmt_title_qualifies(title, min_claimed_elo=1600)
    candidate = PasteCandidate(
        paste_url="https://pokepast.es/abcdef0123456789",
        source_name="recent_rmt",
        source_kind="rmt_forum",
        source_url="https://www.smogon.com/forums/threads/example.1/",
        source_title=title,
        base_quality=45,
        ruleset_epoch="2026-recent-validator-confirmed",
    )
    score, elo, rank, gxe = quality_score(candidate)
    assert score == 57
    assert (elo, rank, gxe) == (1704, None, None)


def test_discover_rmt_fetches_only_qualified_threads():
    source = SourceSpec(
        name="rmt",
        kind="rmt_forum",
        url="https://www.smogon.com/forums/forums/sv-ou-teams.748/",
        min_claimed_elo=1850,
    )
    index = """
    <a href="/forums/threads/good.123/">Peak #4 2010 ELO balance</a>
    <a href="/forums/threads/new.124/">please fix my team</a>
    """
    thread = '<a href="https://pokepast.es/abcdef0123456789">paste</a>'
    documents = {
        source.url: index,
        "https://www.smogon.com/forums/threads/good.123/": thread,
    }
    found = discover_source(source, documents.__getitem__)
    assert [item.paste_url for item in found] == ["https://pokepast.es/abcdef0123456789"]
    assert found[0].ruleset_epoch == "unknown"


def test_resource_stop_marker_excludes_historical_pastes():
    source = SourceSpec(
        name="samples",
        kind="resource",
        url="https://www.smogon.com/forums/threads/samples.123/",
        stop_marker="Sample Teams Change Log",
    )
    document = """
    <a href="https://pokepast.es/aaaaaaaaaaaaaaaa">current</a>
    Sample Teams Change Log
    <a href="https://pokepast.es/bbbbbbbbbbbbbbbb">historical</a>
    """
    found = discover_source(source, lambda _: document)
    assert [item.paste_url for item in found] == ["https://pokepast.es/aaaaaaaaaaaaaaaa"]


def test_full_team_screen_rejects_description_pages():
    assert obvious_full_team(_team())[0]
    ok, reason = obvious_full_team("Description by builder: this is not an import.")
    assert not ok
    assert "expected 6" in reason


def test_normalize_paste_handles_pokepaste_cr_cr_lf():
    malformed = _team().replace("\n", "\r\r\n")
    normalized = normalize_paste(malformed)
    assert "\r" not in normalized
    assert obvious_full_team(normalized)[0]


def test_http_cache_preserves_cr_cr_lf_for_normalization(tmp_path):
    fetcher = HttpFetcher(tmp_path, respect_robots=False)
    url = "https://pokepast.es/aaaaaaaaaaaaaaaa/raw"
    fetcher._cache_path(url).write_bytes(b"Pokemon @ Item\r\r\nAbility: Test\r\r\n")
    assert fetcher.get_text(url) == "Pokemon @ Item\r\r\nAbility: Test\r\r\n"


def test_collect_bank_deduplicates_and_writes_loadable_package(tmp_path):
    source = SourceSpec(
        name="tour",
        kind="tournament",
        url="https://www.smogon.com/forums/threads/tour.123/",
        base_quality=80,
    )
    page = """
    <a href="https://pokepast.es/aaaaaaaaaaaaaaaa">one</a>
    <a href="https://pokepast.es/bbbbbbbbbbbbbbbb">duplicate</a>
    """
    paste = _team()
    documents = {
        source.url: page,
        "https://pokepast.es/aaaaaaaaaaaaaaaa/raw": paste,
        "https://pokepast.es/bbbbbbbbbbbbbbbb/raw": paste,
    }

    manifest = collect_bank(
        [source],
        tmp_path,
        fetch=documents.__getitem__,
        canonicalize=lambda value: "a]b]c]d]e]f",
        validate=lambda value, fmt: (True, ""),
        min_quality=60,
        target=10,
    )
    assert manifest["team_count"] == 1
    assert any(item["reason"] == "semantic duplicate" for item in manifest["rejections"])
    package = json.loads((tmp_path / "pool.json").read_text(encoding="utf-8"))
    assert package["format"] == "gen9ou"
    assert len(package["teams"]) == 1
    record = manifest["teams"][0]
    artifact = (tmp_path / record["file"]).read_bytes()
    assert hashlib.sha256(artifact).hexdigest() == record["paste_sha256"]
