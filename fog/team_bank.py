"""Public Gen 9 OU team-bank discovery and curation.

The important boundary in this module is between an *exact public paste* and a replay-derived
reconstruction.  Only the former is collected.  Replays are retained as provenance/evidence, but
they are never used to invent unrevealed moves, items, abilities, natures, or EVs.

The collector is deliberately standard-library-only.  It can crawl a small, explicit allowlist of
Smogon resource/tournament threads and high-ladder RMT index pages, fetch Pokepaste's public
``/raw`` representation, canonicalize with poke-env, validate with the repository's pinned
Showdown checkout, and emit a package already understood by :func:`fog.env.teams.load_pool`.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

USER_AGENT = "fog/0.1 (public-team research collector; contact via project repository)"
POKEPASTE_HOSTS = {"pokepast.es", "www.pokepast.es"}
SMOGON_HOSTS = {"smogon.com", "www.smogon.com"}
REPLAY_RE = re.compile(
    r"https?://replay\.pokemonshowdown\.com/[^\s\"'<>]+", re.IGNORECASE
)
ELO_RE = re.compile(
    r"\b(1[5-9]\d{2}|2[0-4]\d{2})\s*(?=\+|\s*(?:elo|rating)\b)", re.IGNORECASE
)
GXE_RE = re.compile(r"\b(\d{2}(?:\.\d+)?)\s*%\s*gxe\b", re.IGNORECASE)
RANK_RE = re.compile(
    r"\b(?:peak(?:ed)?(?:\s+rank)?|rank|top)\s*#?\s*(\d{1,4})\b", re.IGNORECASE
)
TOURNAMENT_RE = re.compile(
    r"\b(?:spl|wcop|olt|scl|ost|smogon tour)\b", re.IGNORECASE
)
THREAD_PATH_RE = re.compile(
    r"^/forums/threads/[^/?#]+\.\d+(?:/page-\d+)?/?$", re.IGNORECASE
)


@dataclass(frozen=True)
class SourceSpec:
    """An explicit discovery source.

    ``kind`` is ``resource`` or ``tournament`` for direct thread crawling, and ``rmt_forum`` for
    discovering high-ladder RMT threads from forum index pages.
    """

    name: str
    kind: str
    url: str
    max_pages: int = 1
    base_quality: int = 0
    min_claimed_elo: int = 1850
    ruleset_epoch: str = "unknown"
    stop_marker: str | None = None


@dataclass
class PasteCandidate:
    paste_url: str
    source_name: str
    source_kind: str
    source_url: str
    source_title: str = ""
    evidence_text: str = ""
    replay_urls: list[str] = field(default_factory=list)
    base_quality: int = 0
    ruleset_epoch: str = "unknown"


@dataclass
class TeamRecord:
    team_id: str
    paste_sha256: str
    packed_sha256: str
    quality_score: int
    quality_tier: str
    paste_url: str
    source_name: str
    source_kind: str
    source_url: str
    source_title: str
    ruleset_epoch: str
    claimed_elo: int | None
    claimed_rank: int | None
    claimed_gxe: float | None
    replay_urls: list[str]
    file: str


@dataclass
class Rejection:
    paste_url: str
    source_url: str
    reason: str


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = dict(attrs)
        self._href = values.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def extract_links(document: str, base_url: str) -> list[tuple[str, str]]:
    parser = _LinkParser()
    parser.feed(document)
    return [
        (urllib.parse.urljoin(base_url, html.unescape(href)), text)
        for href, text in parser.links
        if href
    ]


def canonical_paste_url(url: str) -> str | None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname and parsed.hostname.lower() not in POKEPASTE_HOSTS:
        return None
    match = re.fullmatch(r"/([0-9a-f]{16})(?:/raw)?/?", parsed.path, re.IGNORECASE)
    if not match:
        return None
    return f"https://pokepast.es/{match.group(1).lower()}"


def canonical_thread_url(url: str) -> str | None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname and parsed.hostname.lower() not in SMOGON_HOSTS:
        return None
    if not THREAD_PATH_RE.fullmatch(parsed.path):
        return None
    # A page link is useful during crawling, but a post fragment/query is not part of identity.
    path = re.sub(r"/page-\d+/?$", "", parsed.path).rstrip("/") + "/"
    return urllib.parse.urlunsplit(("https", "www.smogon.com", path, "", ""))


def page_url(url: str, page: int) -> str:
    if page <= 1:
        return url
    return url.rstrip("/") + f"/page-{page}"


def _max_match(pattern: re.Pattern[str], text: str, cast: Callable[[str], float]) -> float | None:
    values = [cast(match.group(1)) for match in pattern.finditer(text)]
    return max(values) if values else None


def evidence_metrics(text: str) -> tuple[int | None, int | None, float | None]:
    elo = _max_match(ELO_RE, text, int)
    ranks = [int(match.group(1)) for match in RANK_RE.finditer(text)]
    gxe = _max_match(GXE_RE, text, float)
    return (
        int(elo) if elo is not None else None,
        min(ranks) if ranks else None,
        float(gxe) if gxe is not None else None,
    )


def rmt_title_qualifies(title: str, min_claimed_elo: int = 1850) -> bool:
    elo, rank, gxe = evidence_metrics(title)
    return bool(
        (elo is not None and elo >= min_claimed_elo)
        or (rank is not None and rank <= 200)
        or (gxe is not None and gxe >= 80.0)
        or re.search(r"\bolt\b.*\bqual", title, re.IGNORECASE)
    )


def quality_score(candidate: PasteCandidate) -> tuple[int, int | None, int | None, float | None]:
    """Score provenance/evidence without pretending that a claim is independently verified."""
    text = f"{candidate.source_title}\n{candidate.evidence_text}"
    elo, rank, gxe = evidence_metrics(text)
    score = candidate.base_quality
    if candidate.source_kind in {"resource", "tournament"}:
        score += 10
    if TOURNAMENT_RE.search(text):
        score += 5
    if elo is not None:
        score += (
            30
            if elo >= 2000
            else 25
            if elo >= 1900
            else 18
            if elo >= 1850
            else 12
            if elo >= 1700
            else 8
            if elo >= 1600
            else 0
        )
    if rank is not None:
        score += 20 if rank <= 25 else 15 if rank <= 100 else 8 if rank <= 200 else 0
    if gxe is not None:
        score += 15 if gxe >= 85 else 10 if gxe >= 80 else 5 if gxe >= 75 else 0
    score += min(len(set(candidate.replay_urls)), 3) * 5
    return min(score, 100), elo, rank, gxe


def quality_tier(score: int, source_kind: str) -> str:
    if source_kind == "resource" and score >= 75:
        return "gold"
    if source_kind == "tournament" and score >= 70:
        return "tournament_public"
    if score >= 70:
        return "high_evidence_candidate"
    if score >= 50:
        return "candidate"
    return "low_evidence"


def normalize_paste(paste: str) -> str:
    # Pokepaste occasionally serves CR-CR-LF. Treat any run of CRs plus its optional LF as one
    # newline; the usual sequential ``replace("\r\n")`` would turn CR-CR-LF into two blank lines
    # and make every field look like a separate Pokemon block.
    normalized_newlines = re.sub(r"\r+\n?", "\n", paste)
    lines = [line.rstrip() for line in normalized_newlines.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    out: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if out and not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out) + "\n"


def obvious_full_team(paste: str) -> tuple[bool, str]:
    """Cheap rejection before invoking poke-env or spawning the Showdown validator."""
    blocks = [block for block in re.split(r"\n\s*\n", normalize_paste(paste)) if block.strip()]
    if len(blocks) != 6:
        return False, f"expected 6 Pokemon blocks, found {len(blocks)}"
    for index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        moves = [line for line in lines if line.startswith("- ")]
        if not moves:
            return False, f"Pokemon block {index} has no moves"
        if not any(line.startswith("Ability:") for line in lines):
            return False, f"Pokemon block {index} has no explicit ability"
    return True, ""


class HttpFetcher:
    """Polite cached HTTP client with robots checks and bounded retries."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        delay: float = 1.0,
        timeout: int = 30,
        retries: int = 3,
        respect_robots: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.respect_robots = respect_robots
        self._last_request = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _cache_path(self, url: str) -> Path:
        suffix = ".txt" if url.endswith("/raw") else ".html"
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}{suffix}"

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urllib.parse.urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            robots_url = origin + "/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                parser.read()
                self._robots[origin] = parser
            except (OSError, urllib.error.URLError):
                # An unavailable robots file is not evidence of a prohibition. Keep the explicit
                # allowlist and rate limit as the safety boundary.
                self._robots[origin] = None
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(USER_AGENT, url)

    def get_text(self, url: str, *, use_cache: bool = True) -> str:
        cache_path = self._cache_path(url)
        if use_cache and cache_path.exists():
            # read_text() performs universal-newline translation. That is normally convenient, but
            # on Windows it can turn a cached CR-CR-LF paste into LF-LF before normalize_paste gets
            # a chance to repair it.
            return cache_path.read_bytes().decode("utf-8")
        if not self._allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")

        error: Exception | None = None
        for attempt in range(self.retries):
            wait = self.delay - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                    charset = response.headers.get_content_charset() or "utf-8"
                self._last_request = time.monotonic()
                text = payload.decode(charset, errors="replace")
                # Preserve the response bytes' line-ending structure across platforms.
                cache_path.write_bytes(text.encode("utf-8"))
                return text
            except urllib.error.HTTPError as exc:
                error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2**attempt)
            except (OSError, urllib.error.URLError) as exc:
                error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"failed to fetch {url}: {error}")


def _document_candidates(
    document: str,
    *,
    page: str,
    source: SourceSpec,
    source_title: str = "",
) -> list[PasteCandidate]:
    replay_urls = sorted(set(REPLAY_RE.findall(document)))
    plain = re.sub(r"<[^>]+>", " ", document)
    evidence = html.unescape(re.sub(r"\s+", " ", plain))
    candidates: list[PasteCandidate] = []
    for url, _ in extract_links(document, page):
        paste_url = canonical_paste_url(url)
        if paste_url:
            candidates.append(
                PasteCandidate(
                    paste_url=paste_url,
                    source_name=source.name,
                    source_kind=source.kind,
                    source_url=page,
                    source_title=source_title,
                    evidence_text=evidence,
                    replay_urls=replay_urls,
                    base_quality=source.base_quality,
                    ruleset_epoch=source.ruleset_epoch,
                )
            )
    return candidates


def discover_source(source: SourceSpec, fetch: Callable[[str], str]) -> list[PasteCandidate]:
    candidates: list[PasteCandidate] = []
    if source.kind != "rmt_forum":
        for page_number in range(1, source.max_pages + 1):
            current = page_url(source.url, page_number)
            document = fetch(current)
            if source.stop_marker and source.stop_marker in document:
                document = document.split(source.stop_marker, 1)[0]
            candidates.extend(_document_candidates(document, page=current, source=source))
        return candidates

    threads: dict[str, str] = {}
    for page_number in range(1, source.max_pages + 1):
        current = page_url(source.url, page_number)
        document = fetch(current)
        for url, title in extract_links(document, current):
            thread = canonical_thread_url(url)
            if thread and rmt_title_qualifies(title, source.min_claimed_elo):
                threads.setdefault(thread, title)

    for thread, title in threads.items():
        document = fetch(thread)
        candidates.extend(
            _document_candidates(document, page=thread, source=source, source_title=title)
        )
    return candidates


def _merge_candidates(candidates: Iterable[PasteCandidate]) -> list[PasteCandidate]:
    merged: dict[str, PasteCandidate] = {}
    for candidate in candidates:
        existing = merged.get(candidate.paste_url)
        if existing is None or quality_score(candidate)[0] > quality_score(existing)[0]:
            merged[candidate.paste_url] = candidate
        elif existing is not None:
            existing.replay_urls = sorted(set(existing.replay_urls + candidate.replay_urls))
    return list(merged.values())


def collect_bank(
    sources: list[SourceSpec],
    out_dir: Path,
    *,
    fetch: Callable[[str], str],
    canonicalize: Callable[[str], str],
    validate: Callable[[str, str], tuple[bool, str]],
    battle_format: str = "gen9ou",
    min_quality: int = 60,
    target: int = 200,
) -> dict:
    """Discover, fetch, validate, deduplicate, and export a loadable team bank."""
    out_dir.mkdir(parents=True, exist_ok=True)
    teams_dir = out_dir / "teams"
    teams_dir.mkdir(parents=True, exist_ok=True)
    for stale_team in teams_dir.glob("*.txt"):
        stale_team.unlink()

    discovered: list[PasteCandidate] = []
    source_errors: list[dict[str, str]] = []
    for source in sources:
        try:
            discovered.extend(discover_source(source, fetch))
        except Exception as exc:  # noqa: BLE001 - one bad external source must not abort the bank
            source_errors.append({"source": source.name, "error": f"{type(exc).__name__}: {exc}"})

    records: list[TeamRecord] = []
    rejections: list[Rejection] = []
    packed_seen: set[str] = set()
    pastes: list[str] = []
    for candidate in sorted(
        _merge_candidates(discovered), key=lambda item: quality_score(item)[0], reverse=True
    ):
        if len(records) >= target:
            break
        score, elo, rank, gxe = quality_score(candidate)
        if score < min_quality:
            rejections.append(
                Rejection(candidate.paste_url, candidate.source_url, f"quality {score} < {min_quality}")
            )
            continue
        try:
            paste = normalize_paste(fetch(candidate.paste_url + "/raw"))
        except Exception as exc:  # noqa: BLE001 - quarantine a bad external paste
            rejections.append(
                Rejection(candidate.paste_url, candidate.source_url, f"fetch: {type(exc).__name__}")
            )
            continue
        complete, reason = obvious_full_team(paste)
        if not complete:
            rejections.append(Rejection(candidate.paste_url, candidate.source_url, reason))
            continue
        try:
            packed = canonicalize(paste)
        except Exception as exc:  # noqa: BLE001 - parser errors belong in the rejection manifest
            rejections.append(
                Rejection(candidate.paste_url, candidate.source_url, f"parse: {type(exc).__name__}")
            )
            continue
        if len(packed.split("]")) != 6:
            rejections.append(
                Rejection(candidate.paste_url, candidate.source_url, "packed team is not six Pokemon")
            )
            continue
        packed_hash = hashlib.sha256(packed.encode()).hexdigest()
        if packed_hash in packed_seen:
            rejections.append(Rejection(candidate.paste_url, candidate.source_url, "semantic duplicate"))
            continue
        ok, validator_output = validate(paste, battle_format)
        if not ok:
            rejections.append(
                Rejection(candidate.paste_url, candidate.source_url, f"illegal: {validator_output}")
            )
            continue

        paste_hash = hashlib.sha256(paste.encode()).hexdigest()
        team_id = packed_hash[:16]
        filename = f"{team_id}.txt"
        (teams_dir / filename).write_bytes(paste.encode("utf-8"))
        packed_seen.add(packed_hash)
        pastes.append(paste)
        records.append(
            TeamRecord(
                team_id=team_id,
                paste_sha256=paste_hash,
                packed_sha256=packed_hash,
                quality_score=score,
                quality_tier=quality_tier(score, candidate.source_kind),
                paste_url=candidate.paste_url,
                source_name=candidate.source_name,
                source_kind=candidate.source_kind,
                source_url=candidate.source_url,
                source_title=candidate.source_title,
                ruleset_epoch=candidate.ruleset_epoch,
                claimed_elo=elo,
                claimed_rank=rank,
                claimed_gxe=gxe,
                replay_urls=sorted(set(candidate.replay_urls)),
                file=f"teams/{filename}",
            )
        )

    generated_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "format": battle_format,
        "target": target,
        "min_quality": min_quality,
        "team_count": len(records),
        "sources": [asdict(source) for source in sources],
        "source_errors": source_errors,
        "teams": [asdict(record) for record in records],
        "rejections": [asdict(rejection) for rejection in rejections],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # load_pool accepts this package directly. Metadata stays beside it in manifest.json.
    (out_dir / "pool.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": battle_format,
                "generated_at": generated_at,
                "teams": pastes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest
