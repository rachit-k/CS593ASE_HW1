"""Search arXiv's public API and turn its Atom feed into result objects.

arXiv throttles automated clients aggressively (HTTP 429), so each search is
retried a couple of times with a growing pause before giving up.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import feedparser
import httpx

API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "AI-Research-Assistant/1.0 (local student project)"
MAX_RESULTS = 20
MAX_QUERY_WORDS = 12
# The orders arXiv can sort by itself; submittedDate is the date of a paper's first version.
SORT_BY = {"relevance": "relevance", "recent": "submittedDate"}
# arXiv leaves these common words (Lucene's standard English stop words) out of its
# title and abstract index, so a query that requires one of them misses most papers.
STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such"
    " that the their then there these they this to was will with".split()
)
TIMEOUT_SECONDS = 15
RETRY_DELAYS = (3, 6)  # seconds to wait before the 2nd and 3rd attempts
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Tests can set this to an httpx.MockTransport to fake arXiv without a network.
TRANSPORT: httpx.BaseTransport | None = None

# New-style IDs (2007 on): 1706.03762, 2401.12345. Old-style: hep-th/9901001, math.GT/0309136.
_ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})$")
_ID_PREFIX_RE = re.compile(r"^(?:https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/|arxiv:)", re.I)
_VERSION_RE = re.compile(r"v\d+$")
_NON_WORD_RE = re.compile(r"[^\w\-.']+")  # anything but letters, digits, _, -, . and '


class ArxivError(Exception):
    """arXiv couldn't be reached, kept refusing, or rejected the query.

    The message is written for the user and can be shown as is.
    """

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ArxivResult:
    arxiv_id: str  # versionless
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    url: str  # abstract page
    pdf_url: str
    published: date | None = None  # first version


def normalize_id(raw: str) -> str | None:
    """Return the versionless arXiv ID in `raw`, or None if it isn't one.

    Accepts plain IDs ("1706.03762v7"), "arXiv:" prefixes, and abs/pdf URLs.
    """
    candidate = _ID_PREFIX_RE.sub("", raw.strip())
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.I)
    candidate = _VERSION_RE.sub("", candidate)
    return candidate if _ARXIV_ID_RE.match(candidate) else None


def abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def build_params(
    query: str, *, sort: str = "relevance", max_results: int = MAX_RESULTS
) -> dict[str, str | int]:
    """Turn what the user typed into arXiv API query parameters.

    An arXiv ID is looked up directly. Otherwise every word except STOP_WORDS must
    match somewhere in the record (`all:` field, joined with AND), and results come
    in the `sort` order (a key of SORT_BY). Raises ValueError if nothing searchable
    is left.
    """
    arxiv_id = normalize_id(query)
    if arxiv_id:
        return {"id_list": arxiv_id, "max_results": 1}
    # arXiv indexes "Don't" as one word, so apostrophes stay inside words (curly ones
    # from copied titles are straightened first).
    words = [word.strip("-.'").lower() for word in _NON_WORD_RE.split(query.replace("’", "'"))]
    words = [word for word in words if word and word not in STOP_WORDS][:MAX_QUERY_WORDS]
    if not words:
        raise ValueError("Type a topic or some keywords to search.")
    return {
        "search_query": " AND ".join(f"all:{word}" for word in words),
        "sortBy": SORT_BY[sort],
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }


def search(
    query: str,
    *,
    sort: str = "relevance",
    max_results: int = MAX_RESULTS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ArxivResult]:
    """Search arXiv and return up to `max_results` results, best match or newest first."""
    params = build_params(query, sort=sort, max_results=max_results)
    with httpx.Client(
        transport=TRANSPORT,
        timeout=TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        feed_text = _get_with_retries(client, params, sleep)
    return parse_feed(feed_text)


def _get_with_retries(client: httpx.Client, params: dict, sleep: Callable[[float], None]) -> str:
    problem = "no response"
    for delay in (0, *RETRY_DELAYS):
        if delay:
            sleep(delay)
        try:
            response = client.get(API_URL, params=params)
        except httpx.TimeoutException:
            problem = "timed out"
            continue
        except httpx.TransportError:
            problem = "couldn't connect"
            continue
        if response.status_code == 200:
            return response.text
        if response.status_code in RETRY_STATUSES:
            problem = f"HTTP {response.status_code}"
            continue
        raise ArxivError(
            f"arXiv rejected the search (HTTP {response.status_code}). Try different keywords.", 502
        )
    attempts = 1 + len(RETRY_DELAYS)
    raise ArxivError(
        f"arXiv is busy right now — try again in a few seconds. ({problem} after {attempts} tries)"
    )


def parse_feed(feed_text: str) -> list[ArxivResult]:
    """Parse an arXiv Atom feed into results, skipping malformed entries."""
    feed = feedparser.parse(feed_text)
    if feed.bozo and not feed.entries:
        raise ArxivError("arXiv sent a response that isn't a search feed. Try again in a moment.", 502)

    results = []
    for entry in feed.entries:
        entry_id = entry.get("id", "")
        if "/api/errors" in entry_id:  # arXiv reports query problems as a fake entry
            reason = _clean(entry.get("summary", "")) or "unknown error"
            raise ArxivError(f"arXiv couldn't run that search: {reason}", 502)
        arxiv_id = normalize_id(entry_id)
        title = _clean(entry.get("title", ""))
        if not arxiv_id or not title:
            continue
        published = entry.get("published_parsed")
        results.append(
            ArxivResult(
                arxiv_id=arxiv_id,
                title=title,
                authors=[_clean(a.get("name", "")) for a in entry.get("authors", []) if a.get("name")],
                year=published.tm_year if published else None,
                abstract=_clean(entry.get("summary", "")),
                url=abs_url(arxiv_id),
                pdf_url=pdf_url(arxiv_id),
                published=date(*published[:3]) if published else None,
            )
        )
    return results


def _clean(text: str) -> str:
    """Collapse the line breaks and runs of spaces arXiv puts in titles and abstracts."""
    return " ".join(text.split())
