"""Citation counts for arXiv papers, from the Semantic Scholar API.

arXiv has no citation data, so a "most cited" search takes arXiv's best matches and
sorts them by the counts Semantic Scholar reports. Its batch endpoint looks up many
papers by arXiv ID in one request and needs no API key.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from app.arxiv_client import USER_AGENT, ArxivResult

API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# How many of arXiv's best matches a "most cited" search ranks (the batch endpoint takes
# up to 500). A bigger pool reaches further down arXiv's ranking, but it also pulls in
# heavily cited papers that only mention the search words in passing.
POOL_SIZE = 100
TIMEOUT_SECONDS = 15
RETRY_DELAYS = (1, 3)  # seconds to wait before the 2nd and 3rd attempts
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Tests can set this to an httpx.MockTransport to fake Semantic Scholar without a network.
TRANSPORT: httpx.BaseTransport | None = None


class CitationError(Exception):
    """Semantic Scholar couldn't be reached, kept refusing, or sent something unexpected.

    The message is written for the user and can be shown as is.
    """

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def most_cited(results: list[ArxivResult], limit: int) -> tuple[list[ArxivResult], dict[str, int]]:
    """The `limit` most-cited papers in `results`, plus the citation counts by arXiv ID.

    Papers Semantic Scholar doesn't know go last; ties keep their order in `results`.
    """
    counts = citation_counts([r.arxiv_id for r in results])
    ranked = sorted(results, key=lambda r: counts.get(r.arxiv_id, -1), reverse=True)
    return ranked[:limit], counts


def citation_counts(
    arxiv_ids: list[str], *, sleep: Callable[[float], None] = time.sleep
) -> dict[str, int]:
    """Citation count for each arXiv ID Semantic Scholar knows. Unknown IDs are left out."""
    if not arxiv_ids:
        return {}
    with httpx.Client(
        transport=TRANSPORT,
        timeout=TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        papers = _post_with_retries(client, arxiv_ids, sleep)
    return {
        arxiv_id: paper["citationCount"]
        for arxiv_id, paper in zip(arxiv_ids, papers)
        if isinstance(paper, dict) and isinstance(paper.get("citationCount"), int)
    }


def _post_with_retries(
    client: httpx.Client, arxiv_ids: list[str], sleep: Callable[[float], None]
) -> list:
    body = {"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]}
    problem = "no response"
    for delay in (0, *RETRY_DELAYS):
        if delay:
            sleep(delay)
        try:
            response = client.post(API_URL, params={"fields": "citationCount"}, json=body)
        except httpx.TimeoutException:
            problem = "timed out"
            continue
        except httpx.HTTPError:
            problem = "couldn't connect"
            continue
        if response.status_code == 200:
            try:
                papers = response.json()
            except ValueError:
                papers = None
            # One entry per ID, in the order sent; null for papers it doesn't know.
            if isinstance(papers, list) and len(papers) == len(arxiv_ids):
                return papers
            raise CitationError(
                "Semantic Scholar sent citation data the app can't read. Sort by arXiv rank instead.", 502
            )
        if response.status_code in RETRY_STATUSES:
            problem = f"HTTP {response.status_code}"
            continue
        raise CitationError(
            f"Semantic Scholar rejected the citation lookup (HTTP {response.status_code}). "
            "Sort by arXiv rank instead.",
            502,
        )
    attempts = 1 + len(RETRY_DELAYS)
    raise CitationError(
        "Semantic Scholar is busy, so citation counts aren't available right now. Try again in a "
        f"few seconds, or sort by arXiv rank. ({problem} after {attempts} tries)"
    )
