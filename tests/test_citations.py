"""citations.py: Semantic Scholar lookups and "most cited" ranking (design section 7)."""

from __future__ import annotations

import json

import httpx
import pytest

from app import arxiv_client, citations
from app.arxiv_client import ArxivResult
from app.citations import CitationError
from helpers import FakeService, recorded


def result(arxiv_id: str) -> ArxivResult:
    return ArxivResult(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors=[],
        year=2024,
        abstract="",
        url=arxiv_client.abs_url(arxiv_id),
        pdf_url=arxiv_client.pdf_url(arxiv_id),
    )


def counts_reply(*counts: int | None) -> list[dict | None]:
    return [None if count is None else {"paperId": "x", "citationCount": count} for count in counts]


# --- Looking up counts --------------------------------------------------------------


def test_one_batch_request_covers_every_id(scholar):
    scholar.reply(200, json=counts_reply(1, 2))
    citations.citation_counts(["2401.00001", "hep-th/9901001"])

    assert len(scholar.requests) == 1
    request = scholar.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.semanticscholar.org/graph/v1/paper/batch?fields=citationCount"
    assert json.loads(request.content) == {"ids": ["ARXIV:2401.00001", "ARXIV:hep-th/9901001"]}
    assert request.headers["User-Agent"].startswith("AI-Research-Assistant/")


def test_a_real_response_maps_counts_to_ids_and_leaves_out_unknown_papers(scholar):
    scholar.reply(200, text=recorded("semantic_scholar_batch.json"))
    counts = citations.citation_counts(["1706.03762", "9999.99999", "hep-th/9901001"])
    assert counts == {"1706.03762": 201731, "hep-th/9901001": 8}


def test_an_entry_without_a_count_is_left_out(scholar):
    scholar.reply(200, json=[{"paperId": "x", "citationCount": None}, {"paperId": "y"}, counts_reply(0)[0]])
    assert citations.citation_counts(["a", "b", "c"]) == {"c": 0}


def test_no_ids_means_no_request(scholar):
    assert citations.citation_counts([]) == {}
    assert scholar.requests == []


def test_retries_wait_1_then_3_seconds(monkeypatch):
    fake = FakeService().reply(429).reply(503).reply(200, json=counts_reply(7))
    monkeypatch.setattr(citations, "TRANSPORT", httpx.MockTransport(fake))
    waits: list[float] = []

    assert citations.citation_counts(["a"], sleep=waits.append) == {"a": 7}
    assert waits == [1, 3]
    assert len(fake.requests) == 3


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_busy_responses_are_retried(scholar, status):
    scholar.reply(status).reply(200, json=counts_reply(7))
    assert citations.citation_counts(["a"]) == {"a": 7}
    assert len(scholar.requests) == 2


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("slow"), httpx.ConnectError("no route")], ids=["timeout", "connection-error"]
)
def test_network_failures_are_retried(scholar, error):
    scholar.fail(error).reply(200, json=counts_reply(7))
    assert citations.citation_counts(["a"]) == {"a": 7}


def test_the_real_rate_limit_response_ends_in_a_busy_error_after_three_tries(scholar):
    scholar.reply(429, text=recorded("semantic_scholar_429.json"))
    with pytest.raises(CitationError, match="Semantic Scholar is busy") as caught:
        citations.citation_counts(["a"])
    assert len(scholar.requests) == 3
    assert caught.value.status_code == 503
    assert "sort by arXiv rank" in str(caught.value)


def test_timeouts_end_in_a_busy_error(scholar):
    scholar.fail(httpx.ReadTimeout("slow"))
    with pytest.raises(CitationError, match="timed out after 3 tries") as caught:
        citations.citation_counts(["a"])
    assert caught.value.status_code == 503


def test_a_rejected_lookup_is_not_retried(scholar):
    scholar.reply(400, json={"error": "bad ids"})
    with pytest.raises(CitationError, match="rejected the citation lookup") as caught:
        citations.citation_counts(["a"])
    assert len(scholar.requests) == 1
    assert caught.value.status_code == 502


@pytest.mark.parametrize(
    "options",
    [{"text": "<html>not json</html>"}, {"json": {"not": "a list"}}, {"json": counts_reply(1, 2)}],
    ids=["not-json", "not-a-list", "wrong-length"],
)
def test_an_unreadable_response_is_an_error(scholar, options):
    scholar.reply(200, **options)
    with pytest.raises(CitationError, match="can't read") as caught:
        citations.citation_counts(["only-one-id"])
    assert caught.value.status_code == 502


# --- Ranking ------------------------------------------------------------------------


def test_most_cited_ranks_by_count_and_returns_the_counts(scholar):
    scholar.reply(200, json=counts_reply(5, 50, 20))
    ranked, counts = citations.most_cited([result("a"), result("b"), result("c")], 10)
    assert [paper.arxiv_id for paper in ranked] == ["b", "c", "a"]
    assert counts == {"a": 5, "b": 50, "c": 20}


def test_most_cited_keeps_only_the_limit(scholar):
    scholar.reply(200, json=counts_reply(1, 2, 3, 4))
    ranked, counts = citations.most_cited([result(name) for name in "abcd"], 2)
    assert [paper.arxiv_id for paper in ranked] == ["d", "c"]
    assert len(counts) == 4


def test_unknown_papers_go_last_even_behind_zero_citations(scholar):
    scholar.reply(200, json=counts_reply(None, 0, 3))
    ranked, _counts = citations.most_cited([result("unknown"), result("zero"), result("three")], 10)
    assert [paper.arxiv_id for paper in ranked] == ["three", "zero", "unknown"]


def test_ties_keep_arxivs_order(scholar):
    scholar.reply(200, json=counts_reply(9, 9, 9, None, None))
    ranked, _counts = citations.most_cited([result(name) for name in "abcde"], 10)
    assert [paper.arxiv_id for paper in ranked] == ["a", "b", "c", "d", "e"]


def test_no_results_means_nothing_to_rank_and_no_request(scholar):
    assert citations.most_cited([], 20) == ([], {})
    assert scholar.requests == []


# --- Found by mutation testing: the messages people actually read -----------------------------------


@pytest.mark.parametrize(
    ("fail", "problem"),
    [
        (lambda service: service.fail(httpx.ReadTimeout("slow")), "timed out"),
        (lambda service: service.fail(httpx.ConnectError("no route")), "couldn't connect"),
        (lambda service: service.reply(429), "HTTP 429"),
    ],
    ids=["timeout", "connection", "throttled"],
)
def test_the_busy_message_says_what_went_wrong(scholar, fail, problem):
    fail(scholar)
    with pytest.raises(citations.CitationError) as caught:
        citations.most_cited([result("2401.00001")], 1)
    assert str(caught.value) == (
        "Semantic Scholar is busy, so citation counts aren't available right now. Try again in a "
        f"few seconds, or sort by arXiv rank. ({problem} after 3 tries)"
    )


def test_a_rejected_lookup_has_the_whole_message(scholar):
    scholar.reply(400)
    with pytest.raises(citations.CitationError) as caught:
        citations.most_cited([result("2401.00001")], 1)
    assert str(caught.value) == "Semantic Scholar rejected the citation lookup (HTTP 400). Sort by arXiv rank instead."
    assert caught.value.status_code == 502
