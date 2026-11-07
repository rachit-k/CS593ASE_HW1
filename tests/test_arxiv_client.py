"""arxiv_client.py: ID recognition, query building, Atom parsing, and retries (design section 7)."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from app import arxiv_client
from app.arxiv_client import ArxivError
from helpers import FakeService, atom_entry, atom_feed, recorded

# --- Recognizing arXiv IDs and links ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1706.03762", "1706.03762"),
        ("2401.12345", "2401.12345"),
        ("0704.0001", "0704.0001"),
        ("1706.03762v7", "1706.03762"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("hep-th/9901001v2", "hep-th/9901001"),
        ("math.GT/0309136", "math.GT/0309136"),
        ("arXiv:1706.03762", "1706.03762"),
        ("arxiv:1706.03762v2", "1706.03762"),
        ("  1706.03762\n", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("http://arxiv.org/abs/1706.03762v7", "1706.03762"),
        ("https://www.arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://export.arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v7.pdf", "1706.03762"),
        ("https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_ids_and_links_normalize_to_the_versionless_id(raw, expected):
    assert arxiv_client.normalize_id(raw) == expected


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("https://arxiv.org/abs/1706.03762/", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762?context=cs.CL", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762#S3", "1706.03762"),
        ("arxiv.org/abs/1706.03762", "1706.03762"),
        ("www.arxiv.org/pdf/1706.03762v7.pdf", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762v2/?context=cs.CL#S3", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v7.pdf?download=1", "1706.03762"),
        ("HTTPS://ARXIV.ORG/ABS/1706.03762", "1706.03762"),
        ("arxiv.org/abs/hep-th/9901001/", "hep-th/9901001"),
    ],
    ids=[
        "trailing-slash",
        "query-string",
        "fragment",
        "no-scheme",
        "no-scheme-www-pdf",
        "all-of-them",
        "pdf-with-query",
        "capitals",
        "old-style",
    ],
)
def test_links_as_browsers_copy_them_are_recognized(link, expected):
    # The README promises: "Pasting an arXiv ID or link goes straight to that paper."
    assert arxiv_client.normalize_id(link) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "attention is all you need",
        "1706",
        "1706.123",
        "17060.3762",
        "1706.123456",
        "hep-th/990100",
        "https://evil.example.com/abs/1706.03762",
        "https://arxiv.org.evil.example/abs/1706.03762",
        "https://evil.example.com/abs/1706.03762?x=1",
        "https://evil.example.com/arxiv.org/abs/1706.03762",
        "evil.example.com/arxiv.org/abs/1706.03762",
        "notarxiv.org/abs/1706.03762",
        "arxiv.org.evil.example/abs/1706.03762",
        "https://arxiv.org/list/1706.03762",
        "arxiv.org/list/1706.03762",
        "https://arxiv.org/abs/not-an-id",
        "https://arxiv.org/abs/1706.03762/extra",
        "https://arxiv.org/abs/1706.03762\n?x=1",
        "1706.03762/",
        "1706.03762?x=1",
        "../../etc/passwd",
        "1706.03762 and more words",
    ],
)
def test_anything_else_is_not_an_id(raw):
    assert arxiv_client.normalize_id(raw) is None


def test_links_are_built_from_the_id():
    assert arxiv_client.abs_url("hep-th/9901001") == "https://arxiv.org/abs/hep-th/9901001"
    assert arxiv_client.pdf_url("1706.03762") == "https://arxiv.org/pdf/1706.03762"


# --- Building the query -------------------------------------------------------------


def query_for(text: str) -> str:
    return arxiv_client.build_params(text)["search_query"]


def test_every_word_is_required_and_lowercased():
    assert query_for("Code Review Automation") == "all:code AND all:review AND all:automation"


def test_a_title_starting_with_a_stop_word_keeps_only_the_words_arxiv_indexes():
    # Regression: requiring all:a hid this paper, because arXiv doesn't index "a" in titles.
    title = "A HAT Trick: Automatically Verifying Representation Invariants Using Symbolic Finite Automata"
    assert query_for(title) == (
        "all:hat AND all:trick AND all:automatically AND all:verifying AND all:representation"
        " AND all:invariants AND all:using AND all:symbolic AND all:finite AND all:automata"
    )


def test_all_33_stop_words_are_dropped_but_ordinary_short_words_are_kept():
    assert len(arxiv_client.STOP_WORDS) == 33
    assert query_for(" ".join(sorted(arxiv_client.STOP_WORDS)) + " transformers") == "all:transformers"
    assert query_for("Attention Is All You Need") == "all:attention AND all:all AND all:you AND all:need"


@pytest.mark.parametrize("text", ["", "   ", "the of and", "!!! ??? ()", "- . '"])
def test_nothing_searchable_left_is_an_error(text):
    with pytest.raises(ValueError, match="Type a topic"):
        arxiv_client.build_params(text)


@pytest.mark.parametrize(
    "text",
    ["Don't Stop Pretraining", "Don’t Stop Pretraining", "'Don't' Stop Pretraining"],
    ids=["straight", "curly", "quoted-word"],
)
def test_apostrophes_stay_inside_words(text):
    # Regression: arXiv indexes "don't" as one word, so splitting it found nothing.
    assert query_for(text) == "all:don't AND all:stop AND all:pretraining"


def test_query_syntax_characters_are_stripped():
    assert query_for('"graph" (neural) networks: [survey]') == (
        "all:graph AND all:neural AND all:networks AND all:survey"
    )


def test_a_user_cannot_inject_arxiv_query_fields_or_operators():
    assert query_for("au:smith OR ti:testing ANDNOT cat:cs.SE") == (
        "all:au AND all:smith AND all:ti AND all:testing AND all:andnot AND all:cat AND all:cs.se"
    )


def test_hyphens_and_dots_stay_inside_words_but_not_at_the_edges():
    assert query_for("pre-training -bert- gpt-4. .net") == "all:pre-training AND all:bert AND all:gpt-4 AND all:net"


def test_non_english_letters_are_kept():
    assert query_for("Schrödinger équation") == "all:schrödinger AND all:équation"


def test_long_queries_keep_the_first_twelve_real_words():
    words = [f"word{n}" for n in range(1, 16)]
    query = query_for("the " + " of ".join(words))
    assert query == " AND ".join(f"all:word{n}" for n in range(1, 13))


@pytest.mark.parametrize(("sort", "sort_by"), [("relevance", "relevance"), ("recent", "submittedDate")])
def test_sort_order_maps_to_arxivs_parameter(sort, sort_by):
    params = arxiv_client.build_params("testing", sort=sort)
    assert params["sortBy"] == sort_by
    assert params["sortOrder"] == "descending"


def test_default_is_twenty_results_by_relevance():
    params = arxiv_client.build_params("testing")
    assert params["max_results"] == 20
    assert params["sortBy"] == "relevance"
    assert params["start"] == 0


def test_a_pasted_id_is_looked_up_directly_in_any_sort_order():
    expected = {"id_list": "1706.03762", "max_results": 1}
    assert arxiv_client.build_params("https://arxiv.org/abs/1706.03762v7") == expected
    assert arxiv_client.build_params("1706.03762", sort="recent", max_results=100) == expected


# --- Reading the feed ---------------------------------------------------------------


def test_a_real_feed_gives_every_field_the_app_shows():
    results = {result.arxiv_id: result for result in arxiv_client.parse_feed(recorded("arxiv_three_papers.xml"))}
    assert set(results) == {"1706.03762", "2404.01484", "hep-th/9901001"}

    attention = results["1706.03762"]
    assert attention.title == "Attention Is All You Need"
    assert attention.authors == [
        "Ashish Vaswani",
        "Noam Shazeer",
        "Niki Parmar",
        "Jakob Uszkoreit",
        "Llion Jones",
        "Aidan N. Gomez",
        "Lukasz Kaiser",
        "Illia Polosukhin",
    ]
    assert attention.abstract.startswith("The dominant sequence transduction models")
    assert attention.abstract.endswith("both with large and limited training data.")
    assert attention.url == "https://arxiv.org/abs/1706.03762"
    assert attention.pdf_url == "https://arxiv.org/pdf/1706.03762"


def test_year_and_date_come_from_the_first_version_not_the_latest_update():
    results = {result.arxiv_id: result for result in arxiv_client.parse_feed(recorded("arxiv_three_papers.xml"))}
    # This paper's feed entry says: published 2017-06-12, updated 2023-08-02 (v7).
    assert results["1706.03762"].year == 2017
    assert results["1706.03762"].published == date(2017, 6, 12)


def test_old_style_ids_and_stray_spaces_in_a_real_feed():
    results = {result.arxiv_id: result for result in arxiv_client.parse_feed(recorded("arxiv_three_papers.xml"))}
    old_paper = results["hep-th/9901001"]
    assert old_paper.url == "https://arxiv.org/abs/hep-th/9901001"
    assert old_paper.abstract.startswith("We explicitly give")


def test_line_breaks_in_a_real_feed_are_collapsed():
    results = arxiv_client.parse_feed(recorded("arxiv_keyword_search.xml"))
    assert len(results) == 2
    # The recording wraps its first abstract over several lines.
    assert "\n" in recorded("arxiv_keyword_search.xml").split("<summary>")[1].split("</summary>")[0]
    for result in results:
        assert "\n" not in result.title
        assert "\n" not in result.abstract
        assert "  " not in result.abstract


def test_a_wrapped_title_is_collapsed():
    feed = atom_feed(atom_entry(title="A Very Long Title\n  Wrapped Over   Two Lines"))
    assert arxiv_client.parse_feed(feed)[0].title == "A Very Long Title Wrapped Over Two Lines"


@pytest.mark.parametrize("recording", ["arxiv_no_results.xml", "arxiv_unknown_id.xml"])
def test_real_feeds_with_no_entries_mean_no_results(recording):
    assert arxiv_client.parse_feed(recorded(recording)) == []


def test_a_real_error_entry_becomes_an_error_with_arxivs_reason():
    with pytest.raises(ArxivError, match="incorrect id format for notanid") as caught:
        arxiv_client.parse_feed(recorded("arxiv_bad_id.xml"))
    assert caught.value.status_code == 502


def test_entries_without_a_title_or_a_usable_id_are_skipped():
    feed = atom_feed(
        atom_entry("2401.00001", title=""),
        "<entry><id>http://example.com/not-arxiv</id><title>Not arXiv</title></entry>",
        atom_entry("2401.00002", title="Good Paper"),
    )
    assert [result.title for result in arxiv_client.parse_feed(feed)] == ["Good Paper"]


def test_a_missing_date_leaves_the_year_unknown():
    result = arxiv_client.parse_feed(atom_feed(atom_entry(published=None)))[0]
    assert result.year is None
    assert result.published is None


@pytest.mark.parametrize(
    "body",
    [
        "Service Unavailable",
        "<html><body><h1>Maintenance<br></h1></body>",
        "<html><body><h1>Maintenance</h1></body></html>",
        "",
    ],
    ids=["plain-text", "malformed-html", "well-formed-html", "empty"],
)
def test_a_response_that_is_not_a_feed_is_an_error_not_an_empty_result(body):
    with pytest.raises(ArxivError) as caught:
        arxiv_client.parse_feed(body)
    assert caught.value.status_code == 502


def test_response_text_is_never_opened_as_a_file(tmp_path):
    feed_file = tmp_path / "feed.xml"
    feed_file.write_text(atom_feed(atom_entry()))
    with pytest.raises(ArxivError):
        arxiv_client.parse_feed(str(feed_file))


def test_response_text_is_never_fetched_as_a_url():
    # If this were fetched, the autouse guard against real connections would fail the test.
    with pytest.raises(ArxivError):
        arxiv_client.parse_feed("https://example.com/feed.xml")


# --- The request and its retries ----------------------------------------------------


def test_search_sends_the_query_to_arxiv_with_a_descriptive_user_agent(arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry()))
    results = arxiv_client.search("Graph Testing", sort="recent", max_results=5)

    assert [result.arxiv_id for result in results] == ["2401.12345"]
    assert len(arxiv.requests) == 1
    request = arxiv.requests[0]
    assert str(request.url).startswith("https://export.arxiv.org/api/query?")
    assert request.url.params["search_query"] == "all:graph AND all:testing"
    assert request.url.params["sortBy"] == "submittedDate"
    assert request.url.params["max_results"] == "5"
    assert request.headers["User-Agent"].startswith("AI-Research-Assistant/")  # arXiv asks clients to identify themselves
    assert request.extensions["timeout"]["read"] == 15


def test_retries_wait_3_then_6_seconds(monkeypatch):
    fake = FakeService().reply(429).reply(503).reply(200, text=atom_feed(atom_entry()))
    monkeypatch.setattr(arxiv_client, "TRANSPORT", httpx.MockTransport(fake))
    waits: list[float] = []

    results = arxiv_client.search("testing", sleep=waits.append)

    assert waits == [3, 6]
    assert len(fake.requests) == 3
    assert len(results) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_busy_responses_are_retried(arxiv, status):
    arxiv.reply(status).reply(200, text=atom_feed(atom_entry()))
    assert len(arxiv_client.search("testing")) == 1
    assert len(arxiv.requests) == 2


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("slow"), httpx.ConnectError("no route")], ids=["timeout", "connection-error"]
)
def test_network_failures_are_retried(arxiv, error):
    arxiv.fail(error).reply(200, text=atom_feed(atom_entry()))
    assert len(arxiv_client.search("testing")) == 1
    assert len(arxiv.requests) == 2


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (lambda fake: fake.reply(429), "HTTP 429"),
        (lambda fake: fake.fail(httpx.ReadTimeout("slow")), "timed out"),
        (lambda fake: fake.fail(httpx.ConnectError("no route")), "couldn't connect"),
    ],
    ids=["rate-limited", "timeout", "connection-error"],
)
def test_after_three_failed_attempts_arxiv_is_reported_busy(arxiv, setup, reason):
    setup(arxiv)
    with pytest.raises(ArxivError) as caught:
        arxiv_client.search("testing")
    assert len(arxiv.requests) == 3
    assert caught.value.status_code == 503
    assert "arXiv is busy" in str(caught.value)
    assert reason in str(caught.value)


def test_a_rejected_query_is_not_retried(arxiv):
    arxiv.reply(400, text=recorded("arxiv_bad_id.xml"))
    with pytest.raises(ArxivError, match="rejected the search") as caught:
        arxiv_client.search("testing")
    assert len(arxiv.requests) == 1
    assert caught.value.status_code == 502


# --- Rare shapes of a feed ---------------------------------------------------------------


def test_an_error_entry_with_no_reason_still_says_something():
    feed = atom_feed("<entry><id>https://arxiv.org/api/errors#unknown</id><title>Error</title><summary></summary></entry>")
    with pytest.raises(ArxivError, match="unknown error") as caught:
        arxiv_client.parse_feed(feed)
    assert caught.value.status_code == 502


def test_an_author_with_no_name_is_skipped():
    entry = atom_entry(authors=["Alice Smith"]).replace(
        "<author>", "<author><email>nobody@example.com</email></author><author>", 1
    )
    assert arxiv_client.parse_feed(atom_feed(entry))[0].authors == ["Alice Smith"]


# --- Found by mutation testing: behavior the tests above ran but never checked --------------------


def test_an_uppercase_pdf_ending_is_dropped_too():
    assert arxiv_client.normalize_id("https://arxiv.org/pdf/2401.12345.PDF") == "2401.12345"


def test_a_redirect_from_the_arxiv_api_is_followed(arxiv):
    arxiv.reply(301, headers={"Location": "https://export.arxiv.org/api/moved"})
    arxiv.reply(200, text=atom_feed(atom_entry(title="After the redirect")))
    assert [result.title for result in arxiv_client.search("testing")] == ["After the redirect"]
    assert str(arxiv.requests[1].url) == "https://export.arxiv.org/api/moved"


@pytest.mark.parametrize(
    ("fail", "problem"),
    [
        (lambda service: service.fail(httpx.ReadTimeout("slow")), "timed out"),
        (lambda service: service.fail(httpx.ConnectError("no route")), "couldn't connect"),
        (lambda service: service.reply(429), "HTTP 429"),
    ],
    ids=["timeout", "connection", "throttled"],
)
def test_the_busy_message_says_what_went_wrong(arxiv, fail, problem):
    fail(arxiv)
    with pytest.raises(arxiv_client.ArxivError) as caught:
        arxiv_client.search("testing")
    assert str(caught.value) == f"arXiv is busy right now — try again in a few seconds. ({problem} after 3 tries)"


def test_a_rejected_search_has_the_whole_message(arxiv):
    arxiv.reply(400)
    with pytest.raises(arxiv_client.ArxivError) as caught:
        arxiv_client.search("testing")
    assert str(caught.value) == "arXiv rejected the search (HTTP 400). Try different keywords."
    assert caught.value.status_code == 502


@pytest.mark.parametrize(
    ("summary", "reason"),
    [("<summary>bad query</summary>", "bad query"), ("<summary></summary>", "unknown error"), ("", "unknown error")],
    ids=["with-a-reason", "empty-reason", "no-reason"],
)
def test_an_error_entry_passes_on_arxivs_reason(summary, reason):
    feed = atom_feed(f"<entry><id>http://arxiv.org/api/errors#bad</id><title>Error</title>{summary}</entry>")
    with pytest.raises(arxiv_client.ArxivError) as caught:
        arxiv_client.parse_feed(feed)
    assert str(caught.value) == f"arXiv couldn't run that search: {reason}"
    assert caught.value.status_code == 502


def test_an_entry_with_no_title_element_is_skipped():
    assert arxiv_client.parse_feed(atom_feed("<entry><id>http://arxiv.org/abs/2401.00001v1</id></entry>")) == []


def test_an_entry_with_no_summary_element_has_an_empty_abstract():
    feed = atom_feed("<entry><id>http://arxiv.org/abs/2401.00001v1</id><title>Only a title</title></entry>")
    assert arxiv_client.parse_feed(feed)[0].abstract == ""
