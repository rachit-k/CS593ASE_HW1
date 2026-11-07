"""GET /: the search page, its three sort orders, and how problems show up (design sections 4, 7, 9)."""

from __future__ import annotations

import json

import httpx
import pytest

from helpers import HTMX, atom_entry, atom_feed, parse_html, recorded


def numbered_feed(count: int) -> str:
    return atom_feed(*(atom_entry(f"2401.{n:05d}", title=f"Paper {n}") for n in range(1, count + 1)))


def titles(html: str) -> list[str]:
    return [title.text for title in parse_html(html).find_all("h2", class_="entry-title")]


def counts_reply(*counts: int | None) -> list[dict | None]:
    return [None if count is None else {"paperId": "x", "citationCount": count} for count in counts]


# --- Before searching ---------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "   "}], ids=["no-query", "empty", "spaces"])
def test_no_query_shows_the_search_box_without_calling_arxiv(client, params):
    response = client.get("/", params=params)
    assert response.status_code == 200
    page = parse_html(response.text)
    assert "required" in page.find("input", name="q").attrs
    assert page.find(id="results").find(class_="entry") is None


def test_a_blank_htmx_search_shows_nothing_and_calls_no_one(client):
    # The isolate fixture fails the test if this reaches arXiv.
    response = client.get("/", params={"q": "  "}, headers=HTMX)
    page = parse_html(response.text)
    assert response.status_code == 200
    assert page.find(class_="entry") is None
    assert page.find(class_="empty-state") is None  # not "No papers matched": nothing was searched


def test_the_sort_menu_offers_the_three_orders(client):
    options = parse_html(client.get("/").text).find("select", name="sort").find_all("option")
    assert [option.attrs["value"] for option in options] == ["relevance", "recent", "citations"]
    assert [option.text for option in options] == ["Sort: arXiv rank", "Sort: most recent", "Sort: most cited"]
    assert [option.attrs["value"] for option in options if "selected" in option.attrs] == ["relevance"]


def test_the_search_form_is_wired_to_rerun_when_the_sort_changes(client):
    form = parse_html(client.get("/").text).find("form", class_="search-form")
    assert form.attrs["hx-get"] == "/"
    assert form.attrs["hx-target"] == "#results"
    assert form.attrs["hx-push-url"] == "true"
    assert "change from:find select" in form.attrs["hx-trigger"]


# --- Results ------------------------------------------------------------------------


def test_a_real_arxiv_response_renders_as_result_cards(client, arxiv):
    arxiv.reply(200, text=recorded("arxiv_three_papers.xml"))

    response = client.get("/", params={"q": "attention"})

    assert response.status_code == 200
    page = parse_html(response.text)
    assert page.find(class_="results-count").text == "3 papers for “attention”, best match first"
    card = page.find_all("li", class_="entry")[0]
    link = card.find("h2").find("a")
    assert link.text == "Attention Is All You Need"
    assert link.attrs["href"] == "https://arxiv.org/abs/1706.03762"
    assert card.find(class_="authors").text == "Ashish Vaswani, Noam Shazeer, Niki Parmar, and 5 more"
    assert card.find(class_="entry-margin").text == "2017"
    assert card.find(class_="abstract").text.startswith("The dominant sequence transduction models")
    assert card.find("form").find("button", type="submit") is not None


def test_results_keep_arxivs_order_and_a_full_page_of_twenty_is_requested(client, arxiv):
    arxiv.reply(200, text=numbered_feed(20))

    response = client.get("/", params={"q": "testing"})

    assert titles(response.text) == [f"Paper {n}" for n in range(1, 21)]
    params = arxiv.requests[0].url.params
    assert params["max_results"] == "20"
    assert params["sortBy"] == "relevance"
    assert params["search_query"] == "all:testing"
    assert len(arxiv.requests) == 1


def test_a_card_carries_everything_needed_to_save_the_paper(client, arxiv):
    arxiv.reply(200, text=recorded("arxiv_three_papers.xml"))
    card = parse_html(client.get("/", params={"q": "attention"}).text).find("li", class_="entry")
    form = card.find("form")
    assert form.attrs["hx-post"] == "/library"
    fields = {field.attrs["name"]: field.attrs["value"] for field in form.find_all("input", type="hidden")}
    assert fields["arxiv_id"] == "1706.03762"
    assert fields["title"] == "Attention Is All You Need"
    assert fields["year"] == "2017"
    assert json.loads(fields["authors"])[0] == "Ashish Vaswani"
    assert fields["abstract"].startswith("The dominant sequence transduction models")


def test_one_result_is_described_in_the_singular(client, arxiv):
    arxiv.reply(200, text=numbered_feed(1))
    response = client.get("/", params={"q": "testing"})
    assert parse_html(response.text).find(class_="results-count").text == "1 paper for “testing”, best match first"


def test_a_long_abstract_collapses_and_a_short_one_does_not(client, arxiv):
    arxiv.reply(
        200,
        text=atom_feed(atom_entry("2401.00001", abstract="Long. " * 100), atom_entry("2401.00002", abstract="Short.")),
    )
    long_card, short_card = parse_html(client.get("/", params={"q": "testing"}).text).find_all("li", class_="entry")
    assert "clamped" in long_card.find("p", id="abstract-1").attrs["class"]
    assert long_card.find("button", class_="link-button").attrs["aria-controls"] == "abstract-1"
    assert "clamped" not in short_card.find("p", id="abstract-2").attrs["class"]
    assert short_card.find("button", class_="link-button") is None


def test_a_paper_with_no_authors_or_year_still_renders(client, arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry(authors=[], published=None)))
    card = parse_html(client.get("/", params={"q": "testing"}).text).find("li", class_="entry")
    assert card.find(class_="authors").text == "Authors unknown"
    assert card.find(class_="entry-margin").text == "—"


def test_a_paper_already_in_the_library_links_to_it_instead_of_offering_save(client, arxiv, add_paper):
    paper_id = add_paper(arxiv_id="2401.00001")
    arxiv.reply(200, text=numbered_feed(2))

    saved, unsaved = parse_html(client.get("/", params={"q": "testing"}).text).find_all("li", class_="entry")

    assert saved.find(class_="saved-note").text == "In your library"
    assert saved.find("a", class_="button").attrs["href"] == f"/papers/{paper_id}"
    assert saved.find("form") is None
    assert unsaved.find("form") is not None
    assert unsaved.find(class_="saved-note") is None


def test_no_matches_says_so(client, arxiv):
    arxiv.reply(200, text=recorded("arxiv_no_results.xml"))
    response = client.get("/", params={"q": "zzqqxx"})
    assert response.status_code == 200
    assert "No papers matched “zzqqxx”" in parse_html(response.text).find(class_="empty-state").text


# --- Looking up an ID ---------------------------------------------------------------


@pytest.mark.parametrize(
    "link",
    [
        "https://arxiv.org/abs/2404.01484v2",
        "https://arxiv.org/abs/2404.01484/",
        "https://arxiv.org/abs/2404.01484?context=cs.PL",
        "arxiv.org/abs/2404.01484",
    ],
    ids=["version", "trailing-slash", "query-string", "no-scheme"],
)
def test_a_pasted_arxiv_link_looks_up_that_paper_directly(client, arxiv, link):
    arxiv.reply(200, text=atom_feed(atom_entry("2404.01484", title="A HAT Trick")))

    response = client.get("/", params={"q": link})

    assert titles(response.text) == ["A HAT Trick"]
    params = arxiv.requests[0].url.params
    assert params["id_list"] == "2404.01484"
    assert "search_query" not in params


def test_an_id_that_does_not_exist_shows_no_results(client, arxiv):
    arxiv.reply(200, text=recorded("arxiv_unknown_id.xml"))
    response = client.get("/", params={"q": "2401.99999"})
    assert response.status_code == 200
    assert parse_html(response.text).find(class_="empty-state") is not None


# --- Partial and full responses -----------------------------------------------------


def test_htmx_requests_get_only_the_results(client, arxiv):
    arxiv.reply(200, text=numbered_feed(1))
    response = client.get("/", params={"q": "testing"}, headers=HTMX)
    assert "<html" not in response.text
    assert parse_html(response.text).find("form", class_="search-form") is None
    assert titles(response.text) == ["Paper 1"]


def test_normal_requests_get_the_full_page_with_the_query_filled_in(client, arxiv):
    arxiv.reply(200, text=numbered_feed(1))
    response = client.get("/", params={"q": "testing", "sort": "recent"})
    page = parse_html(response.text)
    assert "<html" in response.text
    assert page.find("input", name="q").attrs["value"] == "testing"
    assert page.find("option", value="recent").attrs.keys() >= {"selected"}
    assert page.find("title").text.startswith("testing |")
    assert titles(response.text) == ["Paper 1"]


def test_a_back_button_restore_gets_the_full_page(client, arxiv):
    arxiv.reply(200, text=numbered_feed(1))
    headers = HTMX | {"HX-History-Restore-Request": "true"}
    assert "<html" in client.get("/", params={"q": "testing"}, headers=headers).text


# --- Sort: most recent --------------------------------------------------------------


def test_most_recent_asks_arxiv_for_newest_first_and_shows_dates(client, arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry(published="2026-03-05T10:00:00Z")))

    response = client.get("/", params={"q": "testing", "sort": "recent"})

    assert arxiv.requests[0].url.params["sortBy"] == "submittedDate"
    page = parse_html(response.text)
    assert page.find(class_="results-count").text == "1 paper for “testing”, newest first"
    assert page.find("li", class_="entry").find(class_="facts").text == "Submitted 5 Mar 2026"


# --- Sort: most cited ---------------------------------------------------------------


def test_most_cited_ranks_arxivs_100_best_matches_and_shows_the_top_20(client, arxiv, scholar):
    arxiv.reply(200, text=numbered_feed(100))
    scholar.reply(200, json=counts_reply(*range(1, 101)))  # the last paper is the most cited

    response = client.get("/", params={"q": "testing", "sort": "citations"})

    assert len(arxiv.requests) == 1
    assert arxiv.requests[0].url.params["max_results"] == "100"
    assert arxiv.requests[0].url.params["sortBy"] == "relevance"
    assert len(scholar.requests) == 1
    assert len(json.loads(scholar.requests[0].content)["ids"]) == 100
    assert titles(response.text) == [f"Paper {n}" for n in range(100, 80, -1)]
    page = parse_html(response.text)
    assert page.find("li", class_="entry").find(class_="facts").text == "100 citations"
    summary = page.find(class_="results-count")
    assert "20 of arXiv's 100 best matches for “testing”, most cited first" in summary.text
    assert summary.find("a").attrs["href"] == "https://www.semanticscholar.org"


def test_fewer_matches_than_the_pool_are_described_plainly(client, arxiv, scholar):
    arxiv.reply(200, text=numbered_feed(3))
    scholar.reply(200, json=counts_reply(1, 2, 3))
    summary = parse_html(client.get("/", params={"q": "testing", "sort": "citations"}).text).find(class_="results-count")
    assert "best matches" not in summary.text
    assert "3 papers for “testing”, most cited first" in summary.text


def test_citation_counts_are_readable(client, arxiv, scholar):
    arxiv.reply(200, text=numbered_feed(4))
    scholar.reply(200, json=counts_reply(196154, 1, 0, None))

    cards = parse_html(client.get("/", params={"q": "testing", "sort": "citations"}).text).find_all("li", class_="entry")

    assert [card.find(class_="facts").text for card in cards] == [
        "196,154 citations",
        "1 citation",
        "0 citations",
        "Citation count unknown",
    ]


def test_most_cited_with_no_matches_does_not_call_semantic_scholar(client, arxiv, scholar):
    arxiv.reply(200, text=recorded("arxiv_no_results.xml"))
    response = client.get("/", params={"q": "zzqqxx", "sort": "citations"})
    assert parse_html(response.text).find(class_="empty-state") is not None
    assert scholar.requests == []


def test_an_arxiv_failure_stops_before_semantic_scholar(client, arxiv, scholar):
    arxiv.reply(503)
    response = client.get("/", params={"q": "testing", "sort": "citations"})
    assert response.status_code == 503
    assert scholar.requests == []


def test_most_cited_marks_papers_already_in_the_library(client, arxiv, scholar, add_paper):
    paper_id = add_paper(arxiv_id="2401.00002")
    arxiv.reply(200, text=numbered_feed(2))
    scholar.reply(200, json=counts_reply(1, 2))
    top_card = parse_html(client.get("/", params={"q": "testing", "sort": "citations"}).text).find("li", class_="entry")
    assert top_card.find("a", class_="button").attrs["href"] == f"/papers/{paper_id}"


def test_an_unknown_sort_falls_back_to_arxiv_rank(client, arxiv):
    arxiv.reply(200, text=numbered_feed(1))
    response = client.get("/", params={"q": "testing", "sort": "bogus"})
    assert response.status_code == 200
    assert arxiv.requests[0].url.params["sortBy"] == "relevance"
    assert parse_html(response.text).find("option", value="relevance").attrs.keys() >= {"selected"}


# --- Problems show up inline, with real status codes --------------------------------


def alert_text(html: str) -> str:
    return parse_html(html).find(class_="alert-error").text


def test_arxiv_staying_busy_is_a_503_with_a_message(client, arxiv):
    arxiv.reply(429)
    response = client.get("/", params={"q": "testing"}, headers=HTMX)
    assert response.status_code == 503
    assert "arXiv is busy right now" in alert_text(response.text)
    assert len(arxiv.requests) == 3


def test_a_failed_search_on_a_full_page_load_keeps_the_form_and_the_query(client, arxiv):
    arxiv.fail(httpx.ConnectError("no route"))
    response = client.get("/", params={"q": "testing"})
    assert response.status_code == 503
    assert "arXiv is busy" in alert_text(response.text)
    page = parse_html(response.text)
    assert page.find("title") is not None
    assert page.find("input", name="q").attrs["value"] == "testing"


def test_a_real_arxiv_rejection_is_a_502(client, arxiv):
    arxiv.reply(400, text=recorded("arxiv_bad_id.xml"))
    response = client.get("/", params={"q": "testing"})
    assert response.status_code == 502
    assert "rejected the search" in alert_text(response.text)


def test_a_real_error_entry_is_a_502_with_arxivs_reason(client, arxiv):
    arxiv.reply(200, text=recorded("arxiv_bad_id.xml"))
    response = client.get("/", params={"q": "testing"})
    assert response.status_code == 502
    assert "incorrect id format" in alert_text(response.text)


@pytest.mark.parametrize(
    "body",
    ["Service Unavailable", "<html><body><h1>Down for maintenance</h1></body></html>", ""],
    ids=["plain-text", "well-formed-html", "empty"],
)
def test_a_maintenance_page_is_reported_not_treated_as_no_results(client, arxiv, body):
    arxiv.reply(200, text=body)
    response = client.get("/", params={"q": "testing"})
    assert response.status_code == 502
    assert "isn't a search feed" in alert_text(response.text)


def test_semantic_scholar_staying_busy_is_a_503_that_suggests_another_sort(client, arxiv, scholar):
    arxiv.reply(200, text=numbered_feed(3))
    scholar.reply(429, text=recorded("semantic_scholar_429.json"))

    response = client.get("/", params={"q": "testing", "sort": "citations"}, headers=HTMX)

    assert response.status_code == 503
    assert "Semantic Scholar is busy" in alert_text(response.text)
    assert "sort by arXiv rank" in alert_text(response.text)
    assert titles(response.text) == []


@pytest.mark.parametrize("query", ["the of and", "!!!"], ids=["only-stop-words", "only-punctuation"])
def test_nothing_searchable_asks_for_keywords_without_calling_arxiv(client, query):
    response = client.get("/", params={"q": query})
    assert response.status_code == 400
    assert "Type a topic or some keywords" in alert_text(response.text)


# --- Untrusted text stays text ------------------------------------------------------


def test_a_query_with_html_is_escaped_everywhere_it_is_shown(client, arxiv):
    arxiv.reply(200, text=recorded("arxiv_no_results.xml"))
    query = "<script>alert(1)</script>"

    response = client.get("/", params={"q": query})

    assert query not in response.text
    page = parse_html(response.text)
    assert [script for script in page.find_all("script") if "alert(1)" in script.text] == []
    assert page.find("input", name="q").attrs["value"] == query


def test_html_in_arxiv_metadata_is_escaped(client, arxiv):
    arxiv.reply(
        200,
        text=atom_feed(
            atom_entry(
                title="<img src=x onerror=alert(1)> Title",
                authors=["<b>Eve</b>"],
                abstract="<script>alert(2)</script> Abstract",
            )
        ),
    )
    response = client.get("/", params={"q": "testing"})
    page = parse_html(response.text)
    results = page.find(id="results")
    assert results.find("img") is None
    assert results.find("script") is None
    assert results.find("b") is None
    assert results.find(class_="entry-title").text == "<img src=x onerror=alert(1)> Title"


# --- Rare shapes of a result ---------------------------------------------------------


def test_a_single_most_cited_result_is_described_in_the_singular(client, arxiv, scholar):
    arxiv.reply(200, text=numbered_feed(1))
    scholar.reply(200, json=counts_reply(5))
    summary = parse_html(client.get("/", params={"q": "testing", "sort": "citations"}).text).find(class_="results-count")
    assert "1 paper for “testing”, most cited first" in summary.text


def test_a_result_with_no_abstract_still_offers_save(client, arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry(abstract="")))
    card = parse_html(client.get("/", params={"q": "testing"}).text).find("li", class_="entry")
    assert card.find(class_="abstract") is None
    assert card.find("button", class_="link-button") is None
    assert card.find("form").find("button", type="submit") is not None
