"""Boundary-value analysis: every limit in the app, tested just below, at, and just above it."""

from __future__ import annotations

import json

import pytest
from google import genai

from app import arxiv_client, citations, config, db, llm, pdfs
from helpers import HTMX, atom_entry, atom_feed, make_pdf, parse_html, recorded


def save_form(**overrides: str) -> dict[str, str]:
    return {"arxiv_id": "2401.12345", "title": "A Paper", "authors": "[]", "year": "2024", "abstract": ""} | overrides


def saved_paper() -> dict:
    papers = db.list_papers()
    assert len(papers) == 1
    return db.get_paper(papers[0]["id"])


# --- Search ------------------------------------------------------------------------


@pytest.mark.parametrize(("length", "kept"), [(299, 299), (300, 300), (301, 300)])
def test_search_queries_are_cut_at_300_characters(client, arxiv, length, kept):
    arxiv.reply(200, text=recorded("arxiv_no_results.xml"))
    response = client.get("/", params={"q": "x" * length})
    assert arxiv.requests[0].url.params["search_query"] == "all:" + "x" * kept
    assert parse_html(response.text).find("input", name="q").attrs["value"] == "x" * kept


@pytest.mark.parametrize(("words", "kept"), [(11, 11), (12, 12), (13, 12)])
def test_search_queries_keep_at_most_12_words(words, kept):
    query = " ".join(f"word{n}" for n in range(words))
    assert arxiv_client.build_params(query)["search_query"].count("all:") == kept


@pytest.mark.parametrize(("length", "clamped"), [(360, False), (361, True)])
def test_only_abstracts_over_360_characters_collapse(client, arxiv, length, clamped):
    arxiv.reply(200, text=atom_feed(atom_entry(abstract="x" * length)))
    card = parse_html(client.get("/", params={"q": "testing"}).text).find("li", class_="entry")
    assert ("clamped" in card.find(class_="abstract").attrs["class"]) is clamped
    assert (card.find("button", class_="link-button") is not None) is clamped


@pytest.mark.parametrize(("pool", "note"), [(99, False), (100, True)])
def test_the_pool_note_appears_only_when_the_pool_is_full(client, arxiv, scholar, pool, note):
    arxiv.reply(200, text=atom_feed(*(atom_entry(f"2401.{n:05d}") for n in range(1, pool + 1))))
    scholar.reply(200, json=[{"citationCount": n} for n in range(pool)])
    summary = parse_html(client.get("/", params={"q": "testing", "sort": "citations"}).text).find(class_="results-count")
    assert ("best matches" in summary.text) is note


@pytest.mark.parametrize(("matches", "shown"), [(19, 19), (20, 20), (21, 20)])
def test_most_cited_shows_at_most_20(client, arxiv, scholar, matches, shown):
    arxiv.reply(200, text=atom_feed(*(atom_entry(f"2401.{n:05d}") for n in range(1, matches + 1))))
    scholar.reply(200, json=[{"citationCount": n} for n in range(matches)])
    page = parse_html(client.get("/", params={"q": "testing", "sort": "citations"}).text)
    assert len(page.find_all("li", class_="entry")) == shown


@pytest.mark.parametrize(
    ("offset", "expected"),
    [(0, True), (1020, True), (1021, False)],
    ids=["at-the-start", "last-position-that-fits", "crosses-the-limit"],
)
def test_the_pdf_marker_must_fall_inside_the_first_1024_bytes(offset, expected):
    assert pdfs.looks_like_pdf(b"x" * offset + b"%PDF-1.7") is expected


# --- Saving ------------------------------------------------------------------------


@pytest.mark.parametrize(("length", "kept"), [(499, 499), (500, 500), (501, 500)])
def test_titles_from_the_form_are_cut_at_500_characters(client, arxiv_pdfs, length, kept):
    arxiv_pdfs.reply(200, content=make_pdf())
    client.post("/library", data=save_form(title="x" * length), headers=HTMX)
    assert saved_paper()["title"] == "x" * kept


@pytest.mark.parametrize(("length", "kept"), [(9_999, 9_999), (10_000, 10_000), (10_001, 10_000)])
def test_abstracts_from_the_form_are_cut_at_10000_characters(client, arxiv_pdfs, length, kept):
    arxiv_pdfs.reply(200, content=make_pdf())
    client.post("/library", data=save_form(abstract="x" * length), headers=HTMX)
    assert saved_paper()["abstract"] == "x" * kept


@pytest.mark.parametrize(("count", "kept"), [(199, 199), (200, 200), (201, 200)])
def test_author_lists_from_the_form_are_cut_at_200_names(client, arxiv_pdfs, count, kept):
    arxiv_pdfs.reply(200, content=make_pdf())
    client.post("/library", data=save_form(authors=json.dumps([f"Author {n}" for n in range(count)])), headers=HTMX)
    assert saved_paper()["authors"] == [f"Author {n}" for n in range(kept)]


# --- Uploading: the same limits, whatever the metadata's source -----------------------


def test_an_absurdly_long_title_from_gemini_is_cut_like_any_other(client, gemini):
    gemini.reply_text(json.dumps({"title": "x" * 501, "authors": [], "abstract": "y" * 10_001}))
    client.post("/library/upload", files={"file": ("p.pdf", make_pdf(), "application/pdf")}, headers=HTMX)
    paper = saved_paper()
    assert paper["title"] == "x" * 500
    assert paper["abstract"] == "y" * 10_000


def test_an_absurdly_long_title_in_a_pdfs_metadata_is_cut_like_any_other(client):
    client.post("/library/upload", files={"file": ("p.pdf", make_pdf(title="x" * 501), "application/pdf")}, headers=HTMX)
    assert saved_paper()["title"] == "x" * 500


# --- Reading a PDF's metadata ----------------------------------------------------------


@pytest.mark.parametrize(("title", "expected"), [("x" * 7, None), ("x" * 8, "x" * 8)], ids=["too-short", "shortest-allowed"])
def test_embedded_titles_need_at_least_8_characters(title, expected):
    assert pdfs.extract(make_pdf(title=title)).embedded_title == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [("abcdefg", None), ("abcdefgh", "abcdefgh")],
    ids=["too-short", "shortest-allowed"],
)
def test_latin_title_lines_need_at_least_8_characters(line, expected):
    assert pdfs.title_from_text(line) == expected


@pytest.mark.parametrize(("length", "kept"), [(299, 299), (300, 300), (301, 300)])
def test_title_lines_are_cut_at_300_characters(length, kept):
    assert pdfs.title_from_text("x" * length) == "x" * kept


@pytest.mark.parametrize(
    ("line", "found"),
    [("12345678 ab cd", False), ("12345678 abc d", True)],
    ids=["two-letters-in-a-row", "three-letters-in-a-row"],
)
def test_a_title_line_needs_three_letters_in_a_row(line, found):
    assert (pdfs.title_from_text(line) is not None) is found


@pytest.mark.parametrize(
    ("names", "expected"),
    [("A; Bob Jones", ["Bob Jones"]), ("Al; Bob Jones", ["Al", "Bob Jones"])],
    ids=["one-letter-is-dropped", "two-letters-are-kept"],
)
def test_author_fragments_need_at_least_2_characters(names, expected):
    assert pdfs.extract(make_pdf(author=names)).embedded_authors == expected


@pytest.mark.parametrize(("length", "kept"), [(2_999, 2_999), (3_000, 3_000), (3_001, 3_000)])
def test_fallback_abstracts_are_cut_at_3000_characters(length, kept):
    assert pdfs.abstract_from_text(f"Abstract\n{'x' * length}\n1 Introduction") == "x" * kept


# --- Questions and context -------------------------------------------------------------


@pytest.mark.parametrize(("earlier", "sent"), [(5, 5), (6, 6), (7, 6)])
def test_up_to_six_earlier_pairs_are_sent(gemini, earlier, sent):
    gemini.reply_text("Answer")
    history = [{"question": f"question-{n}?", "answer": "a."} for n in range(earlier)]
    llm.answer({"title": "T", "full_text": "x"}, "New?", history)
    assert gemini.prompt().count("Q: question-") == sent


# --- Timeouts: each one must reach the request ------------------------------------------


def test_semantic_scholar_requests_have_a_15_second_timeout(scholar):
    scholar.reply(200, json=[{"citationCount": 1}])
    citations.citation_counts(["2401.12345"])
    timeout = scholar.requests[0].extensions["timeout"]
    assert timeout["read"] == 15
    assert timeout["connect"] == 15


def test_gemini_requests_have_a_two_minute_timeout(monkeypatch):
    created = {}

    def spy(**options):
        created.update(options)
        return object()

    monkeypatch.setattr(genai, "Client", spy)
    monkeypatch.setattr(config, "settings", config.Settings("the-key", "test-model", config.settings.data_dir))
    llm.get_client()
    assert created["api_key"] == "the-key"
    assert created["http_options"].timeout == 120_000
