"""Metamorphic testing: change an input in a way that shouldn't matter, and check the result doesn't change.

These need no known right answer. They compare two runs of the app with each other, which catches
mistakes that a test written from one example can't see.
"""

from __future__ import annotations

import json

import httpx
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from app import arxiv_client, db, pdfs
from app.arxiv_client import STOP_WORDS
from helpers import BODY_FACT, HTMX, PAPER_ABSTRACT, PAPER_AUTHORS, PAPER_TITLE, atom_entry, atom_feed, make_pdf, parse_html

words = st.lists(st.from_regex(r"[a-z]{3,10}", fullmatch=True).filter(lambda word: word not in STOP_WORDS), min_size=1, max_size=12)


def query_for(text: str) -> str:
    return arxiv_client.build_params(text)["search_query"]


def rendered_results(html: str) -> list[tuple[str, str, str]]:
    return [
        (card.find("h2").text, card.find(class_="authors").text, card.find(class_="entry-margin").text)
        for card in parse_html(html).find_all("li", class_="entry")
    ]


# --- The same search, typed differently, is the same search ------------------------------------


@given(words, st.lists(st.tuples(st.integers(0, 12), st.sampled_from(sorted(STOP_WORDS))), max_size=6))
def test_adding_stop_words_never_changes_the_query(typed, inserts):
    with_stop_words = list(typed)
    for position, word in inserts:
        with_stop_words.insert(min(position, len(with_stop_words)), word)
    assert query_for(" ".join(with_stop_words)) == query_for(" ".join(typed))


@given(words, st.sampled_from([str.upper, str.title, str.swapcase]))
def test_capitalization_never_changes_the_query(typed, change):
    assert query_for(change(" ".join(typed))) == query_for(" ".join(typed))


@given(words, st.sampled_from(["  ", "\t", "\n", ", ", " : ", " ; ", " / ", " ( ", ") ", ' " ', "? ", "! "]))
def test_punctuation_and_spacing_between_words_never_change_the_query(typed, separator):
    assert query_for(f"{separator}{separator.join(typed)}{separator}") == query_for(" ".join(typed))


@given(words)
def test_word_order_is_kept(typed):
    assert query_for(" ".join(typed)) == " AND ".join(f"all:{word}" for word in typed)


@given(st.from_regex(r"[a-z]{2,8}", fullmatch=True), st.from_regex(r"[a-z]{1,3}", fullmatch=True))
def test_straight_and_curly_apostrophes_give_the_same_query(start, end):
    assume(f"{start}'{end}" not in STOP_WORDS)
    assert query_for(f"{start}'{end} testing") == query_for(f"{start}’{end} testing")


@pytest.mark.parametrize(
    "form",
    [
        "2404.01484",
        "arXiv:2404.01484",
        "2404.01484v2",
        "https://arxiv.org/abs/2404.01484",
        "http://arxiv.org/abs/2404.01484v2",
        "https://www.arxiv.org/pdf/2404.01484v2.pdf",
        "arxiv.org/abs/2404.01484/",
        "https://export.arxiv.org/abs/2404.01484?x=1#y",
    ],
)
def test_every_way_of_naming_a_paper_gives_the_same_page(client, arxiv, form):
    arxiv.reply(200, text=atom_feed(atom_entry("2404.01484", title="A HAT Trick")))
    expected = rendered_results(client.get("/", params={"q": "2404.01484"}).text)
    response = client.get("/", params={"q": form})
    assert rendered_results(response.text) == expected == [("A HAT Trick", "Alice Smith, Bob Jones", "2024")]
    assert all(request.url.params["id_list"] == "2404.01484" for request in arxiv.requests)


# --- The same results, shown two ways, are the same results ----------------------------------------


@pytest.mark.parametrize("sort", ["relevance", "recent"])
def test_the_full_page_and_the_htmx_fragment_show_the_same_results(client, arxiv, sort):
    arxiv.reply(200, text=atom_feed(*(atom_entry(f"2401.{n:05d}", title=f"Paper {n}") for n in range(1, 8))))
    full = client.get("/", params={"q": "testing", "sort": sort})
    fragment = client.get("/", params={"q": "testing", "sort": sort}, headers=HTMX)
    assert rendered_results(full.text) == rendered_results(fragment.text)
    assert len(rendered_results(full.text)) == 7


def test_most_cited_shows_the_same_top_papers_whatever_order_arxiv_returned(client, arxiv, scholar):
    counts = {f"2401.{n:05d}": n * 10 for n in range(1, 6)}

    def answer_for(request):
        ids = [item.removeprefix("ARXIV:") for item in json.loads(request.content)["ids"]]
        return httpx.Response(200, json=[{"citationCount": counts[i]} for i in ids])

    entries = [atom_entry(i, title=i) for i in counts]
    arxiv.reply(200, text=atom_feed(*entries))
    scholar.reply_with(answer_for)
    forward = rendered_results(client.get("/", params={"q": "testing", "sort": "citations"}).text)

    arxiv.reset().reply(200, text=atom_feed(*reversed(entries)))
    scholar.reset().reply_with(answer_for)
    backward = rendered_results(client.get("/", params={"q": "testing", "sort": "citations"}).text)

    assert forward == backward
    assert [row[0] for row in forward] == sorted(counts, key=counts.get, reverse=True)


# --- Doing something twice, or undoing it, changes nothing else --------------------------------------


def stored_fields(paper_id: int) -> tuple:
    paper = db.get_paper(paper_id)
    return tuple(paper[name] for name in ("source", "arxiv_id", "title", "authors", "year", "abstract", "url", "full_text"))


SAVE_FORM = {"arxiv_id": "2401.12345", "title": PAPER_TITLE, "authors": json.dumps(PAPER_AUTHORS), "year": "2024", "abstract": PAPER_ABSTRACT}


@pytest.mark.parametrize("second", ["2401.12345", "2401.12345v2", "arXiv:2401.12345", "https://arxiv.org/abs/2401.12345/"])
def test_saving_the_same_paper_again_changes_nothing(client, arxiv_pdfs, second):
    arxiv_pdfs.reply(200, content=make_pdf())
    client.post("/library", data=SAVE_FORM, headers=HTMX)
    paper_id = db.list_papers()[0]["id"]
    before = stored_fields(paper_id)

    client.post("/library", data=SAVE_FORM | {"arxiv_id": second, "title": "Different"}, headers=HTMX)

    assert db.count_papers() == 1
    assert stored_fields(paper_id) == before
    assert len(arxiv_pdfs.requests) == 1


def test_deleting_and_saving_again_restores_the_same_paper(client, arxiv_pdfs):
    arxiv_pdfs.reply(200, content=make_pdf())
    client.post("/library", data=SAVE_FORM, headers=HTMX)
    first = db.list_papers()[0]["id"]
    before = stored_fields(first)

    client.delete(f"/papers/{first}", headers=HTMX)
    client.post("/library", data=SAVE_FORM, headers=HTMX)

    assert stored_fields(db.list_papers()[0]["id"]) == before


def test_regenerating_with_the_same_reply_changes_nothing(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_text("**TL;DR:** The same.")
    first = client.post(f"/papers/{paper_id}/summary", headers=HTMX)
    second = client.post(f"/papers/{paper_id}/summary", headers=HTMX)
    assert parse_html(first.text).find(class_="summary").text == parse_html(second.text).find(class_="summary").text
    assert gemini.prompt(0) == gemini.prompt(1)


@given(st.lists(st.from_regex(r"[A-Za-z]{1,8}", fullmatch=True), min_size=1, max_size=6), st.sampled_from(["  ", "\t", "\n ", "   \n"]))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_extra_spacing_in_a_question_sends_the_same_prompt(client, gemini, add_paper_once, typed, gap):
    gemini.reset().reply_text("Answer.")
    client.post(f"/papers/{add_paper_once}/questions", data={"question": " ".join(typed)}, headers=HTMX)
    client.post(f"/papers/{add_paper_once}/questions", data={"question": gap + gap.join(typed) + gap}, headers=HTMX)
    # The second prompt also carries the first pair as history, so compare only the question itself.
    assert gemini.prompt(0).rsplit("Question: ", 1)[1] == gemini.prompt(1).rsplit("Question: ", 1)[1]


@pytest.fixture
def add_paper_once(add_paper):
    return add_paper(source="upload", arxiv_id=None, url=None)


def test_a_failed_summary_then_a_good_one_is_like_only_the_good_one(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_error(500, "Internal error.", "INTERNAL").reply_text("Good summary.")
    client.post(f"/papers/{paper_id}/summary", headers=HTMX)
    client.post(f"/papers/{paper_id}/summary", headers=HTMX)
    assert db.get_paper(paper_id)["summary"] == "Good summary."


# --- What a PDF says doesn't depend on how it's dressed ----------------------------------------------


@pytest.mark.parametrize(
    "dressing",
    [{"title": "A Different Title Entirely"}, {"author": "Someone Else"}, {"title": "x", "author": "y"}],
    ids=["title", "author", "both"],
)
def test_a_pdfs_metadata_never_changes_its_text(dressing):
    assert pdfs.extract(make_pdf(**dressing)).full_text == pdfs.extract(make_pdf()).full_text


def test_adding_later_pages_never_changes_the_first_pages_text():
    pages = ("Page one has some words on it. " * 10, "Page two has other words on it. " * 10)
    shorter = pdfs.extract(make_pdf(pages))
    longer = pdfs.extract(make_pdf((*pages, "Page three is new. " * 10, "Page four is new as well. " * 10)))
    assert longer.first_pages_text == shorter.first_pages_text
    assert longer.full_text.startswith(shorter.full_text)
    assert "Page four" in longer.full_text


def test_a_downloaded_upload_is_byte_identical_and_uploading_it_again_gives_the_same_text(client):
    original = make_pdf()
    first = client.post("/library/upload", files={"file": ("p.pdf", original, "application/pdf")}, headers=HTMX)
    first_id = int(first.headers["HX-Redirect"].split("/")[2].split("?")[0])
    downloaded = client.get(f"/papers/{first_id}/pdf").content
    second = client.post("/library/upload", files={"file": ("again.pdf", downloaded, "application/pdf")}, headers=HTMX)
    second_id = int(second.headers["HX-Redirect"].split("/")[2].split("?")[0])

    assert downloaded == original
    assert db.get_paper(first_id)["full_text"] == db.get_paper(second_id)["full_text"]
    assert BODY_FACT in db.get_paper(second_id)["full_text"]
