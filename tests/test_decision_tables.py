"""Decision tables and combinatorial testing: every combination of inputs that interact.

Each test computes what should happen from the design's rules, so a wrong combination can't pass
just because nobody thought to write that case down.
"""

from __future__ import annotations

import itertools
import json

import pytest

from app import db
from helpers import (
    BODY_FACT,
    HTMX,
    PAPER_ABSTRACT,
    PAPER_TITLE,
    atom_entry,
    atom_feed,
    make_pdf,
    parse_html,
)

# --- Where an upload's metadata comes from (design section 8) ------------------------------

GEMINI_TITLE = "Title From Gemini"
GEMINI_REPLY = {"title": GEMINI_TITLE, "authors": ["Gemini Author"], "year": 2023, "abstract": "Abstract from Gemini."}
EMBEDDED_TITLE = "Title Embedded In The File"
BODY_WITHOUT_HEADING = (
    f"{PAPER_TITLE}\nAlice Smith\nThis opening text has no heading above it, so nothing here counts as an abstract at "
    f"all. It keeps going for long enough to be a usable paper in every other way.\n1 Introduction\n{BODY_FACT}"
)


def set_up_gemini(state, request):
    if state == "no-key":
        return
    gemini = request.getfixturevalue("gemini")
    if state == "works":
        gemini.reply_text(json.dumps(GEMINI_REPLY))
    elif state == "finds-no-abstract":
        gemini.reply_text(json.dumps(GEMINI_REPLY | {"abstract": None}))
    elif state == "quota-error":
        gemini.reply_error(429, "Quota exhausted.", "RESOURCE_EXHAUSTED")
    elif state == "unusable-reply":
        gemini.reply_text("this is not json")


@pytest.mark.parametrize(
    ("gemini_state", "embedded", "has_heading"),
    list(
        itertools.product(
            ["no-key", "works", "finds-no-abstract", "quota-error", "unusable-reply"],
            ["real", "placeholder", "none"],
            [True, False],
        )
    ),
)
def test_upload_metadata_decision_table(client, request, gemini_state, embedded, has_heading):
    set_up_gemini(gemini_state, request)
    pages = None if has_heading else [BODY_WITHOUT_HEADING]
    options = {
        "real": {"title": EMBEDDED_TITLE, "author": "Carol White; Dan Brown"},
        "placeholder": {"title": "Microsoft Word - draft3.docx", "author": "Carol White; Dan Brown"},
        "none": {},
    }[embedded]
    pdf = make_pdf(pages, **options) if pages else make_pdf(**options)

    response = client.post("/library/upload", files={"file": ("p.pdf", pdf, "application/pdf")}, headers=HTMX)

    assert response.status_code == 200
    paper = db.get_paper(db.list_papers()[0]["id"])
    gemini_used = gemini_state in ("works", "finds-no-abstract")

    # Title and authors: Gemini first, then the PDF's own fields when they look real, then the first line.
    if gemini_used:
        assert paper["title"] == GEMINI_TITLE
        assert paper["authors"] == ["Gemini Author"]
        assert paper["year"] == 2023
    else:
        assert paper["title"] == (EMBEDDED_TITLE if embedded == "real" else PAPER_TITLE)
        assert paper["authors"] == ([] if embedded == "none" else ["Carol White", "Dan Brown"])
        assert paper["year"] is None

    # Abstract: Gemini's if it found one, then the text under a heading, otherwise nothing.
    if gemini_state == "works":
        assert paper["abstract"] == "Abstract from Gemini."
    else:
        assert paper["abstract"] == (PAPER_ABSTRACT if has_heading else None)

    # The page warns about incomplete details exactly when Gemini wasn't used.
    assert response.headers["HX-Redirect"].endswith("?notice=metadata-fallback") is not gemini_used

    # Whatever the metadata, the full text is always stored.
    assert BODY_FACT in paper["full_text"]


# --- How a search behaves (design sections 4 and 7) ------------------------------------------


@pytest.mark.parametrize(
    ("sort", "query", "rendering", "already_saved"),
    list(
        itertools.product(
            ["relevance", "recent", "citations"],
            ["keywords", "plain-id", "link"],
            ["full-page", "htmx"],
            [True, False],
        )
    ),
)
def test_search_combinations(client, arxiv, scholar, add_paper, sort, query, rendering, already_saved):
    typed = {"keywords": "graph testing", "plain-id": "2401.12345", "link": "https://arxiv.org/abs/2401.12345v2/"}[query]
    paper_id = add_paper() if already_saved else None
    arxiv.reply(200, text=atom_feed(atom_entry("2401.12345", title="The Paper", published="2024-03-05T10:00:00Z")))
    scholar.reply(200, json=[{"citationCount": 7}])

    response = client.get("/", params={"q": typed, "sort": sort}, headers=HTMX if rendering == "htmx" else None)

    assert response.status_code == 200

    # What arXiv is asked: an ID is looked up directly, and otherwise the sort decides.
    params = arxiv.requests[0].url.params
    assert len(arxiv.requests) == 1
    if query == "keywords":
        assert params["search_query"] == "all:graph AND all:testing"
        assert params["sortBy"] == ("submittedDate" if sort == "recent" else "relevance")
        assert params["max_results"] == ("100" if sort == "citations" else "20")
    else:
        assert params["id_list"] == "2401.12345"
        assert "search_query" not in params
        assert params["max_results"] == "1"

    # Semantic Scholar is only ever involved in a most-cited search.
    assert len(scholar.requests) == (1 if sort == "citations" else 0)

    # What is rendered: only the results for htmx, and the whole page otherwise.
    assert ("<html" in response.text) is (rendering == "full-page")
    page = parse_html(response.text)
    card = page.find("li", class_="entry")
    assert card.find("h2").text == "The Paper"

    # The card's extra fact follows the sort.
    facts = card.find(class_="facts")
    if sort == "relevance":
        assert facts is None
    elif sort == "recent":
        assert facts.text == "Submitted 5 Mar 2024"
    else:
        assert facts.text == "7 citations"

    # A saved paper links to the library, and any other offers Save.
    assert (card.find("form") is None) is already_saved
    if already_saved:
        assert card.find("a", class_="button").attrs["href"] == f"/papers/{paper_id}"


# --- Acting on a paper (design sections 4 and 9) ---------------------------------------------


@pytest.mark.parametrize(
    ("action", "paper_exists", "has_key", "rendering"),
    list(itertools.product(["summary", "questions"], [True, False], [True, False], ["full-page", "htmx"])),
)
def test_paper_action_combinations(client, request, add_paper, action, paper_exists, has_key, rendering):
    if has_key:
        request.getfixturevalue("gemini").reply_text("An answer.")
    paper_id = add_paper() if paper_exists else 999
    form = {"question": "Why?"} if action == "questions" else None

    response = client.post(f"/papers/{paper_id}/{action}", data=form, headers=HTMX if rendering == "htmx" else None)

    expected = 404 if not paper_exists else (200 if has_key else 503)
    assert response.status_code == expected
    if paper_exists:
        stored = db.get_paper(paper_id)["summary"] if action == "summary" else db.list_questions(paper_id)
        assert bool(stored) is has_key
    if expected != 200 and rendering == "htmx":
        assert parse_html(response.text).find(class_="alert-error") is not None
