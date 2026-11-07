"""Whole journeys through the app, following the links and forms each page actually offers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, db, pdfs
from app.main import app
from helpers import (
    BODY_FACT,
    HTMX,
    PAPER_ABSTRACT,
    PAPER_AUTHORS,
    PAPER_TITLE,
    atom_entry,
    atom_feed,
    hidden_fields,
    make_pdf,
    parse_html,
    stored_pdfs,
)

REPO = Path(__file__).resolve().parent.parent
SUMMARY = "**TL;DR:** Agents write useful tests.\n\n### Problem\nTesting is slow."
ANSWER = "They used the **Zephyr** benchmark."


def search_feed() -> str:
    return atom_feed(atom_entry("2401.12345", title=PAPER_TITLE, authors=PAPER_AUTHORS, abstract=PAPER_ABSTRACT))


def submit_save_form(client, page):
    """Press Save on the first result: submit exactly what its form holds."""
    form = page.find(class_="save-slot").find("form")
    return client.post(form.attrs["hx-post"], data=hidden_fields(form), headers=HTMX)


def test_find_save_read_summarize_ask_and_delete(client, arxiv, arxiv_pdfs, gemini):
    arxiv.reply(200, text=search_feed())
    arxiv_pdfs.reply(200, content=make_pdf())
    gemini.reply_text(SUMMARY).reply_text(ANSWER)

    # Search and save the first result.
    results = parse_html(client.get("/", params={"q": "coding agents"}).text)
    saved = submit_save_form(client, results)
    assert saved.status_code == 200
    paper_url = parse_html(saved.text).find("a", class_="button").attrs["href"]

    # Searching again now links to the saved paper instead of offering Save.
    results = parse_html(client.get("/", params={"q": "coding agents"}).text)
    assert results.find(class_="save-slot").find("a").attrs["href"] == paper_url
    assert results.find(id="library-count").text == "1"

    # The library lists it, and its link opens the paper.
    library = parse_html(client.get("/library").text)
    assert library.find("h2", class_="entry-title").find("a").attrs["href"] == paper_url
    paper = parse_html(client.get(paper_url).text)
    assert paper.find("h1").text == PAPER_TITLE
    assert paper.find(class_="abstract").text == PAPER_ABSTRACT

    # Press Summarize, then a suggested question.
    summary = client.post(paper.find(id="summary-panel").find("button").attrs["hx-post"], headers=HTMX)
    assert parse_html(summary.text).find(class_="summary").find("h3").text == "Problem"
    assert BODY_FACT in gemini.prompt(0)
    chip = paper.find("button", class_="chip")
    answer = client.post(chip.attrs["hx-post"], data=json.loads(chip.attrs["hx-vals"]), headers=HTMX)
    assert parse_html(answer.text).find(class_="answer").find("strong").text == "Zephyr"

    # Reopening shows both, and the library says so.
    paper = parse_html(client.get(paper_url).text)
    assert paper.find(class_="summary").find("strong").text == "TL;DR:"
    assert paper.find(class_="question").text == "What problem does this paper address?"
    row = parse_html(client.get("/library").text).find("li", class_="entry")
    assert row.find(class_="facts").text == "From arXiv Summarized 1 question"

    # Delete from the library: the paper and its PDF are gone.
    deleted = client.delete(row.find("button", class_="button-danger").attrs["hx-delete"], headers=HTMX)
    assert parse_html(deleted.text).find(id="library-count").text == "0"
    assert client.get(paper_url).status_code == 404
    assert stored_pdfs() == []

    # The paper can be found and saved again afterwards.
    results = parse_html(client.get("/", params={"q": "coding agents"}).text)
    assert submit_save_form(client, results).status_code == 200
    assert db.count_papers() == 1


def test_upload_read_summarize_and_ask(client, gemini):
    pdf = make_pdf()
    metadata = {"title": "An Uploaded Paper", "authors": ["Carol White"], "year": 2022, "abstract": "Uploaded abstract."}
    gemini.reply_text(json.dumps(metadata)).reply_text(SUMMARY).reply_text(ANSWER)

    uploaded = client.post("/library/upload", files={"file": ("paper.pdf", pdf, "application/pdf")}, headers=HTMX)
    paper_url = uploaded.headers["HX-Redirect"]

    # It renders like a saved search result.
    paper = parse_html(client.get(paper_url).text)
    assert paper.find("h1").text == "An Uploaded Paper"
    assert paper.find("header", class_="paper-head").find(class_="authors").text == "Carol White"
    assert paper.find("header", class_="paper-head").find(class_="facts").text == "2022 Uploaded PDF"
    assert paper.find(class_="abstract").text == "Uploaded abstract."
    row = parse_html(client.get("/library").text).find("li", class_="entry")
    assert row.find("h2").text == "An Uploaded Paper"
    assert row.find(class_="entry-margin").text == "2022"

    # Its PDF opens, byte for byte.
    assert client.get(paper.find("a", class_="button", target="_blank").attrs["href"]).content == pdf

    # Summaries and questions use the uploaded PDF's text.
    client.post(f"{paper_url}/summary", headers=HTMX)
    assert BODY_FACT in gemini.prompt(1)
    client.post(f"{paper_url}/questions", data={"question": "Which benchmark?"}, headers=HTMX)
    assert BODY_FACT in gemini.prompt(2)
    assert db.get_paper(int(paper_url.rsplit("/", 1)[1]))["summary"] == SUMMARY


def test_everything_keeps_working_without_a_gemini_key(client, arxiv, arxiv_pdfs):
    arxiv.reply(200, text=search_feed())
    arxiv_pdfs.reply(200, content=make_pdf())

    results = parse_html(client.get("/", params={"q": "coding agents"}).text)
    assert submit_save_form(client, results).status_code == 200
    uploaded = client.post("/library/upload", files={"file": ("paper.pdf", make_pdf(), "application/pdf")}, headers=HTMX)
    assert uploaded.status_code == 200

    assert client.get("/library").status_code == 200
    assert client.get(uploaded.headers["HX-Redirect"]).status_code == 200
    assert db.count_papers() == 2


@pytest.mark.parametrize(
    ("title", "authors", "abstract"),
    [
        ('Foo & "Bar" <baz> \'qux\'', ["Conan O'Brien", 'Jane "JJ" Doe'], "Uses <tags> & 'quotes' \"everywhere\"."),
        ("Schrödinger's Équation: 量子", ["José Müller", "李明"], "Résumé — naïve café."),
    ],
    ids=["html-characters", "non-english"],
)
def test_special_characters_survive_from_arxiv_to_the_library(client, arxiv, arxiv_pdfs, title, authors, abstract):
    arxiv.reply(200, text=atom_feed(atom_entry(title=title, authors=authors, abstract=abstract)))
    arxiv_pdfs.reply(200, content=make_pdf())

    results = parse_html(client.get("/", params={"q": "anything"}).text)
    saved = submit_save_form(client, results)

    paper = db.get_paper(db.find_by_arxiv_id("2401.12345"))
    assert (paper["title"], paper["authors"], paper["abstract"]) == (title, authors, abstract)
    page = parse_html(client.get(parse_html(saved.text).find("a", class_="button").attrs["href"]).text)
    assert page.find("h1").text == " ".join(title.split())


def test_the_library_survives_an_app_restart(arxiv_pdfs, gemini):
    pdf = make_pdf()
    arxiv_pdfs.reply(200, content=pdf)
    gemini.reply_text(SUMMARY).reply_text(ANSWER)
    form = {"arxiv_id": "2401.12345", "title": PAPER_TITLE, "authors": json.dumps(PAPER_AUTHORS), "year": "2024"}

    with TestClient(app) as first_run:
        saved = first_run.post("/library", data=form, headers=HTMX)
        paper_url = parse_html(saved.text).find("a", class_="button").attrs["href"]
        first_run.post(f"{paper_url}/summary", headers=HTMX)
        first_run.post(f"{paper_url}/questions", data={"question": "Which benchmark?"}, headers=HTMX)

    with TestClient(app) as second_run:
        page = parse_html(second_run.get(paper_url).text)
        assert page.find("h1").text == PAPER_TITLE
        assert page.find(class_="summary").find("h3").text == "Problem"
        assert page.find(class_="question").text == "Which benchmark?"
        assert second_run.get(f"{paper_url}/pdf").content == pdf


def test_saved_papers_are_on_disk_for_a_separate_process(add_paper):
    paper_id = add_paper()
    db.set_summary(paper_id, "Stored summary.")
    script = (
        "from app import db\n"
        f"paper = db.get_paper({paper_id})\n"
        "print(paper['title'], '|', paper['summary'], '|', db.count_papers())\n"
    )

    finished = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env=os.environ | {"DATA_DIR": str(config.settings.data_dir), "GEMINI_API_KEY": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == f"{PAPER_TITLE} | Stored summary. | 1"
    assert pdfs.pdf_path(db.get_paper(paper_id)["pdf_path"]).is_file()
