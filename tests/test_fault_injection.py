"""Fault injection: make a dependency fail at each step, and check what the user gets and what's left behind.

Design section 9 promises friendly errors with real status codes and nothing half-saved. These tests
break the database, the disk, and each outside service in every way they can realistically fail.
"""

from __future__ import annotations

import errno
import json
import logging
import shutil
import sqlite3
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import arxiv_client, citations, config, db, llm, pdfs
from app.main import _parse_authors, app
from helpers import HTMX, PAPER_TITLE, atom_entry, atom_feed, make_pdf, parse_html, recorded, stored_pdfs

SAVE_FORM = {"arxiv_id": "2401.99999", "title": "A Paper", "authors": "[]", "year": "2024", "abstract": ""}
SECRET = "secret internal detail"


def locked(*_args, **_kwargs):
    raise sqlite3.OperationalError(SECRET)


@pytest.fixture
def unforgiving_client():
    """A client that reports server errors as responses, the way a browser sees them."""
    return TestClient(app, raise_server_exceptions=False)


# --- The database fails partway through --------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "form", "failing"),
    [
        ("GET", "/library", None, "list_papers"),
        ("GET", "/papers/{id}", None, "get_paper"),
        ("GET", "/papers/{id}", None, "list_questions"),
        ("GET", "/papers/{id}/pdf", None, "get_paper"),
        ("POST", "/papers/{id}/summary", None, "get_paper"),
        ("POST", "/papers/{id}/summary", None, "set_summary"),
        ("POST", "/papers/{id}/questions", {"question": "Why?"}, "get_paper"),
        ("POST", "/papers/{id}/questions", {"question": "Why?"}, "list_questions"),
        ("POST", "/papers/{id}/questions", {"question": "Why?"}, "add_question"),
        ("DELETE", "/papers/{id}", None, "delete_paper"),
        ("POST", "/library", SAVE_FORM, "find_by_arxiv_id"),
        ("POST", "/library", SAVE_FORM, "add_paper"),
        ("GET", "/?q=testing", None, "library_ids_for"),
    ],
    ids=lambda value: value.replace("/", "-") if isinstance(value, str) else "",
)
def test_a_database_failure_is_a_friendly_error_that_reveals_nothing(
    unforgiving_client, gemini, arxiv, arxiv_pdfs, add_paper, monkeypatch, method, path, form, failing
):
    paper_id = add_paper()
    gemini.reply_text("An answer.")
    arxiv.reply(200, text=atom_feed(atom_entry()))
    arxiv_pdfs.reply(200, content=make_pdf())
    monkeypatch.setattr(db, failing, locked)

    response = unforgiving_client.request(method, path.format(id=paper_id), data=form, headers=HTMX)

    assert response.status_code == 500
    assert SECRET not in response.text
    assert "<html" not in response.text
    assert "Something went wrong" in parse_html(response.text).find(class_="alert-error").text


def test_a_failed_summary_write_leaves_no_summary(unforgiving_client, gemini, add_paper, monkeypatch):
    paper_id = add_paper()
    gemini.reply_text("A summary.")
    with monkeypatch.context() as patch:
        patch.setattr(db, "set_summary", locked)
        response = unforgiving_client.post(f"/papers/{paper_id}/summary", headers=HTMX)
    assert response.status_code == 500
    assert db.get_paper(paper_id)["summary"] is None


def test_a_full_page_failure_shows_the_error_page_with_a_way_out(unforgiving_client, monkeypatch):
    monkeypatch.setattr(db, "list_papers", locked)
    response = unforgiving_client.get("/library")
    page = parse_html(response.text)
    assert response.status_code == 500
    assert page.find("h1").text == "Something went wrong"
    assert page.find("a", href="/") is not None
    assert SECRET not in response.text


def test_a_corrupted_database_file_never_crashes_a_page(unforgiving_client):
    config.settings.db_path.write_bytes(b"this is not a database at all, " * 100)

    home = unforgiving_client.get("/")
    library = unforgiving_client.get("/library")

    assert home.status_code == 200
    assert parse_html(home.text).find(id="library-count").text == ""
    assert library.status_code == 500
    assert parse_html(library.text).find("h1").text == "Something went wrong"


def test_a_data_folder_removed_while_running_never_crashes_a_page(unforgiving_client):
    shutil.rmtree(config.settings.data_dir)

    home = unforgiving_client.get("/")
    library = unforgiving_client.get("/library")

    assert home.status_code == 200
    assert library.status_code == 500
    assert parse_html(library.text).find("h1").text == "Something went wrong"


# --- The disk fails -----------------------------------------------------------------------


@pytest.fixture
def full_disk(monkeypatch):
    def no_space(_path, _data):
        raise OSError(errno.ENOSPC, SECRET)

    monkeypatch.setattr(Path, "write_bytes", no_space)


def test_a_full_disk_while_saving_stores_nothing(unforgiving_client, arxiv_pdfs, full_disk):
    arxiv_pdfs.reply(200, content=make_pdf())
    response = unforgiving_client.post("/library", data=SAVE_FORM, headers=HTMX)
    assert response.status_code == 500
    assert SECRET not in response.text
    assert db.count_papers() == 0


def test_a_full_disk_while_uploading_stores_nothing(unforgiving_client, full_disk):
    response = unforgiving_client.post("/library/upload", files={"file": ("p.pdf", make_pdf(), "application/pdf")}, headers=HTMX)
    assert response.status_code == 500
    assert SECRET not in response.text
    assert db.count_papers() == 0


def test_a_pdf_that_cannot_be_removed_does_not_block_deleting_the_paper(client, add_paper, monkeypatch):
    paper_id = add_paper()

    def refuse(_path, *_args, **_kwargs):
        raise PermissionError("the file is locked")

    monkeypatch.setattr(Path, "unlink", refuse)
    response = client.delete(f"/papers/{paper_id}", headers=HTMX)

    assert response.status_code == 200
    assert db.get_paper(paper_id) is None
    assert parse_html(response.text).find(id="library-count").text == "0"


def test_summaries_and_questions_keep_working_if_the_pdf_file_disappears(client, gemini, add_paper):
    paper_id = add_paper()
    pdfs.delete_pdf(db.get_paper(paper_id)["pdf_path"])
    assert stored_pdfs() == []
    gemini.reply_text("Still works.")

    assert client.post(f"/papers/{paper_id}/summary", headers=HTMX).status_code == 200
    assert client.post(f"/papers/{paper_id}/questions", data={"question": "Why?"}, headers=HTMX).status_code == 200
    assert client.get(f"/papers/{paper_id}").status_code == 200


# --- Outside services fail in every way httpx can report ---------------------------------

NETWORK_FAILURES = [
    httpx.ConnectError("no route"),
    httpx.ConnectTimeout("slow"),
    httpx.ReadTimeout("slow"),
    httpx.WriteTimeout("slow"),
    httpx.PoolTimeout("slow"),
    httpx.ReadError("reset"),
    httpx.WriteError("reset"),
    httpx.RemoteProtocolError("garbled"),
    httpx.ProxyError("proxy down"),
    httpx.DecodingError("bad gzip"),
    httpx.TooManyRedirects("loop"),
]
FAILURE_IDS = [type(failure).__name__ for failure in NETWORK_FAILURES]


@pytest.mark.parametrize("failure", NETWORK_FAILURES, ids=FAILURE_IDS)
def test_arxiv_failing_in_any_way_says_it_is_busy(client, arxiv, failure):
    arxiv.fail(failure)
    response = client.get("/", params={"q": "testing"}, headers=HTMX)
    assert response.status_code == 503
    assert "arXiv is busy" in parse_html(response.text).find(class_="alert-error").text
    assert len(arxiv.requests) == 3


@pytest.mark.parametrize("failure", NETWORK_FAILURES, ids=FAILURE_IDS)
def test_arxiv_failing_in_any_way_can_recover_on_a_retry(client, arxiv, failure):
    arxiv.fail(failure).reply(200, text=atom_feed(atom_entry(title="Recovered")))
    response = client.get("/", params={"q": "testing"}, headers=HTMX)
    assert response.status_code == 200
    assert parse_html(response.text).find("h2").text == "Recovered"


@pytest.mark.parametrize("failure", NETWORK_FAILURES, ids=FAILURE_IDS)
def test_semantic_scholar_failing_in_any_way_says_it_is_busy(client, arxiv, scholar, failure):
    arxiv.reply(200, text=atom_feed(atom_entry()))
    scholar.fail(failure)
    response = client.get("/", params={"q": "testing", "sort": "citations"}, headers=HTMX)
    assert response.status_code == 503
    assert "Semantic Scholar is busy" in parse_html(response.text).find(class_="alert-error").text
    assert len(scholar.requests) == 3


@pytest.mark.parametrize("failure", NETWORK_FAILURES, ids=FAILURE_IDS)
def test_a_pdf_download_failing_in_any_way_saves_nothing(client, arxiv_pdfs, failure):
    arxiv_pdfs.fail(failure)
    response = client.post("/library", data=SAVE_FORM, headers=HTMX)
    assert response.status_code == 502
    assert parse_html(response.text).find(class_="inline-error") is not None
    assert db.count_papers() == 0
    assert stored_pdfs() == []


@pytest.mark.parametrize("failure", NETWORK_FAILURES, ids=FAILURE_IDS)
def test_gemini_failing_in_any_way_stores_nothing(client, gemini, add_paper, failure):
    paper_id = add_paper()
    gemini.fail(failure)

    summary = client.post(f"/papers/{paper_id}/summary", headers=HTMX)
    question = client.post(f"/papers/{paper_id}/questions", data={"question": "Why?"}, headers=HTMX)

    assert summary.status_code in (502, 504)
    assert question.status_code in (502, 504)
    assert parse_html(summary.text).find(class_="alert-error") is not None
    assert parse_html(question.text).find(class_="alert-error") is not None
    assert db.get_paper(paper_id)["summary"] is None
    assert db.list_questions(paper_id) == []


@pytest.mark.parametrize("failure", NETWORK_FAILURES, ids=FAILURE_IDS)
def test_an_upload_survives_gemini_failing_in_any_way(client, gemini, failure):
    gemini.fail(failure)
    response = client.post("/library/upload", files={"file": ("p.pdf", make_pdf(), "application/pdf")}, headers=HTMX)
    assert response.status_code == 200
    assert response.headers["HX-Redirect"].endswith("?notice=metadata-fallback")
    assert db.get_paper(db.list_papers()[0]["id"])["title"] == PAPER_TITLE


# --- Gemini answers, but the answer is unusable ------------------------------------------------

UNUSABLE_REPLIES = [
    {"text": "<html>Service Unavailable</html>"},
    {"text": ""},
    {"text": '{"candidates": "this should be a list"}'},
    {"text": '{"candidates": [{"content": "this should be an object"}]}'},
    {"text": "[1, 2, 3]"},
    {"text": "null"},
]
UNUSABLE_IDS = ["html", "empty-body", "candidates-not-a-list", "content-not-an-object", "a-list", "null"]


@pytest.mark.parametrize("reply", UNUSABLE_REPLIES, ids=UNUSABLE_IDS)
def test_an_unusable_gemini_reply_is_a_friendly_error(client, gemini, add_paper, reply):
    paper_id = add_paper()
    gemini.reply(200, **reply)

    summary = client.post(f"/papers/{paper_id}/summary", headers=HTMX)
    question = client.post(f"/papers/{paper_id}/questions", data={"question": "Why?"}, headers=HTMX)

    assert summary.status_code == 502
    assert question.status_code == 502
    assert parse_html(summary.text).find(class_="alert-error") is not None
    assert db.get_paper(paper_id)["summary"] is None
    assert db.list_questions(paper_id) == []


@pytest.mark.parametrize("reply", UNUSABLE_REPLIES, ids=UNUSABLE_IDS)
def test_an_upload_survives_an_unusable_gemini_reply(client, gemini, reply):
    gemini.reply(200, **reply)
    response = client.post("/library/upload", files={"file": ("p.pdf", make_pdf(), "application/pdf")}, headers=HTMX)
    assert response.status_code == 200
    assert response.headers["HX-Redirect"].endswith("?notice=metadata-fallback")


# --- A paper disappears while Gemini is still working on it ----------------------------------


def delete_then_answer(paper_id: int):
    def handler(_request: httpx.Request) -> httpx.Response:
        db.delete_paper(paper_id)
        candidate = {"content": {"role": "model", "parts": [{"text": "An answer."}]}, "finishReason": "STOP"}
        return httpx.Response(200, json={"candidates": [candidate]})

    return handler


def test_a_paper_deleted_during_a_question_gets_a_clear_404(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_with(delete_then_answer(paper_id))

    response = client.post(f"/papers/{paper_id}/questions", data={"question": "Why?"}, headers=HTMX)

    assert response.status_code == 404
    assert response.headers["HX-Retarget"] == "#qa-error"
    assert "deleted" in parse_html(response.text).find(class_="alert-error").text
    assert db.list_questions(paper_id) == []


def test_a_paper_deleted_during_a_summary_gets_a_clear_404(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_with(delete_then_answer(paper_id))

    response = client.post(f"/papers/{paper_id}/summary", headers=HTMX)

    assert response.status_code == 404
    assert parse_html(response.text).find(class_="summary") is None
    assert "deleted" in parse_html(response.text).find(class_="alert-error").text


# --- Mixed retry sequences ----------------------------------------------------------------------


def test_arxiv_recovers_after_two_different_kinds_of_failure(client, arxiv):
    arxiv.fail(httpx.ReadTimeout("slow")).reply(503).reply(200, text=recorded("arxiv_no_results.xml"))
    response = client.get("/", params={"q": "testing"}, headers=HTMX)
    assert response.status_code == 200
    assert len(arxiv.requests) == 3


def test_semantic_scholar_recovers_after_two_different_kinds_of_failure(scholar):
    scholar.fail(httpx.ConnectError("no route")).reply(500).reply(200, json=[{"citationCount": 3}])
    assert citations.citation_counts(["a"]) == {"a": 3}
    assert len(scholar.requests) == 3


def test_a_refusal_on_the_last_attempt_names_that_reason(arxiv):
    arxiv.fail(httpx.ReadTimeout("slow")).reply(429)
    with pytest.raises(arxiv_client.ArxivError, match="HTTP 429 after 3 tries"):
        arxiv_client.search("testing")


def test_the_no_key_path_never_touches_the_network(client, add_paper):
    # The autouse guard would fail this test if the app tried to reach Gemini without a key.
    response = client.post(f"/papers/{add_paper()}/summary", headers=HTMX)
    assert response.status_code == 503
    assert llm.LLMError.__name__ not in response.text


# --- The JSON parser gives up -------------------------------------------------------------------


def test_authors_too_deeply_nested_to_parse_count_as_no_authors(monkeypatch):
    # Python 3.12's parser raises this for about 100,000 open brackets, and 3.14's doesn't, so the
    # fault is injected here to test the same thing on every version.
    def gives_up(_text):
        raise RecursionError

    monkeypatch.setattr(json, "loads", gives_up)
    assert _parse_authors('["Alice"]') == []


# --- The real error reaches the terminal --------------------------------------------------------


@pytest.mark.parametrize("route", ["save", "upload"])
def test_a_database_failure_while_adding_a_paper_keeps_its_own_error_for_the_terminal(client, arxiv_pdfs, monkeypatch, route):
    arxiv_pdfs.reply(200, content=make_pdf())
    monkeypatch.setattr(db, "add_paper", locked)
    # This client lets server errors through, so this is what the terminal running the app would show.
    with pytest.raises(sqlite3.OperationalError, match=SECRET):
        if route == "save":
            client.post("/library", data={"arxiv_id": "2401.12345", "title": "T"}, headers=HTMX)
        else:
            client.post("/library/upload", files={"file": ("p.pdf", make_pdf(), "application/pdf")}, headers=HTMX)
    assert stored_pdfs() == []


def test_gemini_failing_during_an_upload_is_noted_in_the_log(client, gemini, caplog):
    gemini.reply_error(500, "Internal error.", "INTERNAL")
    with caplog.at_level(logging.WARNING, logger="app"):
        response = client.post("/library/upload", files={"file": ("p.pdf", make_pdf(), "application/pdf")}, headers=HTMX)
    assert response.headers["HX-Redirect"].endswith("?notice=metadata-fallback")
    assert "Gemini couldn't extract upload metadata, using fallbacks: Gemini is having trouble" in caplog.text
