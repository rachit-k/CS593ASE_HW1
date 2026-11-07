"""Concurrency testing: real simultaneous requests and real database locks."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi.testclient import TestClient

from app import config, db
from app.main import app
from helpers import HTMX, make_pdf, parse_html, stored_pdfs

FORM = {"arxiv_id": "2401.12345", "title": "A Paper", "authors": "[]", "year": "2024", "abstract": ""}


def at_the_same_time(count: int, action: Callable[[TestClient, int], httpx.Response]) -> list[httpx.Response]:
    """Run `action` from `count` threads, each with its own client, as close together as possible."""
    start = threading.Barrier(count)

    def run(number: int) -> httpx.Response:
        client = TestClient(app)
        start.wait(timeout=10)
        return action(client, number)

    with ThreadPoolExecutor(count) as pool:
        return list(pool.map(run, range(count)))


def test_many_simultaneous_saves_of_one_paper_keep_one_copy_and_no_stray_files(arxiv_pdfs):
    count = 6
    pdf = make_pdf()
    # Every request has checked that the paper isn't saved yet before any of them finishes downloading.
    all_downloading = threading.Barrier(count)

    def wait_for_the_others(_request: httpx.Request) -> httpx.Response:
        all_downloading.wait(timeout=10)
        return httpx.Response(200, content=pdf)

    arxiv_pdfs.reply_with(wait_for_the_others)

    responses = at_the_same_time(count, lambda client, _n: client.post("/library", data=FORM, headers=HTMX))

    assert [response.status_code for response in responses] == [200] * count
    assert db.count_papers() == 1
    assert len(stored_pdfs()) == 1
    paper_id = db.find_by_arxiv_id("2401.12345")
    for response in responses:
        assert parse_html(response.text).find("a", class_="button").attrs["href"] == f"/papers/{paper_id}"
    assert stored_pdfs()[0].name == db.get_paper(paper_id)["pdf_path"]


def test_simultaneous_uploads_are_all_kept():
    pdf = make_pdf()
    count = 8
    responses = at_the_same_time(
        count,
        lambda client, n: client.post("/library/upload", files={"file": (f"{n}.pdf", pdf, "application/pdf")}, headers=HTMX),
    )
    assert [response.status_code for response in responses] == [200] * count
    assert db.count_papers() == count
    assert len(stored_pdfs()) == count
    assert len({response.headers["HX-Redirect"] for response in responses}) == count


def test_simultaneous_questions_are_all_kept(gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_text("An answer.")
    count = 8
    responses = at_the_same_time(
        count,
        lambda client, n: client.post(f"/papers/{paper_id}/questions", data={"question": f"Question {n}?"}, headers=HTMX),
    )
    assert [response.status_code for response in responses] == [200] * count
    stored = {item["question"] for item in db.list_questions(paper_id)}
    assert stored == {f"Question {n}?" for n in range(count)}


def test_simultaneous_deletes_never_crash_and_leave_nothing(add_paper):
    paper_id = add_paper()
    responses = at_the_same_time(6, lambda client, _n: client.delete(f"/papers/{paper_id}", headers=HTMX))
    statuses = [response.status_code for response in responses]
    assert 200 in statuses
    assert set(statuses) <= {200, 404}
    assert db.get_paper(paper_id) is None
    assert stored_pdfs() == []


def test_reading_the_library_while_it_is_being_written_never_fails(add_paper):
    pdf = make_pdf()
    add_paper()

    def read_or_write(client: TestClient, number: int) -> httpx.Response:
        if number % 2:
            return client.post("/library/upload", files={"file": ("p.pdf", pdf, "application/pdf")}, headers=HTMX)
        return client.get("/library")

    responses = at_the_same_time(10, read_or_write)
    assert [response.status_code for response in responses] == [200] * 10
    assert db.count_papers() == 6


def test_a_write_waits_for_a_brief_lock_instead_of_failing(add_paper):
    paper_id = add_paper()
    blocker = sqlite3.connect(config.settings.db_path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    failures: list[Exception] = []

    def write() -> None:
        try:
            db.add_question(paper_id, "Why?", "Because.")
        except Exception as exc:
            failures.append(exc)

    writer = threading.Thread(target=write)
    writer.start()
    time.sleep(0.3)  # long enough that a write with no patience would already have failed
    assert writer.is_alive()
    blocker.execute("COMMIT")
    blocker.close()
    writer.join(timeout=10)

    assert failures == []
    assert [item["question"] for item in db.list_questions(paper_id)] == ["Why?"]


def test_two_summaries_at_once_leave_one_complete_summary(gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_text("A complete summary.")
    responses = at_the_same_time(2, lambda client, _n: client.post(f"/papers/{paper_id}/summary", headers=HTMX))
    assert [response.status_code for response in responses] == [200, 200]
    assert db.get_paper(paper_id)["summary"] == "A complete summary."
