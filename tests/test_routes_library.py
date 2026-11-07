"""Saving search results, the library page, uploading PDFs, and deleting (design sections 4, 7, 8, 9)."""

from __future__ import annotations

import json
import re
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, db, pdfs
from app.main import app, format_authors, upload_metadata
from helpers import (
    BODY_FACT,
    HTMX,
    PAPER_ABSTRACT,
    PAPER_TITLE,
    hidden_fields,
    make_pdf,
    make_scanned_pdf,
    parse_html,
    stored_pdfs,
)


def save_form(**overrides: str) -> dict[str, str]:
    return {
        "arxiv_id": "2401.12345",
        "title": PAPER_TITLE,
        "authors": json.dumps(["Alice Smith", "Bob Jones"]),
        "year": "2024",
        "abstract": "A short abstract.",
    } | overrides


def pdf_file(data: bytes | None = None, name: str = "paper.pdf") -> dict:
    return {"file": (name, make_pdf() if data is None else data, "application/pdf")}


def only_paper() -> dict:
    papers = db.list_papers()
    assert len(papers) == 1
    return db.get_paper(papers[0]["id"])


# --- Saving a search result ---------------------------------------------------------


def test_saving_stores_the_paper_its_text_and_its_pdf(client, arxiv_pdfs):
    pdf = make_pdf()
    arxiv_pdfs.reply(200, content=pdf)

    response = client.post("/library", data=save_form(), headers=HTMX)

    assert response.status_code == 200
    paper = only_paper()
    assert paper["source"] == "arxiv"
    assert paper["arxiv_id"] == "2401.12345"
    assert paper["title"] == PAPER_TITLE
    assert paper["authors"] == ["Alice Smith", "Bob Jones"]
    assert paper["year"] == 2024
    assert paper["abstract"] == "A short abstract."
    assert paper["url"] == "https://arxiv.org/abs/2401.12345"
    assert BODY_FACT in paper["full_text"]
    assert pdfs.pdf_path(paper["pdf_path"]).read_bytes() == pdf


def test_saving_swaps_the_button_for_a_link_and_updates_the_count(client, arxiv_pdfs):
    arxiv_pdfs.reply(200, content=make_pdf())

    page = parse_html(client.post("/library", data=save_form(), headers=HTMX).text)

    slot = page.find(class_="save-slot")
    assert slot.find(class_="saved-note").text == "In your library"
    assert slot.find("a", class_="button").attrs["href"] == f"/papers/{only_paper()['id']}"
    assert slot.find("form") is None
    count = page.find(id="library-count")
    assert count.attrs["hx-swap-oob"] == "true"
    assert count.text == "1"


def test_saving_downloads_only_from_arxiv_whatever_the_form_says(client, arxiv_pdfs):
    arxiv_pdfs.reply(200, content=make_pdf())

    client.post(
        "/library",
        data=save_form(url="https://evil.example.com/x", pdf_url="https://evil.example.com/x.pdf"),
        headers=HTMX,
    )

    assert [str(request.url) for request in arxiv_pdfs.requests] == ["https://arxiv.org/pdf/2401.12345"]
    assert only_paper()["url"] == "https://arxiv.org/abs/2401.12345"


@pytest.mark.parametrize(
    ("submitted", "stored"),
    [
        ("2401.12345v3", "2401.12345"),
        ("arXiv:2401.12345", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345v3", "2401.12345"),
        ("hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_ids_are_stored_without_their_version(client, arxiv_pdfs, submitted, stored):
    arxiv_pdfs.reply(200, content=make_pdf())
    client.post("/library", data=save_form(arxiv_id=submitted), headers=HTMX)
    assert only_paper()["arxiv_id"] == stored
    assert only_paper()["url"] == f"https://arxiv.org/abs/{stored}"
    assert str(arxiv_pdfs.requests[0].url) == f"https://arxiv.org/pdf/{stored}"


def test_saving_again_downloads_nothing_and_links_to_the_existing_paper(client, arxiv_pdfs, add_paper):
    paper_id = add_paper()

    response = client.post("/library", data=save_form(arxiv_id="2401.12345v2", title="Changed"), headers=HTMX)

    assert response.status_code == 200
    assert arxiv_pdfs.requests == []
    assert db.count_papers() == 1
    assert db.get_paper(paper_id)["title"] == PAPER_TITLE
    page = parse_html(response.text)
    assert page.find("a", class_="button").attrs["href"] == f"/papers/{paper_id}"
    assert page.find(id="library-count") is None


def test_a_double_click_keeps_one_copy_and_no_stray_file(client, arxiv_pdfs, add_paper):
    # The first click's request finishes while this one is still downloading the PDF.
    first_click: list[int] = []

    def other_request_saves_first(_request):
        first_click.append(add_paper())
        return httpx.Response(200, content=make_pdf())

    arxiv_pdfs.reply_with(other_request_saves_first)

    response = client.post("/library", data=save_form(), headers=HTMX)

    assert response.status_code == 200
    assert db.count_papers() == 1
    assert len(stored_pdfs()) == 1
    assert parse_html(response.text).find("a", class_="button").attrs["href"] == f"/papers/{first_click[0]}"


@pytest.mark.parametrize(
    "overrides",
    [
        {"arxiv_id": ""},
        {"arxiv_id": "not-an-id"},
        {"arxiv_id": "https://evil.example.com/abs/2401.12345"},
        {"arxiv_id": "../../etc/passwd"},
        {"title": ""},
        {"title": "    "},
    ],
    ids=["no-id", "bad-id", "other-site", "path", "no-title", "blank-title"],
)
def test_invalid_submissions_are_refused_without_downloading(client, arxiv_pdfs, overrides):
    response = client.post("/library", data=save_form(**overrides), headers=HTMX)

    assert response.status_code == 400
    assert parse_html(response.text).find(class_="inline-error") is not None
    assert arxiv_pdfs.requests == []
    assert db.count_papers() == 0


def test_a_form_with_nothing_in_it_is_refused(client):
    assert client.post("/library", headers=HTMX).status_code == 400
    assert db.count_papers() == 0


@pytest.mark.parametrize(
    ("reply", "status", "message"),
    [
        ({"status": 503}, 502, "HTTP 503"),
        ({"status": 200, "text": "<html>Not a PDF</html>"}, 502, "didn't return a PDF"),
    ],
    ids=["arxiv-error", "not-a-pdf"],
)
def test_a_failed_download_saves_nothing_and_keeps_the_save_button(client, arxiv_pdfs, reply, status, message):
    arxiv_pdfs.reply(**reply)

    response = client.post("/library", data=save_form(), headers=HTMX)

    assert response.status_code == status
    slot = parse_html(response.text).find(class_="save-slot")
    assert message in slot.find(class_="inline-error").text
    assert slot.find("form") is not None
    assert db.count_papers() == 0
    assert stored_pdfs() == []


def test_a_scanned_pdf_from_arxiv_saves_nothing(client, arxiv_pdfs):
    arxiv_pdfs.reply(200, content=make_scanned_pdf())

    response = client.post("/library", data=save_form(), headers=HTMX)

    assert response.status_code == 400
    assert "no readable text" in parse_html(response.text).find(class_="inline-error").text
    assert db.count_papers() == 0
    assert stored_pdfs() == []


def test_a_database_failure_while_saving_leaves_no_stray_file(arxiv_pdfs, monkeypatch):
    arxiv_pdfs.reply(200, content=make_pdf())

    def locked(**_fields):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "add_paper", locked)
    response = TestClient(app, raise_server_exceptions=False).post("/library", data=save_form(), headers=HTMX)

    assert response.status_code == 500
    assert stored_pdfs() == []


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        ('["Alice Smith", "  Bob   Jones "]', ["Alice Smith", "Bob Jones"]),
        ('["Alice", "", "  "]', ["Alice"]),
        ("[]", []),
        ("not json", []),
        ('{"not": "a list"}', []),
    ],
    ids=["names", "blanks", "empty", "not-json", "not-a-list"],
)
def test_authors_from_the_form_are_cleaned_up(client, arxiv_pdfs, authors, expected):
    arxiv_pdfs.reply(200, content=make_pdf())
    assert client.post("/library", data=save_form(authors=authors), headers=HTMX).status_code == 200
    assert only_paper()["authors"] == expected


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        ("2024", 2024),
        (" 2024 ", 2024),
        ("1900", 1900),
        ("2100", 2100),
        ("", None),
        ("abc", None),
        ("1899", None),
        ("2101", None),
        ("-2024", None),
        ("²", None),
        ("①②③④", None),
        pytest.param("9" * 5000, None, id="absurdly-long-number"),
    ],
)
def test_implausible_years_from_the_form_are_dropped(client, arxiv_pdfs, year, expected):
    arxiv_pdfs.reply(200, content=make_pdf())
    assert client.post("/library", data=save_form(year=year), headers=HTMX).status_code == 200
    assert only_paper()["year"] == expected


def test_long_or_messy_text_from_the_form_is_tidied(client, arxiv_pdfs):
    arxiv_pdfs.reply(200, content=make_pdf())
    client.post("/library", data=save_form(title="A  Messy\n Title " + "x" * 600, abstract="  An \n abstract. "), headers=HTMX)
    paper = only_paper()
    assert paper["title"].startswith("A Messy Title x")
    assert len(paper["title"]) == 500
    assert paper["abstract"] == "An abstract."


def test_a_blank_abstract_is_stored_as_none(client, arxiv_pdfs):
    arxiv_pdfs.reply(200, content=make_pdf())
    client.post("/library", data=save_form(abstract="   "), headers=HTMX)
    assert only_paper()["abstract"] is None


# --- The library page ---------------------------------------------------------------


def test_an_empty_library_offers_search_and_upload(client):
    page = parse_html(client.get("/library").text)
    assert page.find("ol", class_="entries").find("li") is None
    assert "Nothing here yet" in page.find(class_="empty-note").text
    assert page.find(id="library-count").text == "0"
    form = page.find("form", class_="upload-form")
    assert form.attrs["hx-post"] == "/library/upload"
    assert form.find("input", type="file").attrs["name"] == "file"


def test_papers_are_listed_newest_first_with_their_details(client, add_paper):
    older = add_paper(arxiv_id="2401.00001", title="Older Paper", year=2021)
    newer = add_paper(source="upload", arxiv_id=None, url=None, title="Newer Paper", year=None, authors=[])

    page = parse_html(client.get("/library").text)

    first, second = page.find("ol", class_="entries").find_all("li", class_="entry")
    assert first.find("h2").find("a").text == "Newer Paper"
    assert first.find("h2").find("a").attrs["href"] == f"/papers/{newer}"
    assert first.find(class_="entry-margin").text == "—"
    assert first.find(class_="authors").text == "Authors unknown"
    assert first.find(class_="facts").text == "Uploaded PDF"
    assert second.find("h2").find("a").text == "Older Paper"
    assert second.find("h2").find("a").attrs["href"] == f"/papers/{older}"
    assert second.find(class_="entry-margin").text == "2021"
    assert second.find(class_="facts").text == "From arXiv"
    assert page.find(id="library-count").text == "2"


def test_the_list_shows_what_has_been_done_with_each_paper(client, add_paper):
    one = add_paper(arxiv_id="2401.00001")
    two = add_paper(arxiv_id="2401.00002")
    db.set_summary(one, "A summary.")
    db.add_question(one, "Q1", "A1")
    db.add_question(two, "Q1", "A1")
    db.add_question(two, "Q2", "A2")

    second, first = parse_html(client.get("/library").text).find("ol", class_="entries").find_all("li", class_="entry")

    assert first.find(class_="facts").text == "From arXiv Summarized 1 question"
    assert second.find(class_="facts").text == "From arXiv 2 questions"


def test_delete_asks_for_confirmation_and_targets_the_row(client, add_paper):
    paper_id = add_paper()
    button = parse_html(client.get("/library").text).find("button", class_="button-danger")
    assert button.attrs["hx-delete"] == f"/papers/{paper_id}"
    assert button.attrs["hx-target"] == "closest li"
    assert PAPER_TITLE in button.attrs["hx-confirm"]


def test_html_in_a_stored_title_is_escaped(client, add_paper):
    add_paper(title='<img src=x onerror=alert(1)> "Quoted"')
    response = client.get("/library")
    page = parse_html(response.text)
    assert page.find("ol", class_="entries").find("img") is None
    assert page.find("h2", class_="entry-title").text == '<img src=x onerror=alert(1)> "Quoted"'
    assert '<img src=x onerror=alert(1)> "Quoted"' in page.find("button", class_="button-danger").attrs["hx-confirm"]


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        ([], "Authors unknown"),
        (["A"], "A"),
        (["A", "B", "C", "D"], "A, B, C, D"),
        (["A", "B", "C", "D", "E"], "A, B, C, and 2 more"),
        ([str(n) for n in range(10)], "0, 1, 2, and 7 more"),
    ],
)
def test_long_author_lists_are_shortened(authors, expected):
    assert format_authors(authors) == expected


# --- Uploading: with Gemini ---------------------------------------------------------

GEMINI_METADATA = {
    "title": "Title From Gemini",
    "authors": ["Gemini Author"],
    "year": 2023,
    "abstract": "An abstract from Gemini.",
}


def test_an_upload_stores_what_gemini_reads_from_the_first_pages(client, gemini):
    pdf = make_pdf()
    gemini.reply_text(json.dumps(GEMINI_METADATA))

    response = client.post("/library/upload", files=pdf_file(pdf), headers=HTMX)

    assert response.status_code == 200
    paper = only_paper()
    assert response.headers["HX-Redirect"] == f"/papers/{paper['id']}"
    assert paper["source"] == "upload"
    assert paper["arxiv_id"] is None
    assert paper["url"] is None
    assert paper["title"] == "Title From Gemini"
    assert paper["authors"] == ["Gemini Author"]
    assert paper["year"] == 2023
    assert paper["abstract"] == "An abstract from Gemini."
    assert BODY_FACT in paper["full_text"]
    assert pdfs.pdf_path(paper["pdf_path"]).read_bytes() == pdf

    prompt = gemini.prompt()
    assert PAPER_TITLE in prompt
    assert "2 Method" in prompt
    assert BODY_FACT not in prompt


def test_when_gemini_finds_no_abstract_the_heading_is_used_and_no_warning_shown(client, gemini):
    gemini.reply_text(json.dumps(GEMINI_METADATA | {"abstract": None}))

    response = client.post("/library/upload", files=pdf_file(), headers=HTMX)

    assert only_paper()["abstract"] == PAPER_ABSTRACT
    assert "notice" not in response.headers["HX-Redirect"]


# --- Uploading: fallbacks -----------------------------------------------------------


def test_without_a_key_the_upload_still_works_from_the_pdf_itself(client):
    response = client.post("/library/upload", files=pdf_file(), headers=HTMX)

    assert response.status_code == 200
    paper = only_paper()
    assert paper["title"] == PAPER_TITLE
    assert paper["authors"] == []
    assert paper["year"] is None
    assert paper["abstract"] == PAPER_ABSTRACT
    assert response.headers["HX-Redirect"] == f"/papers/{paper['id']}?notice=metadata-fallback"


def test_real_looking_embedded_metadata_beats_the_first_line(client):
    pdf = make_pdf(title="Title Embedded In The File", author="Carol White; Dan Brown")
    client.post("/library/upload", files=pdf_file(pdf), headers=HTMX)
    paper = only_paper()
    assert paper["title"] == "Title Embedded In The File"
    assert paper["authors"] == ["Carol White", "Dan Brown"]


def test_a_placeholder_embedded_title_falls_back_to_the_first_line(client):
    client.post("/library/upload", files=pdf_file(make_pdf(title="Microsoft Word - draft3.docx")), headers=HTMX)
    assert only_paper()["title"] == PAPER_TITLE


@pytest.mark.parametrize(
    "fail",
    [
        lambda fake: fake.reply_error(429, "Quota exhausted.", "RESOURCE_EXHAUSTED"),
        lambda fake: fake.reply_error(500, "Internal error.", "INTERNAL"),
        lambda fake: fake.reply_text("this is not json"),
        lambda fake: fake.reply_text(json.dumps({"title": "  ", "authors": []})),
    ],
    ids=["quota", "server-error", "bad-format", "blank-title"],
)
def test_an_upload_never_fails_because_gemini_did(client, gemini, fail):
    fail(gemini)

    response = client.post("/library/upload", files=pdf_file(), headers=HTMX)

    assert response.status_code == 200
    paper = only_paper()
    assert paper["title"] == PAPER_TITLE
    assert response.headers["HX-Redirect"].endswith("?notice=metadata-fallback")


def test_a_pdf_with_no_usable_title_line_is_called_untitled():
    extracted = pdfs.ExtractedPDF(full_text="12 34 56", first_pages_text="12 34 56")
    meta, used_fallback = upload_metadata(extracted)
    assert meta["title"] == "Untitled paper"
    assert used_fallback is True


# --- Uploading: responses and checks ------------------------------------------------


def test_a_normal_form_post_redirects_to_the_paper(client):
    response = client.post("/library/upload", files=pdf_file(), follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == f"/papers/{only_paper()['id']}?notice=metadata-fallback"


def test_files_are_stored_under_generated_names_never_the_users(client):
    client.post("/library/upload", files=pdf_file(name="../../evil.pdf"), headers=HTMX)
    stored = stored_pdfs()
    assert len(stored) == 1
    assert re.fullmatch(r"[0-9a-f]{32}\.pdf", stored[0].name)
    assert only_paper()["pdf_path"] == stored[0].name
    assert not (config.settings.data_dir.parent / "evil.pdf").exists()


def test_the_contents_decide_what_counts_as_a_pdf_not_the_file_name(client):
    assert client.post("/library/upload", files=pdf_file(name="report.txt"), headers=HTMX).status_code == 200
    assert client.post("/library/upload", files=pdf_file(b"plain text", name="paper.pdf"), headers=HTMX).status_code == 400
    assert db.count_papers() == 1


def test_the_same_pdf_can_be_uploaded_twice(client):
    client.post("/library/upload", files=pdf_file(), headers=HTMX)
    client.post("/library/upload", files=pdf_file(), headers=HTMX)
    assert db.count_papers() == 2
    assert len(stored_pdfs()) == 2


@pytest.mark.parametrize(
    ("make_data", "message"),
    [
        (lambda: b"", "Choose a PDF"),
        (lambda: b"just a text file", "isn't a PDF"),
        (lambda: b"%PDF-1.7\n" + b"garbage " * 200, "couldn't be opened"),
        (lambda: make_scanned_pdf(), "no readable text"),
        (lambda: make_pdf(password="secret"), "password"),
    ],
    ids=["empty", "not-a-pdf", "corrupt", "scanned", "password"],
)
def test_unusable_files_show_an_inline_error_and_leave_nothing_behind(client, make_data, message):
    response = client.post("/library/upload", files=pdf_file(make_data()), headers=HTMX)

    assert response.status_code == 400
    assert message in parse_html(response.text).find(class_="alert-error").text
    assert db.count_papers() == 0
    assert stored_pdfs() == []


def test_a_file_over_50_mb_is_refused(client):
    too_big = b"%PDF" + b"\0" * (50 * 1024 * 1024 - 3)
    response = client.post("/library/upload", files=pdf_file(too_big), headers=HTMX)
    assert response.status_code == 413
    assert "larger than 50 MB" in parse_html(response.text).find(class_="alert-error").text
    assert stored_pdfs() == []


def test_a_file_of_exactly_the_limit_is_accepted(client, monkeypatch):
    pdf = make_pdf()
    monkeypatch.setattr(pdfs, "MAX_PDF_BYTES", len(pdf))
    assert client.post("/library/upload", files=pdf_file(pdf), headers=HTMX).status_code == 200
    assert db.count_papers() == 1
    monkeypatch.setattr(pdfs, "MAX_PDF_BYTES", len(pdf) - 1)
    assert client.post("/library/upload", files=pdf_file(pdf), headers=HTMX).status_code == 413
    assert db.count_papers() == 1
    assert len(stored_pdfs()) == 1


def test_no_file_at_all_asks_for_one(client):
    response = client.post("/library/upload", headers=HTMX)
    assert response.status_code == 400
    assert "Choose a PDF" in parse_html(response.text).find(class_="alert-error").text


def test_a_text_field_instead_of_a_file_is_refused(client):
    response = client.post("/library/upload", data={"file": "not a file"}, headers=HTMX)
    assert response.status_code == 400
    assert parse_html(response.text).find(class_="alert-error") is not None
    assert db.count_papers() == 0


def test_a_database_failure_while_uploading_leaves_no_stray_file(monkeypatch):
    def locked(**_fields):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "add_paper", locked)
    response = TestClient(app, raise_server_exceptions=False).post("/library/upload", files=pdf_file(), headers=HTMX)

    assert response.status_code == 500
    assert stored_pdfs() == []


# --- Deleting -----------------------------------------------------------------------


def test_delete_removes_the_paper_its_questions_and_its_pdf(client, add_paper):
    paper_id = add_paper()
    db.add_question(paper_id, "Q", "A")

    response = client.delete(f"/papers/{paper_id}", headers=HTMX)

    assert response.status_code == 200
    assert db.get_paper(paper_id) is None
    assert db.list_questions(paper_id) == []
    assert stored_pdfs() == []
    count = parse_html(response.text).find(id="library-count")
    assert count.attrs["hx-swap-oob"] == "true"
    assert count.text == "0"


def test_delete_leaves_other_papers_alone(client, add_paper):
    doomed = add_paper(arxiv_id="2401.00001")
    kept = add_paper(arxiv_id="2401.00002")
    kept_file = pdfs.pdf_path(db.get_paper(kept)["pdf_path"])

    response = client.delete(f"/papers/{doomed}", headers=HTMX)

    assert parse_html(response.text).find(id="library-count").text == "1"
    assert db.get_paper(kept) is not None
    assert stored_pdfs() == [kept_file]


def test_deleting_twice_is_a_404_with_an_inline_message(client, add_paper):
    paper_id = add_paper()
    client.delete(f"/papers/{paper_id}", headers=HTMX)

    response = client.delete(f"/papers/{paper_id}", headers=HTMX)

    assert response.status_code == 404
    assert "already deleted" in parse_html(response.text).find(class_="alert-error").text


# --- Found by mutation testing -------------------------------------------------------------------


def test_a_save_form_with_no_abstract_saves_the_paper_without_one(client, arxiv_pdfs):
    arxiv_pdfs.reply(200, content=make_pdf())
    form = save_form()
    del form["abstract"]
    assert client.post("/library", data=form, headers=HTMX).status_code == 200
    assert only_paper()["abstract"] is None


def test_retrying_a_failed_download_sends_the_clean_id(client, arxiv_pdfs):
    arxiv_pdfs.reply(503)
    response = client.post("/library", data=save_form(arxiv_id="  https://arxiv.org/abs/2401.12345 "), headers=HTMX)
    assert response.status_code == 502
    assert hidden_fields(parse_html(response.text).find("form"))["arxiv_id"] == "2401.12345"


def test_the_form_shown_after_a_bad_id_keeps_what_was_sent_cut_to_40_characters(client):
    response = client.post("/library", data=save_form(arxiv_id="  " + "x" * 50), headers=HTMX)
    assert response.status_code == 400
    assert hidden_fields(parse_html(response.text).find("form"))["arxiv_id"] == "x" * 40


def test_the_form_shown_after_a_missing_id_stays_empty(client):
    response = client.post("/library", data={"title": PAPER_TITLE}, headers=HTMX)
    assert response.status_code == 400
    assert hidden_fields(parse_html(response.text).find("form"))["arxiv_id"] == ""
