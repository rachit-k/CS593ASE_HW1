"""Error pages and the page shell: 404s, bad requests, unexpected failures (design section 9)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from app import config, db
from app.main import app
from helpers import HTMX, parse_html


def boom(*_args, **_kwargs):
    raise RuntimeError("secret internal detail")


# --- Not found ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/no/such/page", "nothing at this address"),
        ("/papers/999", "isn't in your library"),
        ("/papers/0", "isn't in your library"),
        ("/papers/-1", "isn't in your library"),
        ("/papers/abc", "nothing at this address"),
        ("/papers/999/pdf", "isn't in your library"),
        pytest.param("/papers/" + "9" * 5000, "nothing at this address", id="/papers/absurdly-long-number"),
    ],
)
def test_unknown_pages_get_the_friendly_404_page(client, path, message):
    response = client.get(path)
    assert response.status_code == 404
    page = parse_html(response.text)
    assert page.find("h1").text == "Not found"
    assert message in page.find(class_="lede").text
    assert page.find("a", href="/library") is not None
    assert page.find("a", href="/") is not None


@pytest.mark.parametrize("paper_id", [2**63 - 1, 2**63, -(2**63) - 1], ids=["largest-valid", "too-large", "too-small"])
def test_an_impossibly_large_paper_id_is_just_not_found(client, paper_id):
    for path in (f"/papers/{paper_id}", f"/papers/{paper_id}/pdf"):
        response = client.get(path)
        assert response.status_code == 404
        assert "isn't in your library" in parse_html(response.text).find(class_="lede").text
    for method, path, form in (
        ("POST", f"/papers/{paper_id}/summary", None),
        ("POST", f"/papers/{paper_id}/questions", {"question": "Why?"}),
        ("DELETE", f"/papers/{paper_id}", None),
    ):
        response = client.request(method, path, headers=HTMX, data=form)
        assert response.status_code == 404
        assert parse_html(response.text).find(class_="alert-error") is not None


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/papers/999/summary"), ("POST", "/papers/999/questions"), ("DELETE", "/papers/999")],
)
def test_htmx_actions_on_a_missing_paper_get_an_inline_404(client, method, path):
    response = client.request(method, path, headers=HTMX, data={"question": "Why?"} if "questions" in path else None)
    assert response.status_code == 404
    assert "<html" not in response.text
    assert parse_html(response.text).find(class_="alert-error") is not None


@pytest.mark.parametrize(("method", "path"), [("POST", "/papers/abc/summary"), ("POST", "/papers/abc/questions"), ("DELETE", "/papers/abc")])
def test_htmx_actions_with_a_nonsense_id_get_an_inline_400(client, method, path):
    response = client.request(method, path, headers=HTMX)
    assert response.status_code == 400
    assert "Reload the page" in parse_html(response.text).find(class_="alert-error").text


@pytest.mark.parametrize(
    ("method", "url", "allowed"),
    [("GET", "/library/upload", {"POST"}), ("POST", "/papers/1", {"DELETE", "GET"}), ("PUT", "/library", {"GET", "POST"})],
)
def test_the_wrong_method_is_refused_and_names_one_that_works(client, method, url, allowed):
    response = client.request(method, url)
    assert response.status_code == 405
    # Starlette lists the methods of the first route that matches the path, so this can be a subset.
    listed = set(response.headers["allow"].split(", "))
    assert listed and listed <= allowed


# --- Unexpected failures ------------------------------------------------------------


def test_an_unexpected_error_shows_a_friendly_page_without_details(monkeypatch):
    monkeypatch.setattr(db, "list_papers", boom)

    response = TestClient(app, raise_server_exceptions=False).get("/library")

    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    page = parse_html(response.text)
    assert page.find("h1").text == "Something went wrong"
    assert page.find("p", class_="lede").text == "Something went wrong on the server. The terminal running the app shows the details."
    assert page.find("a", href="/library") is not None


def test_an_unexpected_error_during_an_htmx_request_is_an_inline_alert(monkeypatch, add_paper):
    paper_id = add_paper()
    monkeypatch.setattr(db, "delete_paper", boom)

    response = TestClient(app, raise_server_exceptions=False).delete(f"/papers/{paper_id}", headers=HTMX)

    assert response.status_code == 500
    assert "<html" not in response.text
    assert "secret internal detail" not in response.text
    assert parse_html(response.text).find(class_="alert-error") is not None


def test_a_broken_database_does_not_break_the_header_count(client, monkeypatch):
    def locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "count_papers", locked)
    response = client.get("/")
    assert response.status_code == 200
    assert parse_html(response.text).find(id="library-count").text == ""


# --- Startup and the page shell -----------------------------------------------------


def test_startup_creates_the_database_in_a_new_data_folder(tmp_path, monkeypatch):
    settings = replace(config.settings, data_dir=tmp_path / "brand" / "new")
    monkeypatch.setattr(config, "settings", settings)
    with TestClient(app) as client:
        assert client.get("/library").status_code == 200
    assert settings.db_path.is_file()
    assert settings.pdf_dir.is_dir()


def test_static_files_the_pages_rely_on_are_served(client):
    page = parse_html(client.get("/").text)
    assert page.find("script", src="/static/htmx.min.js") is not None
    assert page.find("link", href="/static/style.css") is not None
    htmx = client.get("/static/htmx.min.js")
    assert htmx.status_code == 200
    assert "javascript" in htmx.headers["content-type"]
    style = client.get("/static/style.css")
    assert style.status_code == 200
    assert "css" in style.headers["content-type"]


def test_htmx_is_configured_to_show_error_responses(client):
    meta = parse_html(client.get("/").text).find("meta", name="htmx-config")
    rules = {rule["code"]: rule for rule in json.loads(meta.attrs["content"])["responseHandling"]}
    assert rules["[45].."]["swap"] is True
    assert rules["[45].."]["error"] is True
    assert rules["[23].."]["swap"] is True


@pytest.mark.parametrize(("path", "current"), [("/", "Search"), ("/library", "Library")])
def test_the_header_marks_the_current_section(client, path, current):
    nav = parse_html(client.get(path).text).find("nav")
    marked = [link.text for link in nav.find_all("a") if link.attrs.get("aria-current") == "page"]
    assert len(marked) == 1
    assert marked[0].startswith(current)


def test_the_header_counts_library_papers_on_every_page(client, add_paper):
    paper_id = add_paper()
    for path in ("/", "/library", f"/papers/{paper_id}"):
        assert parse_html(client.get(path).text).find(id="library-count").text == "1"


def test_an_http_error_without_a_text_message_gets_a_generic_one(client, monkeypatch):
    def structured_error(_paper_id):
        raise HTTPException(409, {"reason": "not text"})

    monkeypatch.setattr(db, "get_paper", structured_error)
    response = client.post("/papers/1/summary", headers=HTMX)
    assert response.status_code == 409
    assert parse_html(response.text).find(class_="alert-error").text == "Something went wrong."
