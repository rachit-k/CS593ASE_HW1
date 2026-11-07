"""Shared test setup: each test gets an empty library in a temp folder and no way to reach a real service."""

from __future__ import annotations

import os
import socket
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import replace

# Set these before anything imports `app`. Real environment variables win over .env (see
# app/config.py), so the user's API key never enters the test process.
os.environ["GEMINI_API_KEY"] = ""
os.environ["GEMINI_MODEL"] = "test-model"
os.environ["DATA_DIR"] = os.path.join(tempfile.gettempdir(), "research-assistant-tests-unused")

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from hypothesis import settings  # noqa: E402

from app import arxiv_client, citations, config, db, llm, pdfs  # noqa: E402
from app.main import app  # noqa: E402
from helpers import (  # noqa: E402
    PAPER_ABSTRACT,
    PAPER_AUTHORS,
    PAPER_TEXT,
    PAPER_TITLE,
    FakeGemini,
    FakeService,
    make_pdf,
)


# A normal run tries the same inputs every time, so a result can't change between runs. To hunt for
# new problems, run: HYPOTHESIS_PROFILE=thorough pytest tests/test_properties.py
settings.register_profile("default", deadline=None, derandomize=True, database=None, max_examples=100)
settings.register_profile("mutation", deadline=None, derandomize=True, database=None, max_examples=15)
settings.register_profile("thorough", deadline=None, database=None, max_examples=2000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def _refuse_connection(*args: object, **kwargs: object) -> None:
    raise AssertionError(f"A test tried to open a real network connection: {args[1:] or args}")


def _unexpected(service: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"A test made an unexpected {service} request: {request.method} {request.url}")

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch) -> None:
    """An empty library in a temp folder, no real network, and no service unless a test fakes it."""
    monkeypatch.setattr(config, "settings", config.Settings(None, "test-model", tmp_path / "data"))
    db.init_db()
    monkeypatch.setattr(socket.socket, "connect", _refuse_connection)
    monkeypatch.setattr(socket, "create_connection", _refuse_connection)
    monkeypatch.setattr(arxiv_client, "TRANSPORT", _unexpected("arXiv"))
    monkeypatch.setattr(pdfs, "TRANSPORT", _unexpected("arXiv PDF"))
    monkeypatch.setattr(citations, "TRANSPORT", _unexpected("Semantic Scholar"))
    monkeypatch.setattr(llm, "_clients", {})


@pytest.fixture
def arxiv(monkeypatch) -> FakeService:
    """A fake arXiv search API, without the pauses between retries."""
    fake = FakeService()
    monkeypatch.setattr(arxiv_client, "TRANSPORT", httpx.MockTransport(fake))
    monkeypatch.setattr(arxiv_client, "RETRY_DELAYS", (0, 0))
    return fake


@pytest.fixture
def arxiv_pdfs(monkeypatch) -> FakeService:
    """A fake arxiv.org for PDF downloads."""
    fake = FakeService()
    monkeypatch.setattr(pdfs, "TRANSPORT", httpx.MockTransport(fake))
    return fake


@pytest.fixture
def scholar(monkeypatch) -> FakeService:
    """A fake Semantic Scholar batch API, without the pauses between retries."""
    fake = FakeService()
    monkeypatch.setattr(citations, "TRANSPORT", httpx.MockTransport(fake))
    monkeypatch.setattr(citations, "RETRY_DELAYS", (0, 0))
    return fake


@pytest.fixture
def gemini(monkeypatch) -> FakeGemini:
    """A fake Gemini endpoint behind the real SDK client, with a (fake) key configured."""
    fake = FakeGemini()
    http_client = httpx.Client(transport=httpx.MockTransport(fake))
    sdk_client = genai.Client(api_key="test-key", http_options=types.HttpOptions(httpx_client=http_client))
    monkeypatch.setattr(llm, "get_client", lambda: sdk_client)
    monkeypatch.setattr(config, "settings", replace(config.settings, gemini_api_key="test-key"))
    return fake


@pytest.fixture
def use_model(monkeypatch) -> Callable[[str], None]:
    """Switch GEMINI_MODEL for one test."""

    def use(name: str) -> None:
        monkeypatch.setattr(config, "settings", replace(config.settings, gemini_model=name))

    return use


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Requests against the app. Startup runs, and server exceptions fail the test."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def add_paper() -> Callable[..., int]:
    """Put a paper straight into the library, with its PDF on disk. Returns its id."""

    def add(**overrides: object) -> int:
        fields = {
            "source": "arxiv",
            "arxiv_id": "2401.12345",
            "title": PAPER_TITLE,
            "authors": PAPER_AUTHORS,
            "year": 2024,
            "abstract": PAPER_ABSTRACT,
            "url": "https://arxiv.org/abs/2401.12345",
            "full_text": PAPER_TEXT,
        } | overrides
        fields.setdefault("pdf_path", pdfs.save_pdf(make_pdf()))
        return db.add_paper(**fields)

    return add
