"""Builders and fakes shared by the tests. Nothing here touches the network or the user's files."""

from __future__ import annotations

import json
import textwrap
from collections import deque
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx
import pymupdf
from google.genai import types

from app import config

HTMX = {"HX-Request": "true"}
RECORDINGS = Path(__file__).parent / "recordings"


def recorded(name: str) -> str:
    """A real API response, recorded once (see tests/recordings/README.md)."""
    return (RECORDINGS / name).read_text()


def stored_pdfs() -> list[Path]:
    """The PDF files currently in the test library."""
    return sorted(config.settings.pdf_dir.iterdir())


# --- A sample paper ----------------------------------------------------------------

PAPER_TITLE = "Testing Web Applications with Coding Agents"
PAPER_AUTHORS = ["Alice Smith", "Bob Jones"]
PAPER_ABSTRACT = (
    "We study how coding agents generate tests for small web applications. We measure coverage, "
    "correctness, and fault detection on twelve projects, and we report where agents succeed and "
    "where they still need feedback from people."
)
# Appears only on the last page, so a prompt that contains it was built from the full text.
BODY_FACT = "The evaluation used 4217 programs from the Zephyr benchmark."
PAPER_PAGES = (
    f"{PAPER_TITLE}\nAlice Smith and Bob Jones\nAbstract\n{PAPER_ABSTRACT}\n"
    "1 Introduction\nCoding agents now write tests for real projects. This paper asks how good those tests are.",
    "2 Method\nWe ran each agent three times on every project and kept all of the generated tests.",
    f"3 Results\n{BODY_FACT}\n4 Conclusion\nAgents write useful tests but need feedback to find real bugs.",
)
PAPER_TEXT = "\n\n".join(PAPER_PAGES)


# --- PDFs --------------------------------------------------------------------------


def make_pdf(
    pages: Iterable[str] = PAPER_PAGES,
    *,
    title: str | None = None,
    author: str | None = None,
    password: str | None = None,
) -> bytes:
    """A real PDF with selectable text, one page per string."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        y = 60
        for paragraph in text.split("\n"):
            for line in textwrap.wrap(paragraph, 85) or [""]:
                page.insert_text((50, y), line, fontsize=10)
                y += 13
    if title is not None or author is not None:
        doc.set_metadata({"title": title or "", "author": author or ""})
    options: dict[str, Any] = {}
    if password:
        options = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": password, "owner_pw": f"owner-{password}"}
    data = doc.tobytes(**options)
    doc.close()
    return data


def make_scanned_pdf() -> bytes:
    """A PDF whose page is only a picture, with no selectable text, like a scan."""
    doc = pymupdf.open()
    page = doc.new_page()
    picture = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 100, 100))
    picture.clear_with(128)
    page.insert_image(page.rect, pixmap=picture)
    data = doc.tobytes()
    doc.close()
    return data


# --- arXiv feeds -------------------------------------------------------------------


def atom_entry(
    arxiv_id: str = "2401.12345",
    *,
    version: int = 1,
    title: str = "A Study of Software Testing",
    authors: Iterable[str] = ("Alice Smith", "Bob Jones"),
    abstract: str = "We study software testing.",
    published: str | None = "2024-01-15T10:00:00Z",
) -> str:
    """One <entry> shaped like the real ones in tests/recordings/arxiv_three_papers.xml."""
    people = "".join(f"<author><name>{escape(name)}</name></author>" for name in authors)
    date = f"<published>{published}</published>" if published else ""
    return (
        f"<entry><id>http://arxiv.org/abs/{arxiv_id}v{version}</id><title>{escape(title)}</title>"
        f"<updated>2025-06-01T10:00:00Z</updated><summary>{escape(abstract)}</summary>{date}{people}</entry>"
    )


def atom_feed(*entries: str) -> str:
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        '<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://www.w3.org/2005/Atom">'
        "<id>https://arxiv.org/api/test</id><title>arXiv Query</title><updated>2026-11-06T14:16:31Z</updated>"
        f"<opensearch:totalResults>{len(entries)}</opensearch:totalResults>{''.join(entries)}</feed>"
    )


# --- Fake web services -------------------------------------------------------------


class FakeService:
    """Stands in for a web service: records each request and plays back queued replies.

    The last reply repeats, so one `reply` covers any number of requests. Pass an instance
    to httpx.MockTransport.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._replies: deque[Callable[[httpx.Request], httpx.Response]] = deque()

    def reset(self) -> FakeService:
        self.requests.clear()
        self._replies.clear()
        return self

    def reply(self, status: int = 200, **response_options: Any) -> FakeService:
        self._replies.append(lambda _request: httpx.Response(status, **response_options))
        return self

    def fail(self, error: Exception) -> FakeService:
        def raise_error(_request: httpx.Request) -> httpx.Response:
            raise error

        self._replies.append(raise_error)
        return self

    def reply_with(self, handler: Callable[[httpx.Request], httpx.Response]) -> FakeService:
        self._replies.append(handler)
        return self

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(f"No reply queued for {request.method} {request.url}")
        handler = self._replies.popleft() if len(self._replies) > 1 else self._replies[0]
        return handler(request)


class FakeGemini(FakeService):
    """A fake Gemini REST endpoint. The real google-genai SDK talks to it, so the SDK runs too."""

    def reply(self, status: int = 200, **response_options: Any) -> FakeGemini:
        # The SDK's schema forbids unknown fields, so a hand-written reply can't drift from the real format.
        if status == 200 and "json" in response_options:
            types.GenerateContentResponse.model_validate(response_options["json"])
        return super().reply(status, **response_options)

    def reply_text(self, text: str, finish_reason: str = "STOP") -> FakeGemini:
        candidate = {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish_reason}
        return self.reply(200, json={"candidates": [candidate]})

    def reply_error(self, code: int, message: str, status: str = "ERROR") -> FakeGemini:
        return self.reply(code, json={"error": {"code": code, "message": message, "status": status}})

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    def prompt(self, index: int = -1) -> str:
        return self.body(index)["contents"][0]["parts"][0]["text"]

    def system(self, index: int = -1) -> str:
        return self.body(index)["systemInstruction"]["parts"][0]["text"]

    def generation_config(self, index: int = -1) -> dict[str, Any]:
        return self.body(index).get("generationConfig", {})


# --- Reading HTML ------------------------------------------------------------------

_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Node:
    """One element of a parsed page, so tests can check structure instead of matching strings."""

    def __init__(self, tag: str, attrs: list[tuple[str, str | None]] | None = None, parent: Node | None = None):
        self.tag = tag
        self.attrs = dict(attrs or [])
        self.parent = parent
        self.children: list[Node | str] = []

    @property
    def text(self) -> str:
        parts = (child if isinstance(child, str) else child.text for child in self.children)
        return " ".join(" ".join(parts).split())

    def _matches(self, tag: str | None, class_: str | None, attrs: dict[str, str]) -> bool:
        return (
            (tag is None or self.tag == tag)
            and (class_ is None or class_ in (self.attrs.get("class") or "").split())
            and all(self.attrs.get(name) == value for name, value in attrs.items())
        )

    def find_all(self, tag: str | None = None, class_: str | None = None, **attrs: str) -> list[Node]:
        found: list[Node] = []
        for child in self.children:
            if isinstance(child, Node):
                if child._matches(tag, class_, attrs):
                    found.append(child)
                found.extend(child.find_all(tag, class_, **attrs))
        return found

    def find(self, tag: str | None = None, class_: str | None = None, **attrs: str) -> Node | None:
        matches = self.find_all(tag, class_, **attrs)
        return matches[0] if matches else None


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.root = self.current = Node("[document]")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, attrs, self.current)
        self.current.children.append(node)
        if tag not in _VOID_TAGS:
            self.current = node

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.current.children.append(Node(tag, attrs, self.current))

    def handle_endtag(self, tag: str) -> None:
        node: Node | None = self.current
        while node is not None and node.tag != tag:
            node = node.parent
        if node is not None and node.parent is not None:
            self.current = node.parent

    def handle_data(self, data: str) -> None:
        self.current.children.append(data)


def parse_html(html: str) -> Node:
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    return builder.root


def hidden_fields(form: Node) -> dict[str, str]:
    """What a browser would submit for the form's hidden inputs."""
    return {field.attrs["name"]: field.attrs.get("value") or "" for field in form.find_all("input", type="hidden")}
