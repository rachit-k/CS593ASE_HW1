"""PDF handling: arXiv downloads, upload checks, text extraction, and files on disk."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pymupdf  # PyMuPDF

from app import arxiv_client, config

MAX_PDF_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30
MIN_TEXT_CHARS = 200  # fewer non-whitespace characters than this means a scanned/image-only PDF
METADATA_PAGES = 2
METADATA_TEXT_CHARS = 8_000
MAX_FALLBACK_ABSTRACT_CHARS = 3_000

# Tests can set this to an httpx.MockTransport to fake arXiv without a network.
TRANSPORT: httpx.BaseTransport | None = None


class PDFError(Exception):
    """A PDF couldn't be downloaded, opened, or used. The message is shown to the user."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ExtractedPDF:
    full_text: str
    first_pages_text: str  # text of the first pages, for metadata extraction
    embedded_title: str | None = None  # from the PDF's own metadata, if it looks real
    embedded_authors: list[str] = field(default_factory=list)


# --- Downloading -----------------------------------------------------------------


def _only_arxiv(request: httpx.Request) -> None:
    """Block any request (including redirects) that would leave arxiv.org."""
    host = request.url.host
    if host != "arxiv.org" and not host.endswith(".arxiv.org"):
        raise PDFError("arXiv redirected the download to another site, so it was stopped.", 502)


def download_arxiv_pdf(arxiv_id: str) -> bytes:
    """Download a paper's PDF from arxiv.org (50 MB cap). Raises PDFError on any problem."""
    url = arxiv_client.pdf_url(arxiv_id)
    chunks: list[bytes] = []
    size = 0
    try:
        with httpx.Client(
            transport=TRANSPORT,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            headers={"User-Agent": arxiv_client.USER_AGENT},
            follow_redirects=True,
            event_hooks={"request": [_only_arxiv]},
        ) as client, client.stream("GET", url) as response:
            if response.status_code != 200:
                raise PDFError(
                    f"Couldn't download the PDF from arXiv (HTTP {response.status_code}). "
                    "Try again in a moment.",
                    502,
                )
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_PDF_BYTES:
                    raise PDFError("The PDF is larger than 50 MB, so it wasn't saved.", 502)
                chunks.append(chunk)
    except httpx.TimeoutException as exc:
        raise PDFError("Downloading the PDF from arXiv timed out. Try again in a moment.", 502) from exc
    except httpx.HTTPError as exc:
        raise PDFError("Couldn't reach arXiv to download the PDF. Try again in a moment.", 502) from exc
    data = b"".join(chunks)
    if not looks_like_pdf(data):
        raise PDFError("arXiv didn't return a PDF for this paper. Try again later.", 502)
    return data


# --- Reading -----------------------------------------------------------------------


def looks_like_pdf(data: bytes) -> bool:
    """PDF files start with "%PDF" (allowing a little junk before it, as readers do)."""
    return b"%PDF" in data[:1024]


def extract(data: bytes) -> ExtractedPDF:
    """Extract text and embedded metadata. Raises PDFError for unusable files."""
    if not looks_like_pdf(data):
        raise PDFError("That file isn't a PDF.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for damaged files
        raise PDFError("That PDF couldn't be opened. It may be damaged.") from exc
    with doc:
        if doc.needs_pass:
            raise PDFError("That PDF is password-protected. Remove the password and upload it again.")
        try:
            pages = [page.get_text("text") for page in doc]
        except Exception as exc:
            raise PDFError("The text in that PDF couldn't be read. It may be damaged.") from exc
        metadata = doc.metadata or {}

    full_text = _tidy("\n\n".join(pages))
    if len(re.sub(r"\s", "", full_text)) < MIN_TEXT_CHARS:
        raise PDFError(
            "That PDF has no readable text (it looks like a scan), and OCR isn't supported. "
            "Try a PDF where you can select the text."
        )
    return ExtractedPDF(
        full_text=full_text,
        first_pages_text=_tidy("\n\n".join(pages[:METADATA_PAGES]))[:METADATA_TEXT_CHARS],
        embedded_title=_embedded_title(metadata.get("title")),
        embedded_authors=_embedded_authors(metadata.get("author")),
    )


def _tidy(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_PLACEHOLDER_TITLES = {"untitled", "title", "paper", "document", "manuscript", "main", "article"}


def _embedded_title(value: str | None) -> str | None:
    """The PDF's own title field, unless it is empty or an obvious placeholder."""
    title = " ".join((value or "").split())
    lowered = title.lower()
    if (
        len(title) < 8
        or lowered in _PLACEHOLDER_TITLES
        or lowered.startswith("microsoft word")
        or re.search(r"\.(pdf|docx?|tex|dvi|ps)$", lowered)
    ):
        return None
    return title


def _embedded_authors(value: str | None) -> list[str]:
    text = " ".join((value or "").split())
    if not text:
        return []
    parts = text.split(";") if ";" in text else re.split(r",|\band\b", text)
    return [part.strip() for part in parts if len(part.strip()) > 1]


def title_from_text(text: str) -> str | None:
    """Last-resort title: the first line that looks like prose, not an arXiv stamp or page number."""
    for line in text.split("\n"):
        line = " ".join(line.split())
        if len(line) < 8 or line.lower().startswith("arxiv:") or not re.search(r"[A-Za-z]{3}", line):
            continue
        return line[:300]
    return None


_ABSTRACT_RE = re.compile(
    r"\babstract\b[\s:.\-—–]*(.+?)"
    r"(?=\n\s*(?:\d+\.?|[IVX]+\.)?\s*(?:introduction|keywords|index terms|ccs concepts)\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def abstract_from_text(text: str) -> str | None:
    """Fallback abstract: the text after an "Abstract" heading, up to the introduction."""
    match = _ABSTRACT_RE.search(text)
    if not match:
        return None
    abstract = " ".join(match.group(1).split())
    if len(abstract) > MAX_FALLBACK_ABSTRACT_CHARS:
        cut = abstract[:MAX_FALLBACK_ABSTRACT_CHARS]
        abstract = cut[: cut.rfind(". ") + 1] or cut
    return abstract if len(abstract) >= 100 else None


# --- Files on disk ---------------------------------------------------------------


def save_pdf(data: bytes) -> str:
    """Store a PDF under DATA_DIR/pdfs with a generated name, and return that name."""
    name = f"{uuid.uuid4().hex}.pdf"
    config.settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    (config.settings.pdf_dir / name).write_bytes(data)
    return name


def pdf_path(name: str) -> Path:
    return config.settings.pdf_dir / Path(name).name


def delete_pdf(name: str) -> None:
    try:
        pdf_path(name).unlink(missing_ok=True)
    except OSError:
        pass  # a leftover file is harmless
