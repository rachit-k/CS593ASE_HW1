"""FastAPI app: HTTP routes and HTML rendering.

Routes stay thin. They check input, call the modules that do the work
(arxiv_client, pdfs, llm, db), and render a Jinja2 template. Requests made by
HTMX get only the fragment that changed (a template in partials/); normal
browser requests get a full page.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup
from starlette.exceptions import HTTPException

from app import arxiv_client, citations, config, db, llm, pdfs

APP_DIR = Path(__file__).resolve().parent
log = logging.getLogger("app")

# The search page's sort menu: ?sort= value -> label after "Sort: ".
SORT_OPTIONS = {"relevance": "arXiv rank", "recent": "most recent", "citations": "most cited"}
MAX_QUESTION_CHARS = 1_000
STARTER_QUESTIONS = [
    "What problem does this paper address?",
    "What is the main idea of the proposed approach?",
    "What datasets are used?",
    "What are the major limitations?",
    "How does this method compare with the baselines?",
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="AI Research Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


# --- Template helpers --------------------------------------------------------------

# Model output is Markdown. Raw HTML inside it is escaped, never rendered.
_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")


def render_markdown(text: str | None) -> Markup:
    return Markup(_markdown.render(text or ""))


def format_authors(authors: list[str], limit: int = 4) -> str:
    """"A, B, C, and 7 more" for long author lists."""
    if not authors:
        return "Authors unknown"
    if len(authors) <= limit:
        return ", ".join(authors)
    shown = limit - 1
    return f"{', '.join(authors[:shown])}, and {len(authors) - shown} more"


def library_count() -> int | str:
    try:
        return db.count_papers()
    except sqlite3.Error:
        return ""


templates.env.filters["markdown"] = render_markdown
templates.env.filters["author_list"] = format_authors
templates.env.globals["library_count"] = library_count


def render(
    request: Request,
    template: str,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    **context: Any,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, template, context, status_code=status_code, headers=headers
    )


def is_htmx(request: Request) -> bool:
    """True for requests made by HTMX, except history restores (those need a full page)."""
    return (
        request.headers.get("HX-Request") == "true"
        and request.headers.get("HX-History-Restore-Request") != "true"
    )


def get_paper_or_404(paper_id: int) -> dict[str, Any]:
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(404, "That paper isn't in your library. It may have been deleted.")
    return paper


# --- Search ------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def search_page(request: Request, q: str = "", sort: str = "relevance") -> HTMLResponse:
    q = q.strip()[:300]
    sort = sort if sort in SORT_OPTIONS else "relevance"
    context: dict[str, Any] = {
        "active": "search",
        "q": q,
        "sort": sort,
        "sort_options": SORT_OPTIONS,
        "results": None,
        "error": None,
        "in_library": {},
        "citation_counts": {},
        "pool_size": citations.POOL_SIZE,
        "pool_full": False,
    }
    status_code = 200
    if q:
        try:
            if sort == "citations":
                # arXiv can't sort by citations, so rank its best matches by Semantic Scholar's counts.
                pool = arxiv_client.search(q, max_results=citations.POOL_SIZE)
                results, context["citation_counts"] = citations.most_cited(pool, arxiv_client.MAX_RESULTS)
                context["pool_full"] = len(pool) == citations.POOL_SIZE
            else:
                results = arxiv_client.search(q, sort=sort)
            context["results"] = results
            context["in_library"] = db.library_ids_for([r.arxiv_id for r in results])
        except (arxiv_client.ArxivError, citations.CitationError) as exc:
            context["error"], status_code = str(exc), exc.status_code
        except ValueError as exc:  # nothing searchable was left after cleaning the query
            context["error"], status_code = str(exc), 400
    template = "partials/results.html" if is_htmx(request) else "search.html"
    return render(request, template, status_code=status_code, **context)


# --- Library -----------------------------------------------------------------------


def _parse_authors(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [" ".join(str(name).split()) for name in value if str(name).strip()][:200]


def _parse_year(raw: str) -> int | None:
    raw = raw.strip()
    return int(raw) if raw.isdigit() and 1900 <= int(raw) <= 2100 else None


@app.post("/library", response_class=HTMLResponse)
def save_paper(
    request: Request,
    arxiv_id: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "",
    authors: Annotated[str, Form()] = "[]",
    year: Annotated[str, Form()] = "",
    abstract: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Save an arXiv search result: download its PDF, extract the text, store it."""
    clean_id = arxiv_client.normalize_id(arxiv_id)
    # Links are rebuilt from the validated ID; only the descriptive fields come from the form.
    result = arxiv_client.ArxivResult(
        arxiv_id=clean_id or arxiv_id.strip()[:40],
        title=" ".join(title.split())[:500],
        authors=_parse_authors(authors),
        year=_parse_year(year),
        abstract=" ".join(abstract.split())[:10_000],
        url=arxiv_client.abs_url(clean_id or ""),
        pdf_url=arxiv_client.pdf_url(clean_id or ""),
    )

    def slot(status_code: int = 200, **extra: Any) -> HTMLResponse:
        return render(request, "partials/save_slot.html", status_code=status_code, result=result, **extra)

    if clean_id is None or not result.title:
        return slot(400, error="That doesn't look like an arXiv paper, so it wasn't saved.")
    existing = db.find_by_arxiv_id(clean_id)
    if existing is not None:
        return slot(paper_id=existing)

    try:
        data = pdfs.download_arxiv_pdf(clean_id)
        extracted = pdfs.extract(data)
    except pdfs.PDFError as exc:
        return slot(exc.status_code, error=str(exc))

    file_name = pdfs.save_pdf(data)
    try:
        paper_id = db.add_paper(
            source="arxiv",
            arxiv_id=clean_id,
            title=result.title,
            authors=result.authors,
            year=result.year,
            abstract=result.abstract or None,
            url=arxiv_client.abs_url(clean_id),
            pdf_path=file_name,
            full_text=extracted.full_text,
        )
    except sqlite3.IntegrityError:
        # A parallel request (a double click) saved it first; keep that copy.
        pdfs.delete_pdf(file_name)
        paper_id = db.find_by_arxiv_id(clean_id)
    return slot(paper_id=paper_id, saved_now=True)


@app.get("/library", response_class=HTMLResponse)
def library_page(request: Request) -> HTMLResponse:
    return render(request, "library.html", active="library", papers=db.list_papers())


def upload_metadata(extracted: pdfs.ExtractedPDF) -> tuple[dict[str, Any], bool]:
    """Title, authors, year, and abstract for an uploaded PDF, plus whether fallbacks were used.

    Gemini reads them from the first pages. Without Gemini (no key, quota spent,
    or an error), they come from the PDF's own metadata and simple text rules.
    """
    text = extracted.first_pages_text
    if config.settings.gemini_api_key:
        try:
            meta = llm.extract_metadata(text)
            abstract = meta.abstract or pdfs.abstract_from_text(text)
            return {"title": meta.title, "authors": meta.authors, "year": meta.year, "abstract": abstract}, False
        except llm.LLMError as exc:
            log.warning("Gemini couldn't extract upload metadata, using fallbacks: %s", exc)
    fallback = {
        "title": extracted.embedded_title or pdfs.title_from_text(text) or "Untitled paper",
        "authors": extracted.embedded_authors,
        "year": None,
        "abstract": pdfs.abstract_from_text(text),
    }
    return fallback, True


@app.post("/library/upload")
def upload_paper(
    request: Request, file: Annotated[UploadFile | None, File()] = None
) -> Response:
    """Add an uploaded PDF to the library, then go to its page."""

    def error(message: str, status_code: int) -> HTMLResponse:
        return render(request, "partials/alert.html", status_code=status_code, message=message, kind="error")

    data = file.file.read(pdfs.MAX_PDF_BYTES + 1) if file is not None else b""
    if not data:
        return error("Choose a PDF file to upload.", 400)
    if len(data) > pdfs.MAX_PDF_BYTES:
        return error("That file is larger than 50 MB. Upload a smaller PDF.", 413)
    try:
        extracted = pdfs.extract(data)
    except pdfs.PDFError as exc:
        return error(str(exc), exc.status_code)

    meta, used_fallback = upload_metadata(extracted)
    file_name = pdfs.save_pdf(data)
    try:
        paper_id = db.add_paper(
            source="upload",
            title=meta["title"],
            authors=meta["authors"],
            year=meta["year"],
            abstract=meta["abstract"],
            url=None,
            pdf_path=file_name,
            full_text=extracted.full_text,
        )
    except Exception:
        pdfs.delete_pdf(file_name)
        raise

    target = f"/papers/{paper_id}" + ("?notice=metadata-fallback" if used_fallback else "")
    if is_htmx(request):
        return Response(headers={"HX-Redirect": target})
    return RedirectResponse(target, status_code=303)


@app.delete("/papers/{paper_id}", response_class=HTMLResponse)
def delete_paper(request: Request, paper_id: int) -> HTMLResponse:
    """Delete a paper, its questions, and its PDF. The response updates the nav count."""
    file_name = db.delete_paper(paper_id)
    if file_name is None:
        raise HTTPException(404, "That paper was already deleted.")
    pdfs.delete_pdf(file_name)
    return render(request, "partials/library_count.html", oob=True)


# --- One paper ---------------------------------------------------------------------


@app.get("/papers/{paper_id}", response_class=HTMLResponse)
def paper_page(request: Request, paper_id: int, notice: str = "") -> HTMLResponse:
    paper = get_paper_or_404(paper_id)
    return render(
        request,
        "paper.html",
        active="library",
        paper=paper,
        questions=db.list_questions(paper_id),
        starter_questions=STARTER_QUESTIONS,
        gemini_ready=bool(config.settings.gemini_api_key),
        truncated=len(paper["full_text"]) > llm.MAX_PAPER_CHARS,
        notice=notice,
    )


def _download_name(title: str) -> str:
    name = re.sub(r"[^\w\- ]+", "", title).strip()[:80].strip()
    return f"{name or 'paper'}.pdf"


@app.get("/papers/{paper_id}/pdf")
def paper_pdf(paper_id: int) -> FileResponse:
    paper = get_paper_or_404(paper_id)
    path = pdfs.pdf_path(paper["pdf_path"])
    if not path.is_file():
        raise HTTPException(404, "This paper's PDF file is missing from the data folder.")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=_download_name(paper["title"]),
        content_disposition_type="inline",
    )


@app.post("/papers/{paper_id}/summary", response_class=HTMLResponse)
def summarize_paper(request: Request, paper_id: int) -> HTMLResponse:
    """Generate (or regenerate) the paper's summary and store it."""
    paper = get_paper_or_404(paper_id)
    try:
        result = llm.summarize(paper)
    except llm.LLMError as exc:
        return render(request, "partials/summary.html", status_code=exc.status_code, paper=paper, error=str(exc))
    db.set_summary(paper_id, result.text)
    paper["summary"] = result.text
    return render(request, "partials/summary.html", paper=paper)


@app.post("/papers/{paper_id}/questions", response_class=HTMLResponse)
def ask_question(
    request: Request, paper_id: int, question: Annotated[str, Form()] = ""
) -> HTMLResponse:
    """Answer a question about the paper and store the pair."""
    paper = get_paper_or_404(paper_id)

    def error(message: str, status_code: int) -> HTMLResponse:
        # Errors go to the error area under the form, not into the Q&A list.
        return render(
            request,
            "partials/alert.html",
            status_code=status_code,
            headers={"HX-Retarget": "#qa-error", "HX-Reswap": "innerHTML"},
            message=message,
            kind="error",
        )

    question = " ".join(question.split())
    if not question:
        return error("Type a question first.", 400)
    if len(question) > MAX_QUESTION_CHARS:
        return error("Keep questions under 1,000 characters.", 400)
    try:
        result = llm.answer(paper, question, db.list_questions(paper_id))
    except llm.LLMError as exc:
        return error(str(exc), exc.status_code)
    item = db.add_question(paper_id, question, result.text)
    return render(request, "partials/qa_item.html", item=item, clear_error=True)


# --- Errors ------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def handle_http_error(request: Request, exc: HTTPException) -> Response:
    message = exc.detail if isinstance(exc.detail, str) else "Something went wrong."
    if exc.status_code == 404 and message == "Not Found":
        message = "There's nothing at this address."
    if is_htmx(request):
        return render(request, "partials/alert.html", status_code=exc.status_code, message=message, kind="error")
    if request.method == "GET" and exc.status_code == 404:
        return render(request, "error.html", status_code=404, heading="Not found", message=message)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def handle_bad_request(request: Request, exc: RequestValidationError) -> Response:
    # For example /papers/abc: the id isn't a number, so no such paper exists.
    if request.method == "GET":
        return await handle_http_error(request, HTTPException(404, "There's nothing at this address."))
    return await handle_http_error(
        request, HTTPException(400, "That request was missing something. Reload the page and try again.")
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> Response:
    # The traceback is still printed in the terminal running the app.
    message = "Something went wrong on the server. The terminal running the app shows the details."
    if is_htmx(request):
        return render(request, "partials/alert.html", status_code=500, message=message, kind="error")
    return render(request, "error.html", status_code=500, heading="Something went wrong", message=message)
