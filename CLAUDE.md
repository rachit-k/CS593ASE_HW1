# AI Research Assistant

A local web app (CS59300 HW1) for searching arXiv, saving papers to a SQLite library, uploading PDFs, and using Google Gemini to summarize a paper or answer questions about it from its full text. FastAPI + Jinja2 templates + HTMX, with no frontend build step. The full design is in `docs/design.md`; read it before changing behavior.

## Commands

- Set up: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
- Run: `uvicorn app.main:app --reload`, then open http://127.0.0.1:8000
- Settings live in `.env` (copied from `.env.example`): `GEMINI_API_KEY`, `GEMINI_MODEL`, `DATA_DIR`.
- There is no test suite yet; HW2 adds one with pytest. Tests must never call the real arXiv or Gemini (see "Testing seams").

## Code map

- `app/main.py`: routes only. Check input, call a module, render a template. HTMX requests get a partial from `templates/partials/`; normal requests get a full page.
- `app/arxiv_client.py`: arXiv search (Atom feed), retries on 429/5xx, arXiv ID validation, and link building.
- `app/pdfs.py`: arXiv PDF download (arxiv.org only, 50 MB cap), upload checks, PyMuPDF text extraction, metadata fallbacks, PDF files on disk.
- `app/llm.py`: Gemini through `google-genai`: `extract_metadata`, `summarize`, `answer`. Every failure becomes `LLMError` with a message the UI shows as is.
- `app/db.py`: SQLite schema and every query. One short-lived connection per call.
- `app/config.py`: `Settings` from the environment. Other modules read `config.settings` at call time.
- `app/templates/`: pages (`search`, `library`, `paper`, `error`) and the HTMX partials.
- `app/static/`: `style.css` and vendored `htmx.min.js` (2.0.11).

## Rules to keep

- Every paper in the library has non-empty `full_text`; `db.add_paper` enforces it. Summaries and Q&A send the whole text (no RAG).
- Network calls happen only in `arxiv_client.py`, `pdfs.download_arxiv_pdf`, and `llm.py`.
- PDF downloads only ever go to arxiv.org. Links are rebuilt from the validated arXiv ID, never taken from form input.
- Model output is rendered as Markdown with raw HTML disabled (the `markdown` filter in `main.py`). Keep it that way.
- Error responses keep real 4xx/5xx status codes. The htmx config in `templates/base.html` swaps them into the page.
- Keep routes thin and dependencies few. Prefer the standard library; add no JavaScript build tooling.
- When behavior changes, update `docs/design.md` and `README.md` in the same change.

## Testing seams

- arXiv and PDF downloads: set `arxiv_client.TRANSPORT` and `pdfs.TRANSPORT` to an `httpx.MockTransport`. Set `arxiv_client.RETRY_DELAYS = (0, 0)` so retry tests don't sleep.
- Gemini: monkeypatch `llm.get_client` to return a fake client, or a real `genai.Client` built with `http_options=types.HttpOptions(httpx_client=httpx.Client(transport=...))` to exercise the SDK against canned responses.
- Data: monkeypatch `config.settings` with `config.Settings(None, "test-model", tmp_path)` and call `db.init_db()`.
- Pages: `fastapi.testclient.TestClient(app)`. Send an `HX-Request: true` header to get partials.

## Gotchas

- Never read, print, or edit `.env`; it holds the user's API key. `.claude/settings.json` blocks the file tools, but not shell commands like `cat`, so don't run those on it either.
- arXiv throttles automated clients (HTTP 429) and asks for at most one request every 3 seconds. Keep one request per search and keep the backoff.
- Gemini free-tier quotas are per model and per project. A 429 from Gemini is a quota, not a bug; `GEMINI_MODEL` is the escape hatch.
- The thinking level is only sent to `gemini-3*` models, because older models reject it.
- `data/` is the user's library. Don't delete it.
