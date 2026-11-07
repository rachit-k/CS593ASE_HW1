# AI Research Assistant

A local web app (CS59300 HW1) for searching arXiv, saving papers to a SQLite library, uploading PDFs, and using Google Gemini to summarize a paper or answer questions about it from its full text. FastAPI + Jinja2 templates + HTMX, with no frontend build step. The full design is in `docs/design.md`; read it before changing behavior.

## Commands

- Set up: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
- Run: `uvicorn app.main:app --reload`, then open http://127.0.0.1:8000
- Check a change in the real app: use the `run-app` skill (`.claude/skills/run-app/`). It runs a throwaway server on port 8765, because the user's own server is usually on 8000.
- Settings live in `.env` (copied from `.env.example`): `GEMINI_API_KEY`, `GEMINI_MODEL`, `DATA_DIR`.
- Test: `pip install -r requirements-dev.txt`, then `pytest`, `pytest -m "not browser"` without the browser tests, or `pytest --cov=app --cov-branch --cov-report=term-missing --cov-fail-under=100` for coverage, which should stay at 100% with no `pragma: no cover`. Tests must never call the real arXiv or Gemini (see "Testing seams").
- Beyond coverage: `tools/path_coverage.py` (conditions, operands, template branches) and `tools/mutation_test.py` (small code changes no test notices; about half an hour). Both work on scratch copies.
- `pytest` should report no failures.

## Code map

- `app/main.py`: routes only. Check input, call a module, render a template. HTMX requests get a partial from `templates/partials/`; normal requests get a full page.
- `app/arxiv_client.py`: arXiv search (Atom feed) by relevance or submission date, retries on 429/5xx, arXiv ID validation, and link building.
- `app/citations.py`: Semantic Scholar citation counts (one batch request, no key) and `most_cited(results, limit)`, which ranks results by them. For the "most cited" sort, the search route passes it arXiv's best `POOL_SIZE` matches.
- `app/pdfs.py`: arXiv PDF download (arxiv.org only, 50 MB cap), upload checks, PyMuPDF text extraction, metadata fallbacks, PDF files on disk.
- `app/llm.py`: Gemini through `google-genai`: `extract_metadata`, `summarize`, `answer`. Every failure becomes `LLMError` with a message the UI shows as is.
- `app/db.py`: SQLite schema and every query. One short-lived connection per call.
- `app/config.py`: `Settings` from the environment. Other modules read `config.settings` at call time.
- `app/templates/`: pages (`search`, `library`, `paper`, `error`) and the HTMX partials.
- `app/static/`: `style.css` and vendored `htmx.min.js` (2.0.11).

## Rules to keep

- Every paper in the library has non-empty `full_text`; `db.add_paper` enforces it. Summaries and Q&A send the whole text (no RAG).
- Network calls happen only in `arxiv_client.py`, `citations.py`, `pdfs.download_arxiv_pdf`, and `llm.py`.
- PDF downloads only ever go to arxiv.org. Links are rebuilt from the validated arXiv ID, never taken from form input.
- Model output is rendered as Markdown with raw HTML and images disabled (the `markdown` filter in `main.py`). Keep it that way.
- Error responses keep real 4xx/5xx status codes. The htmx config in `templates/base.html` swaps them into the page.
- Keep routes thin and dependencies few. Prefer the standard library; add no JavaScript build tooling.
- When behavior changes, update `docs/design.md` and `README.md` in the same change.

## Testing seams

`tests/conftest.py` already provides these as fixtures (`arxiv`, `arxiv_pdfs`, `scholar`, `gemini`, `use_model`, `client`, `add_paper`), and an autouse fixture gives every test an empty temp library and refuses real connections. `tests/recordings/` holds real API responses and real papers to replay, and `tests/helpers.py` builds PDFs and feeds and parses pages. `FakeGemini.reply` checks JSON replies against the SDK's schema, so a hand-written reply can't drift from the real format.

- arXiv and PDF downloads: set `arxiv_client.TRANSPORT` and `pdfs.TRANSPORT` to an `httpx.MockTransport`. Set `arxiv_client.RETRY_DELAYS = (0, 0)` so retry tests don't sleep.
- Citation counts: set `citations.TRANSPORT` to an `httpx.MockTransport` that answers the batch POST with one entry per ID (`null` for unknown), and `citations.RETRY_DELAYS = (0, 0)`.
- Gemini: monkeypatch `llm.get_client` to return a fake client, or a real `genai.Client` built with `http_options=types.HttpOptions(httpx_client=httpx.Client(transport=...))` to exercise the SDK against canned responses.
- Data: monkeypatch `config.settings` with `config.Settings(None, "test-model", tmp_path)` and call `db.init_db()`.
- Pages: `fastapi.testclient.TestClient(app)`. Send an `HX-Request: true` header to get partials.

## Gotchas

- Never read, print, or edit `.env`; it holds the user's API key. `.claude/settings.json` blocks the file tools, but not shell commands like `cat`, so don't run those on it either.
- arXiv throttles automated clients (HTTP 429) and asks for at most one request every 3 seconds. Keep one request per search and keep the backoff.
- arXiv doesn't index stop words (`arxiv_client.STOP_WORDS`) in titles and abstracts, so `build_params` must skip them; a required `all:a` silently hides most papers. arXiv also indexes "don't" as one word, so apostrophes stay inside words.
- htmx's back-button snapshots don't include typed or picked values. A `htmx:historyRestore` handler in `base.html` sets the search form from the URL, so any new search control needs a URL parameter and a line there, or Back shows it out of sync with the results. `tests/test_browser.py` checks this. Don't copy values into attributes after a swap instead: htmx can take its snapshot first.
- Semantic Scholar's no-key rate limit is shared by everyone, so a 429 there is load, not a bug. Only the "most cited" sort depends on it.
- Gemini free-tier quotas are per model and per project. A 429 from Gemini is a quota, not a bug; `GEMINI_MODEL` is the escape hatch.
- The thinking level is only sent to `gemini-3*` models, because older models reject it.
- `data/` is the user's library. Don't delete it.
- `tests/test_contracts.py` compares the fakes with recordings and the SDK's schema, and the code with its docs and these rules. A new route needs a row in the design doc's route table.
- `.gitignore`'s `data/` matches a folder named `data` anywhere in the repo, so a test fixture folder with that name would silently never be committed.
- `feedparser.parse` opens a string or bytes that look like a file path or URL, so `parse_feed` passes it a stream. Keep it that way.
- The upload fallbacks are heuristics, and synthetic tests alone have missed real cases. `tests/test_real_papers.py` replays 29 real papers with arXiv's metadata as the right answer; add a paper to `tests/recordings/real_papers.json` when a new layout turns up. On this machine, `hw2-report/tools/check_real_pdfs.py` also checks freshly downloaded PDFs.
