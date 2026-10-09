# AI Research Assistant — Design

Status: draft for review · Date: 2026-10-05 · Context: CS59300-ASE Homework 1

## 1. Goal

A small local web app that helps a researcher find, collect, and understand papers. It supports the five HW1 features: search papers through a public API, save them to a persistent local library, upload PDFs with automatic metadata extraction, summarize a paper from its full text with an LLM, and answer natural-language questions about a paper.

Success means all five features work end to end in the demo video, someone can clone the repo and run it by following the README, no secrets are ever committed, and the code stays small and modular enough that the student understands every major component and can write tests against it in HW2.

## 2. Key decisions

| Decision | Choice | Why |
|---|---|---|
| Stack | FastAPI + Jinja2 templates + HTMX | All Python, one process, no Node or build step. Still a clear split: templates, HTMX, and CSS on the frontend; routes, services, and the database on the backend. |
| LLM | Google Gemini API, free tier, default model `gemini-3.8-flash` | Free with no payment method; a 1M-token context window fits whole papers; the model name is configurable. |
| Paper search | arXiv API | No key needed, and every result has a free PDF, which full-text summaries require. |
| Storage | SQLite through Python's built-in `sqlite3`; PDFs on disk | Zero setup, and data survives restarts. |
| Summary and Q&A context | Whole paper in the prompt (no RAG) | A paper is roughly 10–30k tokens. Full context answers questions like "what are the limitations?" better than retrieved chunks. |
| Deployment | Local only (`uvicorn`) | The student's choice; no Docker and no hosting. |
| Tests | None committed in HW1 | HW2 has the student generate the test suite with a coding agent; starting from zero gives a clean baseline. |

## 3. Architecture

One FastAPI process serves HTML pages rendered with Jinja2. HTMX, vendored into `app/static/` rather than loaded from a CDN, handles partial updates: search results, the save button, the summary panel, Q&A answers, and deletions. Route handlers stay thin and delegate to modules that each own one concern.

`app/arxiv_client.py` queries the arXiv API, retries when throttled, and parses the Atom feed into result objects. `app/pdfs.py` downloads PDFs, validates them, and extracts text with PyMuPDF. `app/llm.py` wraps the Gemini SDK in three functions: `extract_metadata(first_pages_text)`, `summarize(paper)`, and `answer(paper, question, history)`. `app/db.py` owns the SQLite schema and every query. `app/config.py` reads settings from the environment.

Network calls happen only in `arxiv_client.py`, in the download function of `pdfs.py`, and in `llm.py`. That gives HW2 tests three clear seams to replace with fakes. The data directory comes from configuration, so tests can point it at a temporary folder.

The three slow actions follow the same shape. **Save** takes the result's metadata from the submitted form, downloads the PDF, extracts its text, and inserts a row. **Upload** stores the file, extracts its text, asks Gemini for metadata from the first two pages, and inserts a row. **Summarize** and **Ask** load the stored full text, call Gemini, store the result, and render a partial.

## 4. Pages and routes

The header links to two sections, Search and Library; each paper also has its own page.

| Method and path | Purpose | Response |
|---|---|---|
| `GET /` | Search page. With `?q=`, it runs the search. HTMX requests get only the results list; normal requests get the full page, so refresh and the back button work. | Page or results partial |
| `POST /library` | Save an arXiv result, using the metadata fields from its card | Updated button: "In library · Open" |
| `GET /library` | Saved papers, newest first, plus the PDF upload form | Page |
| `POST /library/upload` | Upload a PDF (multipart form) | `HX-Redirect` to the new paper's page |
| `DELETE /papers/{id}` | Delete a paper, its questions, and its PDF file | Empty; the row disappears |
| `GET /papers/{id}` | Paper page: metadata, abstract, PDF link, summary panel, Q&A panel | Page |
| `GET /papers/{id}/pdf` | The stored PDF | `application/pdf` |
| `POST /papers/{id}/summary` | Generate the summary, or regenerate it | Summary partial |
| `POST /papers/{id}/questions` | Ask a question | New Q&A item, appended to the list |

Search shows up to 10 results sorted by relevance. Each card shows the title (linked to the arXiv abstract page), authors, year, and abstract (collapsed after a few lines), with a Save button. Papers already in the library show "In library · Open" instead. Deletion happens from the library list and asks for confirmation first.

The paper page shows the stored summary if one exists, otherwise a Summarize button. The Q&A panel lists earlier questions and offers one-click starter questions based on the assignment's examples: the problem addressed, the main idea, the datasets, the limitations, and the comparison with baselines.

Gemini returns Markdown, which the server renders to HTML with raw HTML disabled, so model output can't inject markup into the page.

## 5. Data model

```sql
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL CHECK (source IN ('arxiv', 'upload')),
    arxiv_id    TEXT    UNIQUE,      -- versionless, e.g. 2401.12345; NULL for uploads
    title       TEXT    NOT NULL,
    authors     TEXT    NOT NULL,    -- JSON array of names (may be empty)
    year        INTEGER,             -- NULL when unknown
    abstract    TEXT,
    url         TEXT,                -- arXiv abstract page; NULL for uploads
    pdf_path    TEXT    NOT NULL,    -- file name inside DATA_DIR/pdfs/
    full_text   TEXT    NOT NULL,    -- invariant: every library paper has text
    summary     TEXT,                -- cached Markdown; NULL until generated
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS questions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

The schema is created at startup if it is missing. Each operation opens its own connection with `PRAGMA foreign_keys = ON`, which is all a single-user local app needs. The arXiv ID is stored without its version suffix, so v1 and v2 of the same paper count as one.

## 6. Gemini usage

The app calls Gemini through the official `google-genai` Python SDK. All three functions request a low thinking level to keep responses fast. The thinking setting is only sent to models that accept it, so switching `GEMINI_MODEL` to an older model still works.

**Metadata extraction (uploads).** Input is the text of the first two pages, capped at 8,000 characters. Gemini returns JSON that must match a schema: `title` (string), `authors` (list of strings), `year` (integer or null), and `abstract` (string or null).

**Summary.** Input is the paper's full text. Output is Markdown with fixed sections: TL;DR (one or two sentences), Problem, Approach, Key results, and Limitations. It is stored in `papers.summary`, and Regenerate overwrites it.

**Q&A.** Input is the full text, the last 6 question-and-answer pairs for context, and the new question. The instructions tell Gemini to answer only from the paper, to say plainly when the paper doesn't cover something, and to name the relevant section when it can. Each pair is stored.

**Text budget.** The full text sent to Gemini is capped at 400,000 characters, roughly 100k tokens. Longer documents, such as theses, are truncated, and the panel says so. This keeps any single request well within free-tier per-minute token limits.

**Free-tier notes.** Quotas are set per model and per project, and the exact numbers are shown in Google AI Studio. Changing `GEMINI_MODEL` (for example, to a Flash-Lite model) is the escape hatch when one model's daily quota runs out. On the free tier, Google may use prompts to improve its products, so the README advises uploading only published papers.

## 7. arXiv integration

**Search.** Search calls `https://export.arxiv.org/api/query` with `sortBy=relevance`, `max_results=10`, and a descriptive User-Agent. The query requires every word the user typed to match somewhere in the record, using the `all:` field with terms joined by `AND`.

**Retries.** The arXiv API has been throttling heavily during 2026, so the client retries on HTTP 429, 5xx responses, and timeouts. It makes up to 3 attempts, waiting 3 s and then 6 s, with a 15 s timeout per attempt. If every attempt fails, the page shows "arXiv is busy right now — try again in a few seconds."

**Result fields.** The year comes from the `published` date of the first version. The abstract page link and the PDF link come from the entry's links.

**Saving.** The card submits the result's metadata along with its arXiv ID, so saving never needs a second arXiv API call. The server checks the ID against arXiv's ID formats and builds the abstract and PDF links from it, rather than accepting links from the form. That way the server only ever downloads from arxiv.org.

Saving downloads the PDF with a 30 s timeout and a 50 MB cap, checks that the file starts with `%PDF`, and extracts its text. If either step fails, nothing is saved and the card shows the error.

## 8. PDF processing

**Upload checks.** Uploads accept files up to 50 MB, and the file must actually be a PDF. Files are stored under `DATA_DIR/pdfs/` with generated names, never the user's file name.

**Text extraction.** PyMuPDF extracts the text page by page. If the result has fewer than 200 non-whitespace characters, the PDF is treated as scanned and rejected, because OCR is out of scope. The stored file is then deleted.

**Metadata fallback.** Upload metadata comes from the first source that works:

1. Gemini extraction
2. the title and author fields embedded in the PDF
3. the first non-empty line of text as the title, with no authors and no year

An upload never fails because of metadata alone. That also means uploads keep working when no Gemini key is configured.

## 9. Error handling

| Situation | Behavior |
|---|---|
| Empty search box | No request is sent; the page prompts for a query |
| No results | "No papers found" |
| arXiv still failing after retries | Inline "arXiv is busy" message (HTTP 503) |
| PDF download fails while saving | Inline error on that card; nothing saved (HTTP 502) |
| Non-PDF, too large, corrupt, or scanned upload | Inline error on the upload form (HTTP 400 or 413) |
| `GEMINI_API_KEY` missing | Summary and Q&A panels explain how to set it; search, save, upload, and browsing keep working |
| Gemini quota exhausted (HTTP 429) | Message names the model and suggests waiting or switching `GEMINI_MODEL` |
| Other Gemini failure or timeout | Short message; nothing is stored |
| Unknown paper id | 404 page |

Error partials carry real HTTP status codes. HTMX is configured to swap 4xx and 5xx responses into the page, so errors appear inline instead of being silently dropped.

## 10. Configuration and dependencies

Settings come from environment variables, loaded from a `.env` file by python-dotenv. `.env.example` is committed; `.env` is gitignored.

```
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
DATA_DIR=data
```

The app runs on Python 3.10 or newer. Its dependencies are:

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | Server |
| `jinja2` | Templates |
| `python-multipart` | Form and upload parsing |
| `httpx` | HTTP client |
| `feedparser` | arXiv Atom parsing |
| `pymupdf` | PDF text extraction |
| `google-genai` | Gemini |
| `python-dotenv` | `.env` loading |
| `markdown-it-py` | Markdown rendering |

## 11. Repository layout

```
ai-research-assistant/
├── app/
│   ├── main.py            # FastAPI app and routes
│   ├── config.py          # settings from the environment
│   ├── arxiv_client.py    # arXiv search, retries, Atom parsing
│   ├── pdfs.py            # PDF download, validation, text extraction
│   ├── llm.py             # Gemini: metadata, summary, Q&A
│   ├── db.py              # SQLite schema and queries
│   ├── templates/         # Jinja2 pages and partials
│   └── static/            # htmx.min.js (vendored), style.css
├── docs/
│   ├── design.md          # this document
│   └── plan.md            # implementation plan
├── .claude/
│   └── settings.json      # Claude Code project settings (shared)
├── CLAUDE.md              # Claude Code project instructions
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

`data/`, which holds the SQLite file and the PDFs, is created at runtime and gitignored.

## 12. Claude Code setup

The repo is prepared for later changes made with Claude Code.

**`CLAUDE.md`**, which Claude Code loads automatically at the start of each session, contains:

- a one-paragraph project summary and the run commands
- the module map from section 3
- the design invariants: every library paper has full text; network calls live only in `arxiv_client.py`, `pdfs.py`, and `llm.py`; routes stay thin
- conventions: keep dependencies minimal, add no frontend build step, and update this document and the README when behavior changes
- known gotchas: arXiv throttling, Gemini free-tier quotas, and never reading or printing `.env`

It mentions `docs/design.md` by path rather than importing it, so each session's context stays small.

**`.claude/settings.json`** is committed and shared. It denies Claude Code's file tools access to `.env` and `.env.*`. Personal permission rules belong in `.claude/settings.local.json`, which is listed in `.gitignore`.

## 13. Verification before handover

No tests are committed in HW1 (see section 2). Before handing the app over, the following are checked in the development sandbox:

- The server starts and every route responds.
- Search works against a sample arXiv Atom response, and retries behave correctly under simulated 429 responses.
- Save and upload work with generated PDFs: one with real text, and one image-only PDF to exercise the scanned-PDF rejection.
- Summary and Q&A work against a fake Gemini client, including the missing-key and quota-exhausted paths.
- After a restart, the library is still there.
- The rendered HTML of each page looks right.

Live arXiv checks happen only if `export.arxiv.org` and `arxiv.org` are allowed in the sandbox's network settings. The first live Gemini call happens on the student's machine, since the API key never enters the chat.

## 14. Out of scope

The following are left out on purpose:

- editing paper metadata
- OCR for scanned PDFs
- RAG and embeddings
- streaming responses
- multi-paper chat or comparison
- tags and collections
- citation export
- user accounts
- Docker or hosting
- additional search sources
- a committed test suite

## 15. Risks

**arXiv throttling** could make search slow or briefly unavailable during the demo. Retries soften this, and uploading a PDF doesn't depend on arXiv at all.

**Gemini free-tier quotas** are unpublished and set per model. A configurable model name and clear error messages handle this. Gemini 3.8 Flash only reached general availability in September 2026, so if its free quota turns out to be tight, a Flash-Lite model is the fallback.
