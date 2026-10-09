# AI Research Assistant — Implementation Plan

Date: 2026-10-09 · Implements: `docs/design.md` · Deadline: HW1 due 2026-10-10, 11:59 PM

Because of the deadline, the student approved building straight through from this plan without a separate plan review.

## Build order

Each step ends with a commit.

1. **Backend modules** (`app/`)
   - `config.py`: `Settings` (API key, model, data dir) loaded from `.env`; relative `DATA_DIR` resolves against the repo root.
   - `db.py`: schema from design §5, plus one function per query. Every function opens its own connection with foreign keys on.
   - `arxiv_client.py`: query building (word search or ID lookup), retries with backoff, Atom parsing into `ArxivResult`, arXiv ID validation, and a link builder.
   - `pdfs.py`: arXiv PDF download (host check, size cap, `%PDF` check), upload validation, text extraction, scanned-PDF rejection, embedded-metadata and abstract fallbacks.
   - `llm.py`: Gemini client, the three functions from design §6, thinking level sent only to Gemini 3 models, and errors mapped to user-facing messages.
2. **Routes and pages**
   - `main.py`: the nine routes from design §4, the HTMX partial vs. full-page logic, and the 404 handler.
   - Templates: `base`, `search`, `library`, `paper`, `error`, and partials for results, the save button, upload status, the summary, Q&A items, and errors.
   - Static: vendored `htmx.min.js` 2.0.11 and `style.css`. The HTMX config swaps 4xx and 5xx responses into the page.
3. **Repo files**: `README.md`, `CLAUDE.md`, `.claude/settings.json`, `requirements.txt`, `.env.example`, `.gitignore`.
4. **Verification**: see below. Fix and recommit until everything passes.
5. **Handover**: a zip of the repo with its git history.

## Verification checklist (sandbox)

The sandbox can't reach arXiv or Gemini, so network calls are replaced with fakes at the three seams named in design §3.

- [ ] App imports and starts; `GET /`, `/library`, and a missing paper's page return 200, 200, and 404.
- [ ] Search parses a sample Atom feed (titles, authors, year, abstract, links); empty and error entries are handled; an arXiv ID query uses `id_list`.
- [ ] Simulated arXiv 429s trigger retries and end in the "arXiv is busy" message with status 503.
- [ ] Save downloads a (fake) PDF and stores the paper; saving again is a no-op; a failed download returns 502 and stores nothing; a forged ID is rejected.
- [ ] Upload of a text PDF stores it and redirects to the paper page; uploads of non-PDF, scanned, and oversized files show inline errors.
- [ ] Upload metadata uses Gemini when available, then the PDF fields, then the first line.
- [ ] Summary and Q&A run through the real `llm.py` against a fake `genai` client, with a Gemini 3 model getting `thinking_level=LOW` and a 2.5 model getting no thinking setting.
- [ ] Missing key, 429 quota, 404 model, and 5xx errors each produce the right inline message, and nothing is stored.
- [ ] Model Markdown renders, and raw `<script>` in model output is escaped.
- [ ] Delete removes the row, its questions, and the PDF file.
- [ ] The library survives a server restart.
- [ ] Screenshots of the search, library, and paper pages in headless Chromium look right on desktop and phone widths.

## After handover (student)

1. Unzip, create a GitHub repo, add it as `origin`, and push.
2. Create a virtual environment, install `requirements.txt`, copy `.env.example` to `.env`, and add a Gemini key from Google AI Studio.
3. Run `uvicorn app.main:app --reload`, then try search, save, upload, summarize, and ask live. The first live run may surface arXiv or Gemini differences that the fakes couldn't catch.
4. Record the demo video and write the reflection.
