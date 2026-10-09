# AI Research Assistant

A local web app for finding, collecting, and understanding research papers. Search arXiv, save papers to a library on your computer, upload your own PDFs, and ask Google Gemini to summarize a paper or answer questions about it. Summaries and answers are based on the paper's full text, not just its abstract.

Built for CS59300 AI-Assisted Software Engineering (Purdue), Homework 1.

## Features

1. **Search arXiv** by topic, keywords, or arXiv ID. You get up to 20 results, sorted by arXiv's ranking, most recent first, or most cited first (citation counts come from Semantic Scholar). Each result shows the title, authors, year, abstract, and a link to the paper.
2. **Save papers** to a local SQLite library. Saving downloads the paper's PDF and extracts its text. The library survives restarts.
3. **Upload PDFs.** The app extracts the text, and Gemini reads the title, authors, year, and abstract from the first pages, so uploads look like saved search results.
4. **Summarize a paper.** Gemini reads the whole paper and writes a TL;DR plus the problem, the approach, the key results, and the limitations. Summaries are stored, and you can regenerate them.
5. **Ask questions about a paper** in plain language, such as "What datasets are used?". Answers come from the paper's text, and each paper keeps its question history.

## Requirements

- Python 3.10 or newer
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey). No payment method is needed. Search, saving, and uploading work without a key; summaries and questions need one.

## Setup

macOS or Linux:

```bash
git clone <your-repo-url> ai-research-assistant
cd ai-research-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell):

```powershell
git clone <your-repo-url> ai-research-assistant
cd ai-research-assistant
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Then open `.env` in an editor and paste your key after `GEMINI_API_KEY=`.

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Stop the server with Ctrl+C. Your library is kept in the `data/` folder.

## Using the app

1. **Search.** Type a topic, for example `code review automation`, and press Search. Every word you type must appear in a paper's record, except common words such as "a", "the", and "of", which arXiv doesn't index. Pasting an arXiv ID or link (for example `1706.03762`) goes straight to that paper. The **Sort** menu next to the box re-runs the search in another order:
   - **arXiv rank**: arXiv's best matches first.
   - **most recent**: the newest matching papers, each with its submission date.
   - **most cited**: arXiv's 100 best matches, sorted by their citation counts on Semantic Scholar.
2. **Save.** Click **Save to library** on a result. The app downloads the PDF from arXiv, which takes a few seconds, and the button changes to **Open**.
3. **Upload.** On the **Library** page, choose a PDF under **Add a PDF** and click **Upload PDF**. You'll land on the new paper's page.
4. **Summarize.** On a paper's page, click **Summarize paper**. It usually takes 5 to 30 seconds.
5. **Ask.** Click a suggested question or type your own and press **Ask**. Earlier questions on the same paper are sent along as context, so follow-up questions work.

To remove a paper, click **Delete** next to it in the library. This also deletes its PDF and its questions.

## Configuration

Settings are read from `.env` in the repo root. Real environment variables take precedence.

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | (empty) | Your Gemini API key. Required for summaries, questions, and Gemini-based upload metadata. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | The Gemini model to use. Switch to another free model, such as `gemini-3.5-flash-lite`, if you run out of quota. |
| `DATA_DIR` | `data` | Where the database (`library.db`) and the saved PDFs (`pdfs/`) are stored. Relative paths are relative to the repo root. |

Restart the app after changing `.env`.

## How it works

The app is a single FastAPI server that renders HTML pages with Jinja2 templates. [HTMX](https://htmx.org) (vendored in `app/static/`, no build step) updates parts of the page without full reloads: search results, the save button, the summary, and new answers.

```
Browser (HTML + HTMX)
        │  HTTP
        ▼
app/main.py ── routes: check input, call a module, render a template
   ├── app/arxiv_client.py   arXiv search API (Atom feed), retries when arXiv is busy
   ├── app/citations.py      citation counts from Semantic Scholar, for "most cited"
   ├── app/pdfs.py           PDF download from arxiv.org, upload checks, text extraction (PyMuPDF)
   ├── app/llm.py            Gemini via google-genai: metadata, summaries, Q&A
   └── app/db.py             SQLite: papers and questions
```

- **Search** calls the arXiv API, which needs no key. arXiv limits automated traffic, so a busy response is retried twice, after 3 and 6 seconds.
- **Most cited** needs data arXiv doesn't have. The app asks arXiv for its 100 best matches, looks up all of them in one request to Semantic Scholar's free API (no key needed), and shows the 20 most cited. Ranking only arXiv's best matches keeps the results on topic. Ranking more pulls in famous papers that merely mention your words, and makes the search slower.
- **Saving** downloads the PDF from arxiv.org and stores its full text in the database. Every paper in the library has its full text, which is what summaries and answers are based on.
- **Uploads** are checked (PDF, at most 50 MB, has selectable text). Gemini then reads the title, authors, year, and abstract from the first two pages. If Gemini isn't available, the app falls back to the PDF's own metadata and the text after an "Abstract" heading.
- **Summaries and answers** send the paper's full text to Gemini, with no retrieval step: a typical paper is 10–30k tokens, far below the model's 1M-token context window. Very long documents are cut at 400,000 characters. Model output is rendered as Markdown with raw HTML disabled.

`docs/design.md` explains these decisions in detail, and `docs/plan.md` is the implementation plan.

## Project structure

```
app/
  main.py            FastAPI app and routes
  config.py          settings from .env
  arxiv_client.py    arXiv search, retries, Atom parsing
  citations.py       Semantic Scholar citation counts
  pdfs.py            PDF download, validation, text extraction
  llm.py             Gemini: metadata, summary, Q&A
  db.py              SQLite schema and queries
  templates/         Jinja2 pages and HTMX partials
  static/            style.css, htmx.min.js
docs/
  design.md          design spec
  plan.md            implementation plan
.claude/settings.json    Claude Code project settings
.claude/skills/run-app/  Claude Code skill: run the app and drive it in a browser
CLAUDE.md                instructions for Claude Code
.env.example             settings template (copy to .env)
requirements.txt
```

## Troubleshooting

| What you see | What to do |
|---|---|
| "arXiv is busy right now" | arXiv is throttling requests. Wait a few seconds and search again. |
| "Couldn't download the PDF from arXiv" | Same cause; try saving again in a moment. |
| "Semantic Scholar is busy" | Its free API is shared by everyone without a key. Try again in a few seconds, or sort by arXiv rank or most recent, which don't use it. |
| "No Gemini API key is set" | Add `GEMINI_API_KEY` to `.env` and restart the app. |
| "Gemini's free-tier quota … is used up" | Wait a minute. If it persists, today's quota for that model is spent: set `GEMINI_MODEL` to another free model and restart. Your limits are shown in Google AI Studio. |
| "Gemini rejected the API key" | Check the key in `.env` for typos or extra spaces. |
| "That PDF has no readable text" | The PDF is a scan. Use a version with selectable text. |
| `Address already in use` | Another app uses port 8000. Run `uvicorn app.main:app --reload --port 8001`. |

## Privacy

Everything you save stays on your computer in `data/`. Searches go to arXiv, and a "most cited" search also sends the arXiv IDs of the matching papers (not your search words) to Semantic Scholar. The paper's text is sent to Google's Gemini API when you upload a PDF, summarize, or ask a question. On Gemini's free tier, Google may use what you send to improve its products, so stick to published papers and don't upload anything confidential.

## Developing with Claude Code

The repo is set up for [Claude Code](https://claude.com/claude-code):

- `CLAUDE.md` is loaded automatically at the start of each session. It holds the commands, a code map, the rules a change must keep, the seams for faking arXiv and Gemini in tests, and known gotchas.
- `.claude/settings.json` (shared, committed) stops Claude Code's file tools from reading or editing `.env`. Put personal settings in `.claude/settings.local.json`, which git ignores.
- `.claude/skills/run-app/` (shared, committed) is a project skill. It starts a throwaway copy of the app on port 8765, with a scratch library and Gemini turned off, then drives the search page in headless Chrome and saves screenshots. Claude Code uses it when you ask it to run the app or check that a change works; you can also type `/run-app`. Your own server, library, and API key are left alone.

## How this was built

The design, plan, and code were produced with Claude (Anthropic) as the coding agent, working from the HW1 requirements and the decisions recorded in `docs/design.md`.
