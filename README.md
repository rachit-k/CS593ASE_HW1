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
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Setup

macOS:

```bash
git clone git@github.com:rachit-k/CS593ASE_HW1.git ai-research-assistant
cd ai-research-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` in an editor and paste your key after `GEMINI_API_KEY=`.

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Stop the server with Ctrl+C. Your library is kept in the `data/` folder.

## Using the app

1. **Search.** Type a paper's name or a topic, for example `code review automation`, and press Search. Every word you type must appear in a paper's record, except common words such as "a", "the", and "of", which arXiv doesn't index. Pasting an arXiv ID or link goes straight to that paper. The **Sort** menu next to the box re-runs the search in another order:
   - **arXiv rank**: arXiv's best matches first.
   - **most recent**: the newest matching papers, each with its submission date.
   - **most cited**: arXiv's 100 best matches, sorted by their citation counts on Semantic Scholar.
2. **Save.** Click **Save to library** on a result. The app downloads the PDF from arXiv, and the button changes to **Open**.
3. **Upload.** On the **Library** page, choose a PDF under **Add a PDF** and click **Upload PDF**. 
4. **Summarize.** On a paper's page, click **Summarize paper**.
5. **Ask a question** Click a suggested question or type your own and press **Ask**. Earlier questions on the same paper are sent along as context, so follow-up questions work.

To remove a paper, click **Delete** next to it in the library. This also deletes its PDF and its questions.

## Tests

The test suite is in `tests/`. It has 1,263 test cases. The tests use temporary folders and fake versions of arXiv, Semantic Scholar, and Gemini, so they need no API key, no `.env`, and no network connection, and they never touch your library.

### 1. Install the dependencies

From the repo root, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt     # the app's dependencies plus pytest, pytest-cov, Hypothesis, and Playwright
```

The browser tests drive your installed Google Chrome. Without Chrome, run `playwright install chromium` once, or the browser tests are skipped.

### 2. Run the tests

```bash
pytest                       # everything; ends with "1263 passed" in about 30 seconds
pytest -m "not browser"      # without the browser tests
pytest tests/test_db.py      # one file
```

### 3. Reproduce the coverage results

```bash
pytest --cov=app --cov-branch --cov-report=term-missing --cov-fail-under=100
```

This prints line and branch coverage for each module in `app/`. The last lines should be:

```
TOTAL                   738      0    154      0   100%
Required test coverage of 100% reached. Total coverage: 100.00%
1263 passed
```

The command fails if coverage drops below 100%. Add `--cov-report=html` to get a browsable report in `htmlcov/`.

### 4. Checks beyond coverage

Three more checks go beyond coverage. Each one leaves your files alone:

```bash
python tools/path_coverage.py                               # conditions, and/or operands, and template branches
python tools/mutation_test.py                               # small code changes that no test notices (about half an hour)
HYPOTHESIS_PROFILE=thorough pytest tests/test_properties.py # the property tests with 2,000 random inputs each
```

`path_coverage.py` should report 100% for all three measures. `mutation_test.py` prints a score per file and a total, which was 97.1% (1,399 of 1,441 changes noticed), and writes the details to `mutation-report.json`.

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
  design.md          design 
  plan.md            implementation plan
tests/
  conftest.py        shared setup: temp library, fakes, no real network
  helpers.py         builders for PDFs and feeds, and an HTML reader
  recordings/        real API responses and real papers that the tests replay
  test_*.py          one file per module, plus routes, errors, and whole journeys, and one
                     per testing strategy (limits, decision tables, fault injection, properties,
                     metamorphic, stateful, security, concurrency, real papers, contracts, markup, browser)
tools/
  path_coverage.py   measures conditions, and/or operands, and template branches
  mutation_test.py   makes small changes to the code and reports the ones no test notices
.claude/settings.json    Claude Code project settings
CLAUDE.md                instructions for Claude Code
.env.example             settings template (copy to .env)
pytest.ini               test configuration
requirements.txt
requirements-dev.txt     test dependencies
```
