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
.claude/settings.json    Claude Code project settings
CLAUDE.md                instructions for Claude Code
.env.example             settings template (copy to .env)
requirements.txt
```
