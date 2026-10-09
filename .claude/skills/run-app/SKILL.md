---
name: run-app
description: Launch the AI Research Assistant (this repo's FastAPI + HTMX app) and drive it like a user. Starts a throwaway server on port 8765 with a scratch library and Gemini turned off, smoke-tests routes and HTMX partials with curl, and drives the search page in headless Chrome with a bundled Playwright script (search, sort menu, Back/Forward, phone layout, light and dark screenshots). Use it whenever you need to run, start, open, screenshot, or demo this app, or to confirm that a change to routes, templates, CSS, htmx behavior, search, or sorting works in the real app rather than only in tests, even if the user just says "check it works" or "try it".
---

# Run and drive the Research Assistant

Run a throwaway copy of the app, not the user's own. The user usually has their own server on port 8000, their library in `data/`, and their Gemini key in `.env`. A separate instance can't disturb any of them, and it can't send anything to Gemini.

## 1. Start a throwaway server

Use a scratch directory outside the repo: the session's scratchpad if your system prompt names one, otherwise `mktemp -d`. It's `$SCRATCH` below.

From the repo root, start the server in the background (it runs until you stop it):

```bash
GEMINI_API_KEY= DATA_DIR="$SCRATCH/data" .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Then wait for it by polling, not sleeping:

```bash
curl -s -o /dev/null --retry 15 --retry-delay 1 --retry-connrefused http://127.0.0.1:8765/
```

Why each part matters:

- **Port 8765, not 8000.** 8000 is where the user runs `uvicorn app.main:app --reload`. If 8765 is taken too (`lsof -nP -iTCP:8765 -sTCP:LISTEN`), pick another.
- **`DATA_DIR` in scratch.** `data/` in the repo is the user's real library. Saves, uploads, and deletes from a test run must not land there.
- **`GEMINI_API_KEY=` (empty).** Real environment variables win over `.env` (`load_dotenv(override=False)` in `app/config.py`), so the empty value keeps the user's key out of the process. Nothing reaches Gemini, and there's never a reason to read `.env`, so don't.
- **No `--reload`.** Restart the server after editing Python. Templates and files in `app/static/` are read from disk on each request, so those changes show up right away.

If `.venv` doesn't exist yet: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

To stop it: `kill $(lsof -ti tcp:8765 -sTCP:LISTEN)`. The background task then reports exit code 143 or 144. That's the kill, not a crash.

## 2. Smoke-test with curl

```bash
B=http://127.0.0.1:8765
for p in / /library /papers/999; do curl -s -o /dev/null -w "$p %{http_code}\n" "$B$p"; done   # 200, 200, 404

# The fragment htmx swaps in. sort is relevance, recent, or citations.
curl -s -G -H 'HX-Request: true' --data-urlencode 'q=code review automation' \
  --data-urlencode 'sort=citations' "$B/" | grep -E 'results-count|entry-title|citations?<'
```

With `HX-Request: true` you get the partial that a click loads; without it, the full page that a refresh loads. Both paths matter.

When a change touches saving or uploading:

```bash
# Save: downloads the real PDF from arxiv.org and extracts its text.
curl -s -H 'HX-Request: true' -X POST "$B/library" --data-urlencode arxiv_id=1706.03762 \
  --data-urlencode 'title=Attention Is All You Need' --data-urlencode 'authors=["Ashish Vaswani"]' \
  --data-urlencode year=2017 --data-urlencode 'abstract=Transformers.'
# Upload any text PDF (here, the one just saved). Expect 303 to the new paper's page.
PDF=$(ls "$SCRATCH"/data/pdfs/*.pdf | head -1)
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' -F "file=@$PDF;type=application/pdf" "$B/library/upload"
sqlite3 "$SCRATCH/data/library.db" 'select id, source, title, length(full_text) from papers'
```

With Gemini off, uploads use the metadata fallbacks (the redirect ends in `?notice=metadata-fallback`), and Summarize and Ask answer "No Gemini API key is set" with HTTP 503. That's expected here. To see the Gemini features work, ask the user to try them on their own server.

## 3. Drive the search page in headless Chrome

Set up Playwright once per session, outside the repo, so nothing gets committed and the app's `.venv` stays clean:

```bash
python3 -m venv "$SCRATCH/pw-venv" && "$SCRATCH/pw-venv/bin/pip" install -q playwright
```

It drives the installed Google Chrome, so there's no browser download. Without Chrome, also run `"$SCRATCH/pw-venv/bin/playwright" install chromium` and add `--bundled-chromium` to the command below.

Run the bundled driver. It takes about a minute, mostly spent waiting out arXiv's rate limit:

```bash
"$SCRATCH/pw-venv/bin/python" .claude/skills/run-app/scripts/drive.py http://127.0.0.1:8765/ "$SCRATCH/shots"
```

It searches for "code review automation" (change it with `--query`) and goes through what a user would do:

1. Loads the results page, then switches the Sort menu to most cited and to most recent. Each switch must go out as an htmx request (no page reload), show 20 results, and update the URL.
2. Goes Back, Back, Forward. The menu must match the results each time.
3. Runs two new searches and goes Back. The search box must show the query behind the results.
4. Loads the phone layout (390 px wide). The search box must get its own row, and nothing may scroll sideways.
5. Fails on any console error, and exits with 1 if any check failed.

It saves `desktop-light.png`, `desktop-dark.png`, `phone-light.png`, and `phone-dark.png` in the output folder. Look at them with the Read tool before reporting success: a page can pass every check and still look broken.

To drive another page (the library, a paper page), follow the same pattern in a small script: `page.goto`, wait for the element you need, act, take a screenshot, check the console. If you change the search page's markup, update the selectors at the top of `drive.py`.

## Gotchas

- **arXiv allows one request every 3 seconds.** Every search, sort change, and full page load with `?q=` calls arXiv, and a "most cited" search also calls Semantic Scholar once. The driver waits 3.5 s between them. Don't loop over searches quickly or run several drivers at once: arXiv answers 429, and the app's retries slow everything down.
- **The user's `--reload` server ignores edits?** Look for files dated in the future: `find . \( -path ./.venv -o -path ./.git \) -prune -o -type f -newermt "$(date '+%Y-%m-%d %H:%M:%S')" -print`. Files unzipped from an archive made in another timezone can carry future timestamps, and uvicorn's reloader only notices edits that look newer. `touch` those files, then have the user restart their server once.
- **Back or Forward shows a stale query or sort?** htmx's history snapshots keep HTML attributes, not typed or picked values. The `htmx:afterSettle` handler in `app/templates/base.html` copies the search form's values into its attributes; steps 2 and 3 of the driver catch regressions there.
