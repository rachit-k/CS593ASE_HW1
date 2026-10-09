"""Drive the Research Assistant's search page in headless Chrome, the way a user would.

Searches, switches the Sort menu, uses Back and Forward, loads the phone layout, and
saves desktop and phone screenshots in light and dark. Prints what it sees at each step
and exits with 1 if any check fails. Needs Playwright; see ../SKILL.md.

Usage: python drive.py BASE_URL OUT_DIR [--query TEXT] [--bundled-chromium]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import Browser, Page, sync_playwright

# Selectors for the search page (app/templates/search.html and partials/results.html).
SEARCH_BOX = "input[name=q]"
SORT_MENU = "select[name=sort]"
SEARCH_BUTTON = ".search-form button"
RESULTS_HEADER = ".results-count"
RESULT_CARD = ".entry"

ARXIV_GAP_MS = 3500  # arXiv asks for at most one request every 3 seconds
WAIT_MS = 60_000  # a search can take a while when arXiv is busy and the app retries

failures = 0


def check(label: str, ok: bool, detail: object = "") -> None:
    global failures
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}{'' if ok else f'  ->  {detail}'}", flush=True)


def state(page: Page, base: str) -> dict:
    """What a user sees: the URL, the search box, the sort menu, the results header, the cards."""
    return {
        "url": page.url.removeprefix(base.rstrip("/")),
        "box": page.eval_on_selector(SEARCH_BOX, "i => i.value"),
        "sort": page.eval_on_selector(SORT_MENU, "s => s.value"),
        "header": page.inner_text(RESULTS_HEADER).strip(),
        "cards": page.locator(RESULT_CARD).count(),
    }


def wait_for_header(page: Page, text: str) -> None:
    page.locator(RESULTS_HEADER).filter(has_text=text).wait_for(timeout=WAIT_MS)


def watch_console(page: Page) -> list[str]:
    errors: list[str] = []
    page.on("console", lambda message: message.type == "error" and errors.append(message.text))
    page.on("pageerror", lambda error: errors.append(str(error)))
    return errors


def drive_desktop(browser: Browser, base: str, query: str, out: Path) -> None:
    page = browser.new_page(viewport={"width": 1280, "height": 900}, color_scheme="light")
    errors = watch_console(page)
    # The HX-Request header of each search: "true" when htmx sent it, None for a page load.
    searches: list[str | None] = []
    page.on("request", lambda r: r.url.startswith(base + "?") and searches.append(r.headers.get("hx-request")))

    page.goto(f"{base}?{urlencode({'q': query})}")
    wait_for_header(page, "best match first")
    s = state(page, base)
    print(f"     {s['header']}")
    check("search: 20 results in arXiv order", s["cards"] == 20 and s["sort"] == "relevance", s)

    for sort, header in (("citations", "most cited first"), ("recent", "newest first")):
        page.wait_for_timeout(ARXIV_GAP_MS)
        page.select_option(SORT_MENU, sort)
        wait_for_header(page, header)
        s = state(page, base)
        print(f"     {s['header']}")
        check(f"sort -> {sort}: htmx request, 20 results, URL updated",
              searches[-1] == "true" and s["cards"] == 20 and f"sort={sort}" in s["url"], (searches[-1], s))
        if sort == "citations":
            page.screenshot(path=out / "desktop-light.png")
            page.emulate_media(color_scheme="dark")
            page.screenshot(path=out / "desktop-dark.png")
            page.emulate_media(color_scheme="light")

    for step, header, sort in (
        ("back", "most cited first", "citations"),
        ("back", "best match first", "relevance"),
        ("forward", "most cited first", "citations"),
    ):
        page.wait_for_timeout(ARXIV_GAP_MS)
        page.go_back() if step == "back" else page.go_forward()
        wait_for_header(page, header)
        s = state(page, base)
        check(f"{step}: the menu matches the results ({sort})", s["sort"] == sort, s)

    for new_query in ("graph neural networks", "program synthesis"):
        page.wait_for_timeout(ARXIV_GAP_MS)
        page.fill(SEARCH_BOX, new_query)
        page.press(SEARCH_BOX, "Enter")
        wait_for_header(page, new_query)
    page.wait_for_timeout(ARXIV_GAP_MS)
    page.go_back()
    wait_for_header(page, "graph neural networks")
    s = state(page, base)
    check("back after two searches: the box shows the query behind the results",
          s["box"] == "graph neural networks", s)

    check("desktop: no console errors", not errors, errors)
    page.close()


def drive_phone(browser: Browser, base: str, query: str, out: Path) -> None:
    page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True,
                            device_scale_factor=2, color_scheme="light")
    errors = watch_console(page)
    page.wait_for_timeout(ARXIV_GAP_MS)
    page.goto(f"{base}?{urlencode({'q': query, 'sort': 'citations'})}")
    wait_for_header(page, "most cited first")

    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check("phone: nothing scrolls sideways", overflow <= 0, f"{overflow}px too wide")
    box, menu, button = (page.locator(sel).bounding_box() for sel in (SEARCH_BOX, SORT_MENU, SEARCH_BUTTON))
    check("phone: the search box has its own row, with the menu and button below",
          menu["y"] > box["y"] + 10 and abs(menu["y"] - button["y"]) < 5, (box, menu, button))
    page.screenshot(path=out / "phone-light.png")
    page.emulate_media(color_scheme="dark")
    page.screenshot(path=out / "phone-dark.png")

    check("phone: no console errors", not errors, errors)
    page.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", help="where the app runs, for example http://127.0.0.1:8765/")
    parser.add_argument("out_dir", type=Path, help="folder for the screenshots")
    parser.add_argument("--query", default="code review automation", help="what to search for")
    parser.add_argument("--bundled-chromium", action="store_true",
                        help="use Playwright's Chromium (after `playwright install chromium`) instead of Google Chrome")
    args = parser.parse_args()
    base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=None if args.bundled_chromium else "chrome")
        drive_desktop(browser, base, args.query, args.out_dir)
        drive_phone(browser, base, args.query, args.out_dir)
        browser.close()

    print(f"\n{failures} failed check(s). Screenshots: {args.out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
