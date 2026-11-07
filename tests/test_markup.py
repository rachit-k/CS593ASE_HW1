"""What the pages promise to browsers and screen readers, beyond the text they show.

Mutation testing found these: any one of these attributes could be removed and no test noticed.
What needs a real browser to check, such as the name a screen reader ends up with, is in test_browser.py.
"""

from __future__ import annotations

import pytest

from app.main import MAX_QUESTION_CHARS
from helpers import HTMX, PAPER_TITLE, atom_entry, atom_feed, parse_html


def page_at(client, url: str):
    return parse_html(client.get(url).text)


def test_every_page_has_the_same_header(client, add_paper):
    for url in ("/", "/library", f"/papers/{add_paper()}", "/papers/999"):
        page = page_at(client, url)
        assert page.find("a", class_="brand").attrs["href"] == "/"
        assert page.find("nav").attrs["aria-label"] == "Main"
        assert [link.attrs["href"] for link in page.find("nav").find_all("a")] == ["/", "/library"]


def test_the_header_marks_the_section_you_are_in(client, add_paper):
    def current(url: str) -> list[str]:
        links = page_at(client, url).find("nav").find_all("a")
        return [link.attrs["href"] for link in links if link.attrs.get("aria-current") == "page"]

    assert current("/") == ["/"]
    assert current("/library") == ["/library"]
    assert current(f"/papers/{add_paper()}") == ["/library"]


def test_the_page_icon_is_built_in_so_the_browser_requests_nothing(client):
    assert page_at(client, "/").find("link", rel="icon").attrs["href"].startswith("data:image/svg+xml,")


def test_the_page_title_says_what_is_on_the_page(client, add_paper):
    assert page_at(client, f"/papers/{add_paper()}").find("title").text == f"{PAPER_TITLE} | Research Assistant"
    assert page_at(client, "/papers/999").find("title").text == "Not found | Research Assistant"


def test_the_error_page_offers_a_way_on(client):
    links = page_at(client, "/papers/999").find(class_="paper-actions").find_all("a")
    assert [(link.text, link.attrs["href"]) for link in links] == [("Go to your library", "/library"), ("Search arXiv", "/")]


def test_the_paper_page_links_back_to_the_library(client, add_paper):
    assert page_at(client, f"/papers/{add_paper()}").find("a", class_="back-link").attrs["href"] == "/library"


def test_the_empty_library_links_to_search(client):
    assert page_at(client, "/library").find(class_="empty-note").find("a").attrs["href"] == "/"


def assert_opens_safely_in_a_new_tab(link) -> None:
    assert link.attrs["target"] == "_blank"
    assert link.attrs["rel"] == "noopener"  # so the other site gets no handle on this page


def test_links_to_other_sites_open_safely_in_a_new_tab(client, arxiv, scholar, add_paper):
    arxiv.reply(200, text=atom_feed(atom_entry()))
    scholar.reply(200, json=[{"citationCount": 1}])
    results = page_at(client, "/?q=testing&sort=citations")
    assert_opens_safely_in_a_new_tab(results.find(class_="entry-title").find("a"))
    assert_opens_safely_in_a_new_tab(results.find(class_="results-count").find("a"))

    paper = page_at(client, f"/papers/{add_paper()}")
    arxiv_link = next(link for link in paper.find(class_="paper-actions").find_all("a") if link.text == "View on arXiv")
    assert_opens_safely_in_a_new_tab(arxiv_link)


def test_the_delete_button_names_its_paper_and_asks_first(client, add_paper):
    add_paper()
    button = page_at(client, "/library").find("button", class_="button-danger")
    assert button.attrs["aria-label"] == f"Delete {PAPER_TITLE}"
    assert PAPER_TITLE in button.attrs["hx-confirm"]


def test_the_upload_box_only_offers_pdfs(client):
    box = page_at(client, "/library").find("input", name="file")
    assert box.attrs["accept"] == "application/pdf,.pdf"


def test_the_text_boxes_stop_at_the_length_the_server_accepts(client, add_paper):
    assert page_at(client, f"/papers/{add_paper()}").find("input", name="question").attrs["maxlength"] == str(MAX_QUESTION_CHARS)
    assert page_at(client, "/").find("input", name="q").attrs["maxlength"] == "300"


@pytest.mark.parametrize(("path", "live_region"), [("/", "results"), ("/library", "upload-status")])
def test_a_screen_reader_hears_what_arrives_in_these_areas(client, path, live_region):
    assert page_at(client, path).find(id=live_region).attrs["aria-live"] == "polite"


def test_a_screen_reader_hears_question_errors_before_and_after_a_retry(client, gemini, add_paper):
    paper_id = add_paper()
    assert page_at(client, f"/papers/{paper_id}").find(id="qa-error").attrs["aria-live"] == "polite"
    gemini.reply_text("An answer.")
    answer = parse_html(client.post(f"/papers/{paper_id}/questions", data={"question": "Why?"}, headers=HTMX).text)
    assert answer.find(id="qa-error").attrs["aria-live"] == "polite"


def test_a_screen_reader_skips_the_decorative_q_and_a_labels(client, gemini, add_paper):
    gemini.reply_text("An answer.")
    item = parse_html(client.post(f"/papers/{add_paper()}/questions", data={"question": "Why?"}, headers=HTMX).text)
    labels = item.find_all(class_="margin-label")
    assert [label.text for label in labels] == ["Q", "A"]
    assert [label.attrs["aria-hidden"] for label in labels] == ["true", "true"]
