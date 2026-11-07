"""One paper: its page, its PDF, summaries, and questions (design sections 4, 6, 9)."""

from __future__ import annotations

import json
import re
from urllib.parse import unquote

import pytest

from app import db, pdfs
from app.main import render_markdown
from helpers import BODY_FACT, HTMX, PAPER_ABSTRACT, PAPER_TITLE, make_pdf, parse_html

QUOTA = (429, "Resource has been exhausted.", "RESOURCE_EXHAUSTED")
SUMMARY = "**TL;DR:** Agents write useful tests.\n\n### Problem\nTesting is slow.\n\n- One point\n- Another point"


def page_of(client, paper_id, **params):
    response = client.get(f"/papers/{paper_id}", params=params)
    assert response.status_code == 200
    return parse_html(response.text)


def alert_kinds(page) -> list[str]:
    return [node.attrs["class"] for node in page.find_all(class_="alert")]


def download_name(disposition: str) -> str:
    """The file name in a Content-Disposition header: filename*=utf-8''... or filename="..."."""
    encoded = re.search(r"filename\*=utf-8''([^;]+)", disposition)
    return unquote(encoded.group(1)) if encoded else re.search(r'filename="([^"]+)"', disposition).group(1)


# --- The paper page -----------------------------------------------------------------


def test_an_arxiv_paper_shows_its_details_and_links(client, add_paper):
    paper_id = add_paper()
    page = page_of(client, paper_id)
    head = page.find("header", class_="paper-head")
    assert head.find("h1").text == PAPER_TITLE
    assert head.find(class_="authors").text == "Alice Smith, Bob Jones"
    assert head.find(class_="facts").text == "2024 arXiv 2401.12345"
    assert head.find("a", href=f"/papers/{paper_id}/pdf") is not None
    assert head.find("a", href="https://arxiv.org/abs/2401.12345") is not None
    assert page.find(class_="abstract").text == PAPER_ABSTRACT


def test_all_authors_are_shown_on_the_paper_page(client, add_paper):
    names = [f"Author {n}" for n in range(8)]
    page = page_of(client, add_paper(authors=names))
    assert page.find("header", class_="paper-head").find(class_="authors").text == ", ".join(names)


def test_an_upload_with_unknown_details_says_so(client, add_paper):
    paper_id = add_paper(source="upload", arxiv_id=None, url=None, authors=[], year=None, abstract=None)
    page = page_of(client, paper_id)
    head = page.find("header", class_="paper-head")
    assert head.find(class_="authors").text == "Authors unknown"
    assert head.find(class_="facts").text == "Year unknown Uploaded PDF"
    assert [link.text for link in head.find_all("a")] == ["Open PDF"]
    assert page.find(class_="abstract").text == "No abstract was found for this paper."


def test_html_in_stored_metadata_is_escaped(client, add_paper):
    paper_id = add_paper(title="<script>alert(1)</script>", authors=["<b>Eve</b>"], abstract="<i>Hi</i>")
    page = page_of(client, paper_id)
    article = page.find("article")
    assert article.find("script") is None
    assert article.find("b") is None
    assert article.find("i") is None
    assert article.find("h1").text == "<script>alert(1)</script>"


def test_no_summary_yet_offers_the_summarize_button(client, gemini, add_paper):
    paper_id = add_paper()
    panel = page_of(client, paper_id).find(id="summary-panel")
    button = panel.find("button")
    assert button.text == "Summarize paper"
    assert button.attrs["hx-post"] == f"/papers/{paper_id}/summary"
    assert button.attrs["hx-target"] == "#summary-panel"
    assert panel.find(class_="summary") is None


def test_a_stored_summary_is_shown_with_a_regenerate_button(client, add_paper):
    paper_id = add_paper()
    db.set_summary(paper_id, SUMMARY)
    panel = page_of(client, paper_id).find(id="summary-panel")
    assert panel.find(class_="summary").find("h3").text == "Problem"
    assert [button.text for button in panel.find_all("button")] == ["Regenerate"]


def test_the_starter_questions_are_the_assignments_examples(client, add_paper):
    paper_id = add_paper()
    chips = page_of(client, paper_id).find_all("button", class_="chip")
    assert [chip.text for chip in chips] == [
        "What problem does this paper address?",
        "What is the main idea of the proposed approach?",
        "What datasets are used?",
        "What are the major limitations?",
        "How does this method compare with the baselines?",
    ]
    for chip in chips:
        assert chip.attrs["hx-post"] == f"/papers/{paper_id}/questions"
        assert json.loads(chip.attrs["hx-vals"]) == {"question": chip.text}


def test_earlier_questions_are_listed_oldest_first(client, add_paper):
    paper_id = add_paper()
    db.add_question(paper_id, "First question?", "First **answer**.")
    db.add_question(paper_id, "Second question?", "Second answer.")
    items = page_of(client, paper_id).find(id="qa-list").find_all("li", class_="qa-item")
    assert [item.find(class_="question").text for item in items] == ["First question?", "Second question?"]
    assert items[0].find(class_="answer").find("strong").text == "answer"


def test_the_metadata_fallback_notice_appears_only_when_asked_for(client, gemini, add_paper):
    paper_id = add_paper()
    assert alert_kinds(page_of(client, paper_id)) == []
    assert alert_kinds(page_of(client, paper_id, notice="anything-else")) == []
    page = page_of(client, paper_id, notice="metadata-fallback")
    assert "may be incomplete" in page.find(class_="alert-info").text


def test_without_a_key_the_page_explains_how_to_add_one(client, add_paper):
    page = page_of(client, add_paper())
    assert "GEMINI_API_KEY" in page.find(class_="alert-warn").text


def test_a_very_long_paper_warns_that_only_part_is_sent(client, gemini, add_paper):
    long_paper = add_paper(arxiv_id="2401.00001", full_text="x" * 400_001)
    exact_paper = add_paper(arxiv_id="2401.00002", full_text="x" * 400_000)
    assert "400,000 characters" in page_of(client, long_paper).find(class_="alert-info").text
    assert page_of(client, exact_paper).find(class_="alert-info") is None


# --- The stored PDF -----------------------------------------------------------------


def test_the_pdf_is_served_inline_with_the_title_as_its_name(client, add_paper):
    pdf = make_pdf()
    paper_id = add_paper(pdf_path=pdfs.save_pdf(pdf), title="Graphs: A Survey! (2nd ed.)")

    response = client.get(f"/papers/{paper_id}/pdf")

    assert response.status_code == 200
    assert response.content == pdf
    assert response.headers["content-type"] == "application/pdf"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("inline")
    assert download_name(disposition) == "Graphs A Survey 2nd ed.pdf"


def test_a_non_ascii_title_makes_a_valid_download_name(client, add_paper):
    response = client.get(f"/papers/{add_paper(title='Schrödinger Equations')}/pdf")
    assert response.status_code == 200
    assert download_name(response.headers["content-disposition"]) == "Schrödinger Equations.pdf"


@pytest.mark.parametrize(
    ("title", "name"),
    [("!!! ???", "paper.pdf"), ("x" * 200, "x" * 80 + ".pdf")],
    ids=["only-punctuation", "very-long"],
)
def test_download_names_are_always_usable(client, add_paper, title, name):
    response = client.get(f"/papers/{add_paper(title=title)}/pdf")
    assert download_name(response.headers["content-disposition"]) == name


def test_a_missing_pdf_file_is_a_404_page(client, add_paper):
    paper_id = add_paper()
    pdfs.delete_pdf(db.get_paper(paper_id)["pdf_path"])
    response = client.get(f"/papers/{paper_id}/pdf")
    assert response.status_code == 404
    assert "missing" in parse_html(response.text).find(class_="lede").text


def test_a_stored_file_name_cannot_serve_files_outside_the_pdf_folder(client, add_paper):
    paper_id = add_paper(pdf_path="../library.db")
    response = client.get(f"/papers/{paper_id}/pdf")
    assert response.status_code == 404
    assert b"SQLite format" not in response.content


# --- Summaries ----------------------------------------------------------------------


def test_summarizing_uses_the_papers_content_and_stores_the_result(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_text(SUMMARY)

    response = client.post(f"/papers/{paper_id}/summary", headers=HTMX)

    assert response.status_code == 200
    assert BODY_FACT in gemini.prompt()
    assert db.get_paper(paper_id)["summary"] == SUMMARY


def test_the_summary_comes_back_as_rendered_markdown_in_a_partial(client, gemini, add_paper):
    gemini.reply_text(SUMMARY)

    response = client.post(f"/papers/{add_paper()}/summary", headers=HTMX)

    assert "<html" not in response.text
    page = parse_html(response.text)
    summary = page.find(class_="summary")
    assert summary.find("strong").text == "TL;DR:"
    assert summary.find("h3").text == "Problem"
    assert [item.text for item in summary.find_all("li")] == ["One point", "Another point"]
    assert [button.text for button in page.find_all("button")] == ["Regenerate"]


def test_regenerating_overwrites_the_old_summary(client, gemini, add_paper):
    paper_id = add_paper()
    db.set_summary(paper_id, "Old summary.")
    gemini.reply_text("New summary.")

    response = client.post(f"/papers/{paper_id}/summary", headers=HTMX)

    assert parse_html(response.text).find(class_="summary").text == "New summary."
    assert db.get_paper(paper_id)["summary"] == "New summary."


def test_summarizing_without_a_key_explains_and_stores_nothing(client, add_paper):
    paper_id = add_paper()

    response = client.post(f"/papers/{paper_id}/summary", headers=HTMX)

    assert response.status_code == 503
    page = parse_html(response.text)
    assert "No Gemini API key" in page.find(class_="alert-error").text
    assert page.find("button").text == "Summarize paper"
    assert db.get_paper(paper_id)["summary"] is None


def test_a_failed_summary_shows_the_error_and_stores_nothing(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_error(*QUOTA)

    response = client.post(f"/papers/{paper_id}/summary", headers=HTMX)

    assert response.status_code == 429
    assert "test-model" in parse_html(response.text).find(class_="alert-error").text
    assert db.get_paper(paper_id)["summary"] is None


def test_a_failed_regeneration_keeps_the_old_summary(client, gemini, add_paper):
    paper_id = add_paper()
    db.set_summary(paper_id, "Old summary.")
    gemini.reply_error(500, "Internal error.", "INTERNAL")

    response = client.post(f"/papers/{paper_id}/summary", headers=HTMX)

    assert response.status_code == 503
    page = parse_html(response.text)
    assert page.find(class_="alert-error") is not None
    assert page.find(class_="summary").text == "Old summary."
    assert db.get_paper(paper_id)["summary"] == "Old summary."


def test_an_empty_answer_from_gemini_stores_nothing(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_text("   ")

    response = client.post(f"/papers/{paper_id}/summary", headers=HTMX)

    assert response.status_code == 502
    assert db.get_paper(paper_id)["summary"] is None


# --- Model output cannot inject markup ----------------------------------------------

NASTY_MARKDOWN = (
    "<script>alert('x')</script>\n\n"
    "<img src=x onerror=alert(1)>\n\n"
    "[click me](javascript:alert(1))\n\n"
    '<a href="https://evil.example.com">raw link</a>'
)


def assert_no_injected_markup(fragment) -> None:
    assert fragment.find("script") is None
    assert fragment.find("img") is None
    for link in fragment.find_all("a"):
        assert not link.attrs.get("href", "").startswith("javascript:")
        assert "evil.example.com" not in link.attrs.get("href", "")


def test_raw_html_in_a_summary_is_shown_as_text(client, gemini, add_paper):
    gemini.reply_text(NASTY_MARKDOWN)
    response = client.post(f"/papers/{add_paper()}/summary", headers=HTMX)
    assert "&lt;script&gt;" in response.text
    assert_no_injected_markup(parse_html(response.text).find(class_="summary"))


def test_raw_html_in_an_answer_is_shown_as_text(client, gemini, add_paper):
    gemini.reply_text(NASTY_MARKDOWN)
    response = client.post(f"/papers/{add_paper()}/questions", data={"question": "Why?"}, headers=HTMX)
    assert "&lt;script&gt;" in response.text
    assert_no_injected_markup(parse_html(response.text).find(class_="answer"))


def test_a_stored_summary_is_still_safe_when_the_page_is_reloaded(client, add_paper):
    paper_id = add_paper()
    db.set_summary(paper_id, NASTY_MARKDOWN)
    assert_no_injected_markup(page_of(client, paper_id).find(class_="summary"))


def test_markdown_tables_and_code_render(client, gemini, add_paper):
    gemini.reply_text("| Model | Score |\n|---|---|\n| A | 1 |\n\nUse `pytest`.")
    response = client.post(f"/papers/{add_paper()}/questions", data={"question": "Scores?"}, headers=HTMX)
    answer = parse_html(response.text).find(class_="answer")
    assert [cell.text for cell in answer.find("table").find_all("td")] == ["A", "1"]
    assert answer.find("code").text == "pytest"


# --- Questions ----------------------------------------------------------------------


def test_asking_uses_the_papers_content_and_stores_the_pair(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_text("The **Zephyr** benchmark.")

    response = client.post(f"/papers/{paper_id}/questions", data={"question": "What datasets are used?"}, headers=HTMX)

    assert response.status_code == 200
    assert BODY_FACT in gemini.prompt()
    assert "What datasets are used?" in gemini.prompt()
    stored = db.list_questions(paper_id)
    assert [(item["question"], item["answer"]) for item in stored] == [
        ("What datasets are used?", "The **Zephyr** benchmark.")
    ]


def test_the_answer_comes_back_as_one_list_item_and_clears_old_errors(client, gemini, add_paper):
    gemini.reply_text("The **Zephyr** benchmark.")

    response = client.post(f"/papers/{add_paper()}/questions", data={"question": "Which?"}, headers=HTMX)

    assert "<html" not in response.text
    page = parse_html(response.text)
    item = page.find("li", class_="qa-item")
    assert item.find(class_="question").text == "Which?"
    assert item.find(class_="answer").find("strong").text == "Zephyr"
    cleared = page.find(id="qa-error")
    assert cleared.attrs["hx-swap-oob"] == "true"
    assert cleared.text == ""


def test_follow_up_questions_send_the_earlier_pairs(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_text("First answer.").reply_text("Second answer.")

    client.post(f"/papers/{paper_id}/questions", data={"question": "First question?"}, headers=HTMX)
    client.post(f"/papers/{paper_id}/questions", data={"question": "Follow-up?"}, headers=HTMX)

    assert "<history>" not in gemini.prompt(0)
    assert "Q: First question?\nA: First answer." in gemini.prompt(1)
    assert gemini.prompt(1).endswith("Question: Follow-up?")


def test_context_is_limited_to_the_last_six_pairs_but_all_are_kept(client, gemini, add_paper):
    paper_id = add_paper()
    for n in range(1, 8):
        db.add_question(paper_id, f"old-question-{n}?", f"old-answer-{n}.")
    gemini.reply_text("Answer.")

    client.post(f"/papers/{paper_id}/questions", data={"question": "New?"}, headers=HTMX)

    assert "old-question-1?" not in gemini.prompt()
    assert "old-question-2?" in gemini.prompt()
    assert len(db.list_questions(paper_id)) == 8


def test_each_paper_keeps_its_own_questions(client, gemini, add_paper):
    one = add_paper(arxiv_id="2401.00001")
    two = add_paper(arxiv_id="2401.00002")
    db.add_question(one, "About paper one?", "Yes.")
    gemini.reply_text("Answer.")

    client.post(f"/papers/{two}/questions", data={"question": "About paper two?"}, headers=HTMX)

    assert "About paper one?" not in gemini.prompt()
    assert [item["question"] for item in db.list_questions(one)] == ["About paper one?"]


def test_extra_spaces_in_a_question_are_removed(client, gemini, add_paper):
    paper_id = add_paper()
    gemini.reply_text("Answer.")
    client.post(f"/papers/{paper_id}/questions", data={"question": "  What   is\n this?  "}, headers=HTMX)
    assert db.list_questions(paper_id)[0]["question"] == "What is this?"


def test_a_question_of_exactly_the_limit_is_accepted(client, gemini, add_paper):
    gemini.reply_text("Answer.")
    paper_id = add_paper()
    response = client.post(f"/papers/{paper_id}/questions", data={"question": "x" * 1000}, headers=HTMX)
    assert response.status_code == 200
    assert [item["question"] for item in db.list_questions(paper_id)] == ["x" * 1000]


@pytest.mark.parametrize(
    ("question", "message"),
    [("", "Type a question"), ("   ", "Type a question"), ("x" * 1001, "under 1,000")],
    ids=["empty", "blank", "too-long"],
)
def test_bad_questions_are_refused_without_calling_gemini(client, gemini, add_paper, question, message):
    paper_id = add_paper()

    response = client.post(f"/papers/{paper_id}/questions", data={"question": question}, headers=HTMX)

    assert response.status_code == 400
    assert message in parse_html(response.text).find(class_="alert-error").text
    assert gemini.requests == []
    assert db.list_questions(paper_id) == []


def test_a_form_with_no_question_field_is_refused_before_asking_gemini(client, gemini, add_paper):
    paper_id = add_paper()
    response = client.post(f"/papers/{paper_id}/questions", headers=HTMX)
    assert response.status_code == 400
    assert parse_html(response.text).find(class_="alert-error") is not None
    assert gemini.requests == []
    assert db.list_questions(paper_id) == []


@pytest.mark.parametrize(
    ("reply", "status"),
    [(QUOTA, 429), ((404, "Model not found.", "NOT_FOUND"), 502), ((500, "Internal error.", "INTERNAL"), 503)],
    ids=["quota", "unknown-model", "server-error"],
)
def test_gemini_errors_go_under_the_form_and_store_nothing(client, gemini, add_paper, reply, status):
    paper_id = add_paper()
    gemini.reply_error(*reply)

    response = client.post(f"/papers/{paper_id}/questions", data={"question": "Why?"}, headers=HTMX)

    assert response.status_code == status
    assert response.headers["HX-Retarget"] == "#qa-error"
    assert response.headers["HX-Reswap"] == "innerHTML"
    assert parse_html(response.text).find(class_="alert-error") is not None
    assert db.list_questions(paper_id) == []


def test_asking_without_a_key_explains_and_stores_nothing(client, add_paper):
    paper_id = add_paper()

    response = client.post(f"/papers/{paper_id}/questions", data={"question": "Why?"}, headers=HTMX)

    assert response.status_code == 503
    assert response.headers["HX-Retarget"] == "#qa-error"
    assert "No Gemini API key" in parse_html(response.text).find(class_="alert-error").text
    assert db.list_questions(paper_id) == []


def test_html_in_a_question_is_escaped(client, gemini, add_paper):
    gemini.reply_text("Answer.")
    response = client.post(
        f"/papers/{add_paper()}/questions", data={"question": "<script>alert(1)</script>?"}, headers=HTMX
    )
    page = parse_html(response.text)
    assert page.find("script") is None
    assert page.find(class_="question").text == "<script>alert(1)</script>?"


def test_nothing_to_render_gives_an_empty_page_fragment():
    assert render_markdown(None) == ""
    assert render_markdown("") == ""
