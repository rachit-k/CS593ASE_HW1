"""pdfs.py: arXiv downloads, text extraction, metadata fallbacks, and files on disk (design sections 7-8)."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pymupdf
import pytest

from app import config, pdfs
from app.pdfs import PDFError
from helpers import BODY_FACT, PAPER_ABSTRACT, PAPER_TITLE, make_pdf, make_scanned_pdf, stored_pdfs

# --- Downloading from arXiv ---------------------------------------------------------


def test_download_fetches_the_pdf_from_arxiv(arxiv_pdfs):
    pdf = make_pdf()
    arxiv_pdfs.reply(200, content=pdf)

    assert pdfs.download_arxiv_pdf("2401.12345") == pdf

    request = arxiv_pdfs.requests[0]
    assert str(request.url) == "https://arxiv.org/pdf/2401.12345"
    assert request.headers["User-Agent"].startswith("AI-Research-Assistant/")
    assert request.extensions["timeout"]["read"] == 30


def test_download_follows_redirects_within_arxiv(arxiv_pdfs):
    pdf = make_pdf()
    arxiv_pdfs.reply(302, headers={"Location": "https://export.arxiv.org/pdf/2401.12345v2"}).reply(200, content=pdf)

    assert pdfs.download_arxiv_pdf("2401.12345") == pdf
    assert [request.url.host for request in arxiv_pdfs.requests] == ["arxiv.org", "export.arxiv.org"]


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.example.com/paper.pdf",
        "https://arxiv.org.evil.example/paper.pdf",
        "https://notarxiv.org/paper.pdf",
    ],
)
def test_a_redirect_off_arxiv_is_stopped_before_it_is_followed(arxiv_pdfs, target):
    arxiv_pdfs.reply(302, headers={"Location": target}).reply(200, content=make_pdf())

    with pytest.raises(PDFError, match="another site") as caught:
        pdfs.download_arxiv_pdf("2401.12345")

    assert caught.value.status_code == 502
    assert len(arxiv_pdfs.requests) == 1


@pytest.mark.parametrize("status", [403, 404, 429, 500])
def test_a_failed_download_names_the_status(arxiv_pdfs, status):
    arxiv_pdfs.reply(status)
    with pytest.raises(PDFError, match=f"HTTP {status}") as caught:
        pdfs.download_arxiv_pdf("2401.12345")
    assert caught.value.status_code == 502


def test_a_response_that_is_not_a_pdf_is_rejected(arxiv_pdfs):
    arxiv_pdfs.reply(200, text="<html>Please try later</html>")
    with pytest.raises(PDFError, match="didn't return a PDF") as caught:
        pdfs.download_arxiv_pdf("2401.12345")
    assert caught.value.status_code == 502


def test_a_download_over_50_mb_is_rejected(arxiv_pdfs):
    arxiv_pdfs.reply(200, content=b"%PDF" + b"\0" * (50 * 1024 * 1024 - 3))
    with pytest.raises(PDFError, match="larger than 50 MB") as caught:
        pdfs.download_arxiv_pdf("2401.12345")
    assert caught.value.status_code == 502


def test_a_download_of_exactly_the_limit_is_allowed(arxiv_pdfs, monkeypatch):
    monkeypatch.setattr(pdfs, "MAX_PDF_BYTES", 1000)
    content = b"%PDF" + b"\0" * 996
    arxiv_pdfs.reply(200, content=content)
    assert pdfs.download_arxiv_pdf("2401.12345") == content


@pytest.mark.parametrize(
    ("error", "message"),
    [(httpx.ReadTimeout("slow"), "timed out"), (httpx.ConnectError("no route"), "Couldn't reach arXiv")],
    ids=["timeout", "connection-error"],
)
def test_network_failures_become_friendly_errors(arxiv_pdfs, error, message):
    arxiv_pdfs.fail(error)
    with pytest.raises(PDFError, match=message) as caught:
        pdfs.download_arxiv_pdf("2401.12345")
    assert caught.value.status_code == 502


# --- Is it a usable PDF? ------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"%PDF-1.7 rest of the file", True),
        (b"some junk first %PDF-1.7", True),
        (b"", False),
        (b"just a text file", False),
        (b"x" * 1024 + b"%PDF-1.7", False),
    ],
    ids=["header", "junk-then-header", "empty", "text", "header-too-late"],
)
def test_looks_like_pdf(data, expected):
    assert pdfs.looks_like_pdf(data) is expected


def test_a_file_that_is_not_a_pdf_is_rejected():
    with pytest.raises(PDFError, match="isn't a PDF") as caught:
        pdfs.extract(b"just some text")
    assert caught.value.status_code == 400


@pytest.mark.parametrize(
    "data",
    [b"%PDF-1.7\n" + b"garbage " * 200, b"this mentions %PDF but is only text " * 20, make_pdf()[:300]],
    ids=["garbage-after-header", "text-mentioning-pdf", "truncated"],
)
def test_a_damaged_pdf_is_rejected_with_a_user_error(data):
    with pytest.raises(PDFError) as caught:
        pdfs.extract(data)
    assert caught.value.status_code == 400


def test_a_password_protected_pdf_is_rejected():
    with pytest.raises(PDFError, match="password") as caught:
        pdfs.extract(make_pdf(password="secret"))
    assert caught.value.status_code == 400


def test_a_scanned_pdf_is_rejected():
    with pytest.raises(PDFError, match="no readable text") as caught:
        pdfs.extract(make_scanned_pdf())
    assert caught.value.status_code == 400


def test_fewer_than_200_characters_counts_as_scanned():
    with pytest.raises(PDFError, match="no readable text"):
        pdfs.extract(make_pdf(["abcdefghij" * 19 + "abcdefghi"]))


def test_exactly_200_characters_is_enough():
    extracted = pdfs.extract(make_pdf(["abcdefghij" * 20]))
    assert len(re.sub(r"\s", "", extracted.full_text)) == 200


# --- Extracting text ----------------------------------------------------------------


def test_full_text_includes_every_page():
    text = pdfs.extract(make_pdf()).full_text
    assert PAPER_TITLE in text
    assert "2 Method" in text
    assert BODY_FACT in text


def test_first_pages_text_stops_after_two_pages():
    first_pages = pdfs.extract(make_pdf()).first_pages_text
    assert PAPER_TITLE in first_pages
    assert "2 Method" in first_pages
    assert BODY_FACT not in first_pages


def test_first_pages_text_is_capped_at_8000_characters():
    extracted = pdfs.extract(make_pdf(["lorem ipsum " * 370, "dolor sit amet " * 300, BODY_FACT]))
    assert len(extracted.first_pages_text) == 8000
    assert len(extracted.full_text) > 8000


def test_extracted_text_has_no_runs_of_blank_lines():
    text = pdfs.extract(make_pdf(["First line\n\n\n\n\nLater line " + "with more words " * 20])).full_text
    assert "\n\n\n" not in text


# --- The PDF's own metadata ---------------------------------------------------------


@pytest.mark.parametrize(
    ("embedded", "expected"),
    [
        ("Deep Learning for Code Review", "Deep Learning for Code Review"),
        ("  Deep   Learning for Review ", "Deep Learning for Review"),
        ("", None),
        ("Untitled", None),
        ("Untitled document", None),
        ("Untitled-1", None),
        ("manuscript", None),
        ("Microsoft Word - draft3", None),
        ("final_paper_v2.docx", None),
        ("my_long_thesis.pdf", None),
        ("main_paper.tex", None),
        ("main_paper.dvi", None),
        ("Short", None),
    ],
)
def test_embedded_title_is_used_only_when_it_looks_real(embedded, expected):
    assert pdfs.extract(make_pdf(title=embedded)).embedded_title == expected


def test_a_pdf_without_metadata_has_no_embedded_title_or_authors():
    extracted = pdfs.extract(make_pdf())
    assert extracted.embedded_title is None
    assert extracted.embedded_authors == []


@pytest.mark.parametrize(
    ("embedded", "expected"),
    [
        ("Alice Smith", ["Alice Smith"]),
        ("Alice Smith; Bob Jones", ["Alice Smith", "Bob Jones"]),
        ("Alice Smith, Bob Jones", ["Alice Smith", "Bob Jones"]),
        ("Alice Smith and Bob Jones", ["Alice Smith", "Bob Jones"]),
        ("Alice Smith, Bob Jones, and Carol White", ["Alice Smith", "Bob Jones", "Carol White"]),
        ("Andrea Anderson", ["Andrea Anderson"]),
        ("", []),
    ],
)
def test_embedded_authors_are_split_into_names(embedded, expected):
    assert pdfs.extract(make_pdf(author=embedded)).embedded_authors == expected


# --- Text-based fallbacks -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A Real Paper Title\nAlice Smith", "A Real Paper Title"),
        ("arXiv:2401.12345v2 [cs.SE] 15 Jan 2024\nA Real Paper Title", "A Real Paper Title"),
        ("12\n\n   \nA Real Paper Title", "A Real Paper Title"),
        ("1234567890 12345\nA Real Paper Title", "A Real Paper Title"),
        ("  A   Spaced    Out  Title \nAuthor", "A Spaced Out Title"),
        ("Обработка естественного языка\nИван Петров", "Обработка естественного языка"),
        ("量子计算研究\n李明", "量子计算研究"),
        ("Draft\nA Real Paper Title", "A Real Paper Title"),
        ("12\n34\n", None),
        ("© 2024\n(1)\n", None),
        ("", None),
    ],
    ids=[
        "first-line",
        "arxiv-stamp",
        "page-number",
        "digits-only",
        "spaces",
        "non-latin",
        "short-chinese",
        "short-latin-line",
        "nothing",
        "symbols-and-digits",
        "empty",
    ],
)
def test_title_from_text_finds_the_first_real_line(text, expected):
    assert pdfs.title_from_text(text) == expected


def test_title_from_text_is_capped_at_300_characters():
    assert len(pdfs.title_from_text("word " * 200)) == 300


# abstract_from_text ignores anything under 100 characters, so this sample must be longer.
ABSTRACT = "We present a method for checking programs. It finds bugs that other tools miss in large projects, at scale."


@pytest.mark.parametrize(
    "heading",
    ["Abstract\n", "ABSTRACT\n", "Abstract: ", "Abstract. ", "Abstract—", "Abstract - ", "Abstract ", "  Abstract\n\n"],
    ids=["own-line", "capitals", "colon", "period", "dash", "spaced-hyphen", "runs-into-the-text", "indented"],
)
def test_abstract_is_found_after_common_headings(heading):
    text = f"A Paper Title\nAlice Smith\n{heading}{ABSTRACT}\n1 Introduction\nPrograms are hard to check."
    assert pdfs.abstract_from_text(text) == ABSTRACT


@pytest.mark.parametrize(
    "next_heading",
    ["1 Introduction", "1. Introduction", "I. INTRODUCTION", "Introduction", "Keywords: testing", "Index Terms: x", "CCS Concepts: x"],
)
def test_abstract_stops_at_the_next_heading(next_heading):
    text = f"Abstract\n{ABSTRACT}\n{next_heading}\nThe rest of the paper."
    assert pdfs.abstract_from_text(text) == ABSTRACT


def test_abstract_lines_are_joined():
    wrapped = ABSTRACT.replace(" for ", "\nfor ").replace(" bugs ", "\nbugs ")
    assert pdfs.abstract_from_text(f"Abstract\n{wrapped}\n1 Introduction") == ABSTRACT


@pytest.mark.parametrize(
    "opening",
    [
        "Abstract Interpretation of Neural Networks\nAlice Smith\n",
        "On Abstract Syntax Trees\nAlice Smith\n",
        "A Framework for\nAbstract Interpretation\nAlice Smith\n",
        "Abstract-Based Verification of Programs\nAlice Smith\n",
    ],
    ids=["title-starts-with-it", "title-contains-it", "wrapped-title", "hyphenated-title"],
)
def test_the_word_abstract_in_the_title_is_not_the_heading(opening):
    text = f"{opening}Abstract\n{ABSTRACT}\n1 Introduction\nBody."
    assert pdfs.abstract_from_text(text) == ABSTRACT


@pytest.mark.parametrize(
    "opening",
    [
        "Abstract Interpretation on Graphs\nAlice Smith\n",
        "Abstract-Based Verification of Programs\nAlice Smith\n",
        "A Paper Title\nWe choose a suitable\nabstract domain for this. It has a natural meaning for each class.\n",
        "A Paper Title\nThis extended\nabstract. It has a natural meaning for each class of the programs we study.\n",
    ],
    ids=["title-starts-with-it", "hyphenated-title", "wrapped-sentence", "wrapped-sentence-with-period"],
)
def test_a_paper_without_a_heading_has_no_abstract_whatever_its_words_are(opening):
    # Modeled on a real extended abstract (arXiv 2203.09191), which has no heading at all.
    text = f"{opening}{ABSTRACT}\n1 Introduction\nBody."
    assert pdfs.abstract_from_text(text) is None


def test_the_word_abstract_in_a_sentence_is_not_the_heading():
    text = f"We compare abstract syntax trees.\nAbstract: {ABSTRACT}\n1 Introduction\nBody."
    assert pdfs.abstract_from_text(text) == ABSTRACT


def test_no_abstract_heading_means_no_abstract():
    assert pdfs.abstract_from_text(f"A Paper Title\n{ABSTRACT}\n1 Introduction") is None


def test_a_very_short_abstract_is_not_trusted():
    assert pdfs.abstract_from_text("Abstract\nToo short to be real.\n1 Introduction") is None


def test_an_abstract_needs_at_least_100_characters():
    assert pdfs.abstract_from_text(f"Abstract\n{'x' * 99}\n1 Introduction") is None
    assert pdfs.abstract_from_text(f"Abstract\n{'x' * 100}\n1 Introduction") == "x" * 100


def test_a_very_long_abstract_is_cut_at_a_sentence_end():
    abstract = pdfs.abstract_from_text("Abstract\n" + "This sentence repeats again and again. " * 200)
    assert 100 <= len(abstract) <= 3000
    assert abstract.endswith("again.")


def test_the_real_pdf_layout_gives_the_abstract():
    first_pages = pdfs.extract(make_pdf()).first_pages_text
    assert pdfs.abstract_from_text(first_pages) == PAPER_ABSTRACT


# --- Files on disk ------------------------------------------------------------------


def test_saved_pdfs_get_generated_names_in_the_pdf_folder():
    name = pdfs.save_pdf(b"%PDF contents")
    assert re.fullmatch(r"[0-9a-f]{32}\.pdf", name)
    assert (config.settings.pdf_dir / name).read_bytes() == b"%PDF contents"
    assert pdfs.save_pdf(b"%PDF contents") != name


def test_save_recreates_a_missing_pdf_folder():
    config.settings.pdf_dir.rmdir()
    name = pdfs.save_pdf(b"%PDF contents")
    assert (config.settings.pdf_dir / name).is_file()


@pytest.mark.parametrize("name", ["../../secret.txt", "/etc/passwd", "sub/dir/file.pdf"])
def test_paths_cannot_leave_the_pdf_folder(name):
    assert pdfs.pdf_path(name).parent == config.settings.pdf_dir


def test_delete_removes_the_file():
    name = pdfs.save_pdf(b"%PDF contents")
    pdfs.delete_pdf(name)
    assert stored_pdfs() == []


def test_deleting_a_missing_file_leaves_the_other_files_alone():
    kept = pdfs.save_pdf(make_pdf())
    pdfs.delete_pdf("not-there.pdf")
    assert [path.name for path in stored_pdfs()] == [kept]


def test_delete_cannot_remove_a_file_outside_the_pdf_folder():
    victim = config.settings.data_dir / "victim.pdf"
    victim.write_bytes(b"keep me")
    pdfs.delete_pdf("../victim.pdf")
    assert victim.read_bytes() == b"keep me"


# --- Rare paths, reached by injecting the failure ------------------------------------


def test_text_that_cannot_be_read_after_opening_is_a_user_error(monkeypatch):
    def broken(_page, *_args, **_kwargs):
        raise RuntimeError("damaged content stream")

    monkeypatch.setattr(pymupdf.Page, "get_text", broken)
    with pytest.raises(PDFError, match="couldn't be read") as caught:
        pdfs.extract(make_pdf())
    assert caught.value.status_code == 400


def test_a_pdf_that_reports_no_metadata_is_still_usable(monkeypatch):
    real_open = pymupdf.open

    def open_without_metadata(*args, **kwargs):
        doc = real_open(*args, **kwargs)
        doc.metadata = None
        return doc

    monkeypatch.setattr(pymupdf, "open", open_without_metadata)
    extracted = pdfs.extract(make_pdf(title="A Real Looking Title", author="Alice Smith"))
    assert extracted.embedded_title is None
    assert extracted.embedded_authors == []
    assert BODY_FACT in extracted.full_text


def test_a_file_that_cannot_be_deleted_is_left_behind_quietly(monkeypatch):
    name = pdfs.save_pdf(b"%PDF contents")

    def refuse(_path, *_args, **_kwargs):
        raise PermissionError("the file is locked")

    monkeypatch.setattr(Path, "unlink", refuse)
    pdfs.delete_pdf(name)
    assert (config.settings.pdf_dir / name).is_file()


def test_a_very_long_abstract_with_no_sentence_break_is_cut_at_the_limit():
    abstract = pdfs.abstract_from_text("Abstract\n" + "word " * 1000)
    assert len(abstract) == 3000
    assert abstract.startswith("word word")


# --- What real first pages print around a title and an abstract (see test_real_papers.py) ------

ABSTRACT_SENTENCES = "We study how people review code in large projects, and we measure what changes.\nWe report what we learned from 200 reviews over a whole year of work.\n"


@pytest.mark.parametrize(
    "line",
    [
        "arXiv:2101.10585v2  [cs.SE]  12 Aug 2021",
        "Noname manuscript No.",
        "Noname manuscript no",
        "Empirical Software Engineering manuscript No.",
        "(will be inserted by the editor)",
        "Published as a conference paper at ICLR 2015",
    ],
)
def test_a_stamp_or_template_line_above_the_title_is_skipped(line):
    assert pdfs.title_from_text(f"{line}\nA Real Title About Reviews\n") == "A Real Title About Reviews"


@pytest.mark.parametrize(
    "title",
    [
        "The Voynich Manuscript: No Longer a Mystery",
        "Why Every manuscript Now Needs a Data Statement",
        "Will It Be Inserted by the Editor? Studying Peer Review",
        "Conference Papers Published as Journal Articles",
    ],
)
def test_a_title_that_only_resembles_a_template_line_is_kept(title):
    assert pdfs.title_from_text(f"{title}\nAnother line\n") == title


@pytest.mark.parametrize(
    "footer",
    [
        "∗Equal contribution.",
        "⋆These authors contributed equally.",
        "†Work performed while at Google Brain.",
        "‡Work performed while at Google Research.",
        "*Corresponding author.",
        "E-mail: someone@example.org",
        "email: someone@example.org",
        "Email : someone@example.org",
        "Tel.: +1-313-577-0731",
        "arXiv:2101.10585v2  [cs.SE]  12 Aug 2021",
        "31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.",
        "2nd Conference on Machine Learning",
        "3rd conference on Systems",
        "14th Conference on Testing",
    ],
)
def test_a_page_footer_ends_the_abstract(footer):
    text = f"Title\nAbstract\n{ABSTRACT_SENTENCES}{footer}\nMore text after the footer.\n1 Introduction\nBody."
    assert pdfs.abstract_from_text(text) == " ".join(ABSTRACT_SENTENCES.split())


@pytest.mark.parametrize(
    "line",
    [
        "email is the main way people discuss code, and we also looked at it.",
        "Tel Aviv University hosted the second half of this study of reviews.",
        "arXiv is where we found the papers that we use for comparison here.",
        "2 conferences hosted this work, so it reached many people in the area.",
        "conference talks were part of the data that we gathered for the study.",
    ],
)
def test_a_sentence_that_only_starts_like_a_footer_stays_in_the_abstract(line):
    text = f"Title\nAbstract\n{ABSTRACT_SENTENCES}{line}\n1 Introduction\nBody."
    assert pdfs.abstract_from_text(text) == " ".join((ABSTRACT_SENTENCES + line).split())


def test_author_lines_before_a_footer_are_dropped_with_it():
    text = f"Title\nAbstract\n{ABSTRACT_SENTENCES}A. Author\nDepartment of Computing, Some University\nE-mail: a@b.org\n"
    assert pdfs.abstract_from_text(text) == " ".join(ABSTRACT_SENTENCES.split())


def test_periods_inside_an_author_line_do_not_end_the_abstract_there():
    text = f"Title\nAbstract\n{ABSTRACT_SENTENCES}A. B. Author, Dept. of Computing\nE-mail: a@b.org\n"
    assert pdfs.abstract_from_text(text) == " ".join(ABSTRACT_SENTENCES.split())


def test_text_before_a_footer_with_no_finished_sentence_is_kept_whole():
    unfinished = "We study how people review code in large projects and we measure what changes in a whole year of their work"
    assert pdfs.abstract_from_text(f"Title\nAbstract\n{unfinished}\nE-mail: a@b.org\n") == unfinished


def test_a_footer_right_after_the_heading_leaves_no_abstract():
    assert pdfs.abstract_from_text("Title\nAbstract\nE-mail: someone@example.org\nThe rest of the page.\n") is None


@pytest.mark.parametrize(
    "value",
    ["arXiv:2404.01484v2  [cs.PL]  26 Sep 2024", "Noname manuscript No.", "Published as a conference paper at ICLR 2015"],
)
def test_a_stamp_in_the_pdfs_title_field_is_not_taken_for_the_title(value):
    extracted = pdfs.extract(make_pdf(title=value))
    assert extracted.embedded_title is None
    assert pdfs.title_from_text(extracted.first_pages_text) == PAPER_TITLE


@pytest.mark.parametrize(
    ("field", "names"),
    [
        ("Ada Lovelace someone@example.org Grace Hopper, Alan Turing", ["Ada Lovelace", "Grace Hopper", "Alan Turing"]),
        ("Ada Lovelace; someone@example.org; Grace Hopper", ["Ada Lovelace", "Grace Hopper"]),
        ("Ada Lovelace someone@example.org Grace Hopper; Alan Turing", ["Ada Lovelace", "Grace Hopper", "Alan Turing"]),
        ("someone@example.org", []),
    ],
    ids=["between-names", "in-a-semicolon-list", "beside-a-semicolon", "nothing-else"],
)
def test_an_email_address_in_the_author_field_only_separates_names(field, names):
    assert pdfs._embedded_authors(field) == names


# --- Found by mutation testing -------------------------------------------------------------------


def test_a_refused_download_has_the_whole_message(arxiv_pdfs):
    arxiv_pdfs.reply(404)
    with pytest.raises(PDFError) as caught:
        pdfs.download_arxiv_pdf("2401.12345")
    assert str(caught.value) == "Couldn't download the PDF from arXiv (HTTP 404). Try again in a moment."
    assert caught.value.status_code == 502


def test_the_download_is_a_plain_get(arxiv_pdfs):
    arxiv_pdfs.reply(200, content=make_pdf())
    pdfs.download_arxiv_pdf("2401.12345")
    assert arxiv_pdfs.requests[0].method == "GET"


def test_pages_are_kept_apart_by_a_blank_line():
    extracted = pdfs.extract(make_pdf(["End of the first page.", "Start of the second page. " * 10]))
    assert "End of the first page.\n\nStart of the second page." in extracted.full_text
    assert "End of the first page.\n\nStart of the second page." in extracted.first_pages_text


def test_tidying_removes_null_characters_and_evens_out_line_endings():
    assert pdfs._tidy("a\x00b\r\nc\rd\n\n\n\n\ne") == "ab\nc\nd\n\ne"


@pytest.mark.parametrize("value", ["Document", "MANUSCRIPT"])
def test_a_long_placeholder_word_is_not_a_title(value):
    assert pdfs._embedded_title(value) is None


LONG_LINE = "is a key idea in computing, and this line goes on for long enough to pass as an abstract if a rule let it through."


def test_the_word_abstract_inside_a_sentence_is_not_a_heading():
    assert pdfs.abstract_from_text(f"A Title\nThen we read the Abstract. This {LONG_LINE}\n1 Introduction\nBody.") is None


def test_a_line_starting_with_a_longer_word_is_not_a_run_in_heading():
    assert pdfs.abstract_from_text(f"A Title\nAbstraction {LONG_LINE}\n1 Introduction\nBody.") is None


def test_an_abstract_of_exactly_the_limit_is_kept_whole():
    body = "A short first sentence. " + "x" * (pdfs.MAX_FALLBACK_ABSTRACT_CHARS - 24)
    assert len(body) == pdfs.MAX_FALLBACK_ABSTRACT_CHARS
    assert pdfs.abstract_from_text(f"A Title\nAbstract\n{body}\n1 Introduction\nBody.") == body


def test_saving_a_pdf_creates_any_missing_folders(monkeypatch, tmp_path):
    folder = tmp_path / "not" / "there" / "yet"
    monkeypatch.setattr(config, "settings", config.Settings(None, "test-model", folder))
    name = pdfs.save_pdf(b"%PDF-1.7")
    assert (folder / "pdfs" / name).read_bytes() == b"%PDF-1.7"
