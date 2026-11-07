"""Golden and differential testing: the upload fallbacks on real papers, checked against arXiv's metadata.

Synthetic text shares the assumptions of whoever wrote it. These are 29 real PDFs in the layouts
people actually publish in, and arXiv's own metadata says what the right answer is.
"""

from __future__ import annotations

import difflib
import json
import re

import pytest

from app import pdfs
from app.main import upload_metadata
from helpers import RECORDINGS

PAPERS = json.loads((RECORDINGS / "real_papers.json").read_text())

# These really have no "Abstract" heading, so no answer is the right answer.
NO_ABSTRACT_HEADING = {"2404.01484", "2203.09191"}
# These abstracts continue on page 2, after a page-1 footer. The fallback stops at the footer.
CONTINUES_ON_PAGE_2 = {"1811.04169", "2007.12580", "2010.02472", "2107.07944", "2305.14837", "2308.11258"}
# arXiv's copy of these abstracts is worded differently from the one in the PDF.
REWORDED_ON_ARXIV = {"1810.04805", "2005.11583", "2111.00875"}
# Known limits of the title fallback, left out of the title check. No simple rule can tell these from a real title.
WRONG_TITLES = {
    "1706.03762": "the PDF's first line is a notice from Google, in plain prose",
    "2104.14787": 'the authors set the PDF\'s title field to their own label, "UBayFS preprint"',
}


def normalize(text: str) -> str:
    """Compare text as a reader would: ignore case, ligatures, line-break hyphens, and spacing."""
    for ligature, letters in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl")):
        text = text.replace(ligature, letters)
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def shown_after_upload(paper: dict) -> dict:
    """What a user sees after uploading this paper with Gemini off, using the route's own decision."""
    extracted = pdfs.ExtractedPDF(
        full_text=paper["text"],
        first_pages_text=paper["text"],
        embedded_title=pdfs._embedded_title(paper["embedded_title"]),
        embedded_authors=pdfs._embedded_authors(paper["embedded_author"]),
    )
    metadata, used_fallbacks = upload_metadata(extracted)
    assert used_fallbacks
    return metadata


TITLE_CASES = sorted(set(PAPERS) - set(WRONG_TITLES))


@pytest.mark.parametrize("arxiv_id", TITLE_CASES)
def test_the_title_is_right_or_the_first_line_of_a_wrapped_one(arxiv_id):
    paper = PAPERS[arxiv_id]
    title, official = normalize(shown_after_upload(paper)["title"]), normalize(paper["official_title"])
    assert official.startswith(title), f"got {title!r}"
    assert len(title) >= 0.3 * len(official), f"got only {title!r}"


@pytest.mark.parametrize("arxiv_id", sorted(PAPERS))
def test_every_author_shown_is_a_real_author(arxiv_id):
    # The embedded field often lists only some of the authors, or none, and that's fine. Nobody else may appear.
    paper = PAPERS[arxiv_id]
    official = {normalize(name) for name in paper["official_authors"]}
    assert {normalize(name) for name in shown_after_upload(paper)["authors"]} <= official


def test_the_sample_has_real_embedded_titles_and_authors():
    shown = [shown_after_upload(paper) for paper in PAPERS.values()]
    assert sum(bool(paper["embedded_title"]) for paper in PAPERS.values()) >= 5
    assert sum(bool(metadata["authors"]) for metadata in shown) >= 3


@pytest.mark.parametrize("arxiv_id", sorted(set(PAPERS) - NO_ABSTRACT_HEADING))
def test_the_abstract_is_arxivs_abstract_and_nothing_else(arxiv_id):
    paper = PAPERS[arxiv_id]
    got, official = normalize(pdfs.abstract_from_text(paper["text"]) or ""), normalize(paper["official_abstract"])
    assert got
    matching = sum(block.size for block in difflib.SequenceMatcher(None, got, official, autojunk=False).get_matching_blocks())
    precision, recall = matching / len(got), matching / len(official)

    assert precision >= (0.85 if arxiv_id in REWORDED_ON_ARXIV else 0.98), f"{1 - precision:.0%} of it isn't the abstract"
    if arxiv_id in CONTINUES_ON_PAGE_2:
        assert official.startswith(got[:200]) and recall >= 0.4, f"only {recall:.0%} of the abstract"
    else:
        assert recall >= (0.85 if arxiv_id in REWORDED_ON_ARXIV else 0.98), f"only {recall:.0%} of the abstract"


@pytest.mark.parametrize("arxiv_id", sorted(NO_ABSTRACT_HEADING))
def test_a_paper_with_no_abstract_heading_gets_no_abstract_rather_than_a_guess(arxiv_id):
    assert pdfs.abstract_from_text(PAPERS[arxiv_id]["text"]) is None


@pytest.mark.parametrize("arxiv_id", sorted(CONTINUES_ON_PAGE_2))
def test_an_abstract_split_by_a_page_break_has_no_contact_details_in_it(arxiv_id):
    assert not re.search(r"\[email\]|E-mail|Tel\.|arXiv:\d|contributed equally|University", pdfs.abstract_from_text(PAPERS[arxiv_id]["text"]))


def test_the_sample_covers_the_layouts_that_matter():
    texts = [paper["text"] for paper in PAPERS.values()]
    assert sum("\nAbstract " in text for text in texts) >= 10  # "Abstract We present...", run into the first sentence
    assert sum(bool(re.search(r"^\s*(Abstract|ABSTRACT)\s*$", text, re.M)) for text in texts) >= 8  # a heading on its own line
    assert sum("manuscript No." in text for text in texts) >= 10  # the Springer template
    assert sum(paper["official_title"].startswith("Abstract") for paper in PAPERS.values()) >= 3  # a trap for the heading rule
    assert sum(text.lstrip().startswith("arXiv:") for text in texts) >= 3  # the stamp comes first
