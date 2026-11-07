"""db.py: the schema, every query, and the library's invariants (design section 5)."""

from __future__ import annotations

import sqlite3

import pytest

from app import config, db


def arxiv_paper(**overrides):
    return {
        "source": "arxiv",
        "arxiv_id": "2401.12345",
        "title": "A Paper",
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2024,
        "abstract": "An abstract.",
        "url": "https://arxiv.org/abs/2401.12345",
        "pdf_path": "abc.pdf",
        "full_text": "The full text.",
    } | overrides


def uploaded_paper(**overrides):
    return arxiv_paper(source="upload", arxiv_id=None, url=None, year=None, authors=[]) | overrides


def test_init_db_creates_the_folders_and_an_empty_library(tmp_path, monkeypatch):
    settings = config.Settings(None, "model", tmp_path / "new" / "folder")
    monkeypatch.setattr(config, "settings", settings)
    db.init_db()
    assert settings.db_path.is_file()
    assert settings.pdf_dir.is_dir()
    assert db.list_papers() == []


def test_init_db_keeps_existing_papers():
    db.add_paper(**arxiv_paper())
    db.init_db()
    assert db.count_papers() == 1


def test_a_paper_reads_back_exactly_as_stored():
    paper_id = db.add_paper(**arxiv_paper())
    paper = db.get_paper(paper_id)
    assert {name: paper[name] for name in arxiv_paper()} == arxiv_paper()
    assert paper["id"] == paper_id
    assert paper["summary"] is None
    assert paper["created_at"]


def test_an_upload_can_have_no_arxiv_id_url_year_or_authors():
    paper = db.get_paper(db.add_paper(**uploaded_paper()))
    assert paper["arxiv_id"] is None
    assert paper["url"] is None
    assert paper["year"] is None
    assert paper["authors"] == []


def test_a_missing_paper_is_none():
    assert db.get_paper(999) is None


@pytest.mark.parametrize(
    "paper_id",
    [0, -1, 2**63 - 1, -(2**63), 2**63, -(2**63) - 1, 10**30],
    ids=["zero", "negative", "largest-integer", "smallest-integer", "too-large", "too-small", "far-too-large"],
)
def test_an_id_that_cannot_exist_is_simply_not_found(paper_id):
    # SQLite integers are 64-bit, and passing it anything larger raises OverflowError.
    assert db.get_paper(paper_id) is None
    assert db.delete_paper(paper_id) is None


@pytest.mark.parametrize("text", ["", "   \n\t "], ids=["empty", "whitespace"])
def test_every_paper_must_have_full_text(text):
    with pytest.raises(ValueError, match="full text"):
        db.add_paper(**arxiv_paper(full_text=text))
    assert db.count_papers() == 0


def test_the_same_arxiv_paper_cannot_be_stored_twice():
    db.add_paper(**arxiv_paper())
    with pytest.raises(sqlite3.IntegrityError):
        db.add_paper(**arxiv_paper(title="Same paper, another title"))
    assert db.count_papers() == 1


def test_uploads_are_not_duplicates_of_each_other():
    db.add_paper(**uploaded_paper())
    db.add_paper(**uploaded_paper())
    assert db.count_papers() == 2


def test_source_must_be_arxiv_or_upload():
    with pytest.raises(sqlite3.IntegrityError):
        db.add_paper(**arxiv_paper(source="somewhere-else"))


def test_list_is_newest_first_and_leaves_out_the_large_columns():
    first = db.add_paper(**arxiv_paper())
    second = db.add_paper(**uploaded_paper(title="Newer"))
    papers = db.list_papers()
    assert [paper["id"] for paper in papers] == [second, first]
    assert papers[1]["authors"] == ["Alice Smith", "Bob Jones"]
    assert "full_text" not in papers[0]
    assert "summary" not in papers[0]


def test_list_reports_summary_and_question_counts():
    paper_id = db.add_paper(**arxiv_paper())
    assert db.list_papers()[0]["has_summary"] == 0
    assert db.list_papers()[0]["question_count"] == 0
    db.set_summary(paper_id, "A summary.")
    db.add_question(paper_id, "Q1", "A1")
    db.add_question(paper_id, "Q2", "A2")
    assert db.list_papers()[0]["has_summary"] == 1
    assert db.list_papers()[0]["question_count"] == 2


def test_find_by_arxiv_id():
    paper_id = db.add_paper(**arxiv_paper())
    assert db.find_by_arxiv_id("2401.12345") == paper_id
    assert db.find_by_arxiv_id("2401.99999") is None


def test_library_ids_for_maps_only_papers_in_the_library():
    paper_id = db.add_paper(**arxiv_paper())
    assert db.library_ids_for(["2401.12345", "2401.99999"]) == {"2401.12345": paper_id}
    assert db.library_ids_for(["2401.99999"]) == {}
    assert db.library_ids_for([]) == {}


def test_delete_removes_the_paper_and_only_its_own_questions():
    doomed = db.add_paper(**arxiv_paper(pdf_path="doomed.pdf"))
    kept = db.add_paper(**uploaded_paper())
    db.add_question(doomed, "Doomed question", "Answer")
    db.add_question(kept, "Kept question", "Answer")

    assert db.delete_paper(doomed) == "doomed.pdf"

    assert db.get_paper(doomed) is None
    assert db.list_questions(doomed) == []
    assert [item["question"] for item in db.list_questions(kept)] == ["Kept question"]


def test_deleting_a_missing_paper_returns_none():
    assert db.delete_paper(999) is None


def test_set_summary_overwrites_the_old_one():
    paper_id = db.add_paper(**arxiv_paper())
    db.set_summary(paper_id, "First")
    db.set_summary(paper_id, "Second")
    assert db.get_paper(paper_id)["summary"] == "Second"


def test_add_question_returns_the_stored_row():
    paper_id = db.add_paper(**arxiv_paper())
    item = db.add_question(paper_id, "What?", "That.")
    assert item["paper_id"] == paper_id
    assert item["question"] == "What?"
    assert item["answer"] == "That."
    assert item["id"]
    assert item["created_at"]


def test_questions_come_back_oldest_first_for_one_paper():
    paper_id = db.add_paper(**arxiv_paper())
    other = db.add_paper(**uploaded_paper())
    db.add_question(paper_id, "First", "A")
    db.add_question(other, "Other paper", "A")
    db.add_question(paper_id, "Second", "A")
    assert [item["question"] for item in db.list_questions(paper_id)] == ["First", "Second"]


def test_a_question_needs_an_existing_paper():
    with pytest.raises(sqlite3.IntegrityError):
        db.add_question(999, "Orphan?", "Yes")


def test_a_failed_block_rolls_back():
    with pytest.raises(RuntimeError), db.connect() as conn:
        conn.execute(
            "INSERT INTO papers (source, title, authors, pdf_path, full_text) VALUES ('upload', 'T', '[]', 'p', 'x')"
        )
        raise RuntimeError("something went wrong mid-way")
    assert db.count_papers() == 0


def test_a_stored_paper_with_a_blank_author_field_reads_as_no_authors():
    paper_id = db.add_paper(**arxiv_paper())
    with db.connect() as conn:
        conn.execute("UPDATE papers SET authors = '' WHERE id = ?", (paper_id,))
    assert db.get_paper(paper_id)["authors"] == []


# --- Found by mutation testing -------------------------------------------------------------------

LARGEST_ID = 2**63 - 1  # what SQLite can hold


def test_the_largest_possible_id_still_finds_and_deletes_its_paper():
    paper_id = db.add_paper(**arxiv_paper())
    with db.connect() as conn:
        conn.execute("UPDATE papers SET id = ? WHERE id = ?", (LARGEST_ID, paper_id))

    assert db.get_paper(LARGEST_ID)["title"] == arxiv_paper()["title"]
    assert db.get_paper(LARGEST_ID + 1) is None
    assert db.delete_paper(LARGEST_ID + 1) is None
    assert db.delete_paper(LARGEST_ID) == arxiv_paper()["pdf_path"]
    assert db.count_papers() == 0


def spy_on_connections(monkeypatch) -> list[tuple[sqlite3.Connection, dict]]:
    opened = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append((connection, kwargs))
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    return opened


def test_each_call_opens_one_short_lived_connection(monkeypatch):
    opened = spy_on_connections(monkeypatch)
    db.count_papers()
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0][0].execute("SELECT 1")


def test_a_locked_database_is_waited_for_ten_seconds(monkeypatch):
    opened = spy_on_connections(monkeypatch)
    db.count_papers()
    assert opened[0][1]["timeout"] == 10
