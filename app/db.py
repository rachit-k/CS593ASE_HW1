"""SQLite storage for papers and their Q&A history.

Every function opens its own short-lived connection. That is plenty for a
single-user local app and keeps each function independent and easy to test.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL CHECK (source IN ('arxiv', 'upload')),
    arxiv_id    TEXT    UNIQUE,      -- versionless, e.g. 2401.12345; NULL for uploads
    title       TEXT    NOT NULL,
    authors     TEXT    NOT NULL,    -- JSON array of names (may be empty)
    year        INTEGER,             -- NULL when unknown
    abstract    TEXT,
    url         TEXT,                -- arXiv abstract page; NULL for uploads
    pdf_path    TEXT    NOT NULL,    -- file name inside DATA_DIR/pdfs/
    full_text   TEXT    NOT NULL,    -- invariant: every library paper has text
    summary     TEXT,                -- cached Markdown; NULL until generated
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS questions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_questions_paper_id ON questions(paper_id);
"""

# Columns for list views: everything except the (large) full text and summary.
_LIST_COLUMNS = """
    id, source, arxiv_id, title, authors, year, abstract, url, pdf_path, created_at,
    summary IS NOT NULL AS has_summary,
    (SELECT COUNT(*) FROM questions q WHERE q.paper_id = papers.id) AS question_count
"""

# SQLite integers are 64-bit. A larger id can't exist, and looking one up raises OverflowError.
_MAX_ID = 2**63 - 1


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a connection; commit if the block succeeds, roll back if it raises."""
    conn = sqlite3.connect(config.settings.db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the data folders and the schema if they don't exist yet."""
    config.settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def _paper(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    paper = dict(row)
    paper["authors"] = json.loads(paper["authors"] or "[]")
    return paper


def add_paper(
    *,
    source: str,
    title: str,
    authors: list[str],
    year: int | None,
    abstract: str | None,
    url: str | None,
    pdf_path: str,
    full_text: str,
    arxiv_id: str | None = None,
) -> int:
    """Insert a paper and return its id. Raises sqlite3.IntegrityError for a duplicate arXiv ID."""
    if not full_text.strip():
        raise ValueError("Every library paper must have full text.")
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO papers (source, arxiv_id, title, authors, year, abstract, url, pdf_path, full_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source, arxiv_id, title, json.dumps(authors), year, abstract, url, pdf_path, full_text),
        )
        return int(cursor.lastrowid)


def get_paper(paper_id: int) -> dict[str, Any] | None:
    """Return one paper with every column (including full text), or None."""
    if not 0 < paper_id <= _MAX_ID:
        return None
    with connect() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    return _paper(row)


def list_papers() -> list[dict[str, Any]]:
    """Return all papers, newest first, without full text or summary."""
    with connect() as conn:
        rows = conn.execute(f"SELECT {_LIST_COLUMNS} FROM papers ORDER BY id DESC").fetchall()
    return [_paper(row) for row in rows]


def count_papers() -> int:
    with connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])


def find_by_arxiv_id(arxiv_id: str) -> int | None:
    """Return the id of the library paper with this arXiv ID, or None."""
    with connect() as conn:
        row = conn.execute("SELECT id FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return int(row["id"]) if row else None


def library_ids_for(arxiv_ids: list[str]) -> dict[str, int]:
    """Map each of the given arXiv IDs that is already in the library to its paper id."""
    if not arxiv_ids:
        return {}
    placeholders = ",".join("?" * len(arxiv_ids))
    with connect() as conn:
        rows = conn.execute(
            f"SELECT arxiv_id, id FROM papers WHERE arxiv_id IN ({placeholders})", arxiv_ids
        ).fetchall()
    return {row["arxiv_id"]: int(row["id"]) for row in rows}


def delete_paper(paper_id: int) -> str | None:
    """Delete a paper and its questions. Return its PDF file name, or None if it didn't exist."""
    if not 0 < paper_id <= _MAX_ID:
        return None
    with connect() as conn:
        row = conn.execute("SELECT pdf_path FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    return str(row["pdf_path"])


def set_summary(paper_id: int, summary: str) -> bool:
    """Store a paper's summary. Return False if the paper no longer exists."""
    with connect() as conn:
        return conn.execute("UPDATE papers SET summary = ? WHERE id = ?", (summary, paper_id)).rowcount > 0


def add_question(paper_id: int, question: str, answer: str) -> dict[str, Any]:
    """Store a question and its answer; return the stored row."""
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO questions (paper_id, question, answer) VALUES (?, ?, ?)",
            (paper_id, question, answer),
        )
        row = conn.execute("SELECT * FROM questions WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def list_questions(paper_id: int) -> list[dict[str, Any]]:
    """Return a paper's questions and answers, oldest first."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM questions WHERE paper_id = ? ORDER BY id", (paper_id,)
        ).fetchall()
    return [dict(row) for row in rows]
