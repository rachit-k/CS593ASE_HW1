"""Model-based testing: random sequences of user actions, checked against a simple model of the library.

Hypothesis picks the actions and their order. After every action, the real app must agree with the
model: the same papers in the same order, the right count in the header, no stray or missing PDF
files, and no questions left over from a deleted paper. When a sequence fails, Hypothesis shrinks it
to the shortest sequence that still fails.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from google import genai
from google.genai import types
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, consumes, invariant, multiple, rule

from app import arxiv_client, config, db, llm, pdfs
from app.main import app
from helpers import HTMX, FakeGemini, FakeService, make_pdf, parse_html

ARXIV_IDS = ["2401.00001", "2401.00002", "hep-th/9901001"]
ID_FORMS = ["{id}", "{id}v2", "arXiv:{id}", "https://arxiv.org/abs/{id}/"]
PDF = make_pdf()


@dataclass
class ModelPaper:
    source: str
    title: str
    arxiv_id: str | None = None
    summary: str | None = None
    questions: list[str] = field(default_factory=list)


class LibraryMachine(RuleBasedStateMachine):
    papers = Bundle("papers")

    def __init__(self) -> None:
        super().__init__()
        self.folder = Path(tempfile.mkdtemp(prefix="stateful-"))
        config.settings = config.Settings("test-key", "test-model", self.folder / "data")
        db.init_db()
        self.model: dict[int, ModelPaper] = {}
        self.gemini_up = True
        self.downloads_up = True
        self.replies = 0

        arxiv_pdfs = FakeService().reply_with(self._pdf_reply)
        pdfs.TRANSPORT = httpx.MockTransport(arxiv_pdfs)
        fake = FakeGemini().reply_with(self._gemini_reply)
        sdk = genai.Client(api_key="test-key", http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(fake))))
        self.real_get_client = llm.get_client
        llm.get_client = lambda: sdk
        self.client = TestClient(app)

    def teardown(self) -> None:
        llm.get_client = self.real_get_client
        self.client.close()
        shutil.rmtree(self.folder, ignore_errors=True)

    def _pdf_reply(self, _request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF) if self.downloads_up else httpx.Response(503)

    def _gemini_reply(self, _request: httpx.Request) -> httpx.Response:
        if not self.gemini_up:
            return httpx.Response(500, json={"error": {"code": 500, "message": "Internal error.", "status": "INTERNAL"}})
        self.replies += 1
        candidate = {"content": {"role": "model", "parts": [{"text": f"Reply {self.replies}"}]}, "finishReason": "STOP"}
        return httpx.Response(200, json={"candidates": [candidate]})

    # --- What a user can do ----------------------------------------------------------------

    @rule(target=papers, arxiv_id=st.sampled_from(ARXIV_IDS), form=st.sampled_from(ID_FORMS))
    def save(self, arxiv_id, form):
        known = next((paper_id for paper_id, paper in self.model.items() if paper.arxiv_id == arxiv_id), None)
        title = f"Saved {arxiv_id}"
        response = self.client.post("/library", data={"arxiv_id": form.format(id=arxiv_id), "title": title, "authors": "[]"}, headers=HTMX)
        if known is not None:
            assert response.status_code == 200
            return multiple()
        if not self.downloads_up:
            assert response.status_code == 502
            return multiple()
        assert response.status_code == 200
        paper_id = db.find_by_arxiv_id(arxiv_id)
        self.model[paper_id] = ModelPaper("arxiv", title, arxiv_id)
        return paper_id

    @rule(target=papers)
    def upload(self):
        self.gemini_up, was_up = False, self.gemini_up  # keep metadata simple: always use the fallbacks
        response = self.client.post("/library/upload", files={"file": ("p.pdf", PDF, "application/pdf")}, headers=HTMX)
        self.gemini_up = was_up
        assert response.status_code == 200
        paper_id = int(response.headers["HX-Redirect"].split("/")[2].split("?")[0])
        self.model[paper_id] = ModelPaper("upload", db.get_paper(paper_id)["title"])
        return paper_id

    @rule(paper_id=consumes(papers))
    def delete(self, paper_id):
        response = self.client.delete(f"/papers/{paper_id}", headers=HTMX)
        if paper_id in self.model:
            assert response.status_code == 200
            del self.model[paper_id]
        else:
            assert response.status_code == 404

    @rule(paper_id=papers)
    def delete_again_or_not(self, paper_id):
        if paper_id not in self.model:
            assert self.client.delete(f"/papers/{paper_id}", headers=HTMX).status_code == 404

    @rule(paper_id=papers)
    def summarize(self, paper_id):
        response = self.client.post(f"/papers/{paper_id}/summary", headers=HTMX)
        if paper_id not in self.model:
            assert response.status_code == 404
        elif self.gemini_up:
            assert response.status_code == 200
            self.model[paper_id].summary = f"Reply {self.replies}"
        else:
            assert response.status_code == 503

    @rule(paper_id=papers, question=st.sampled_from(["What is this?", "Which datasets?", "  Spaced   out?  "]))
    def ask(self, paper_id, question):
        response = self.client.post(f"/papers/{paper_id}/questions", data={"question": question}, headers=HTMX)
        if paper_id not in self.model:
            assert response.status_code == 404
        elif self.gemini_up:
            assert response.status_code == 200
            self.model[paper_id].questions.append(" ".join(question.split()))
        else:
            assert response.status_code == 503

    @rule()
    def restart_the_app(self):
        self.client.close()
        self.client = TestClient(app)
        with self.client:
            assert self.client.get("/library").status_code == 200

    @rule(up=st.booleans())
    def gemini_goes_up_or_down(self, up):
        self.gemini_up = up

    @rule(up=st.booleans())
    def arxiv_goes_up_or_down(self, up):
        self.downloads_up = up

    # --- What must always be true ----------------------------------------------------------

    @invariant()
    def the_library_page_matches_the_model(self):
        page = parse_html(self.client.get("/library").text)
        rows = page.find("ol", class_="entries").find_all("li", class_="entry")
        assert [row.find("h2").find("a").attrs["href"] for row in rows] == [f"/papers/{paper_id}" for paper_id in sorted(self.model, reverse=True)]
        assert page.find(id="library-count").text == str(len(self.model))
        for row, paper_id in zip(rows, sorted(self.model, reverse=True)):
            paper = self.model[paper_id]
            assert row.find("h2").text == paper.title
            facts = row.find(class_="facts").text
            assert facts.startswith("From arXiv" if paper.source == "arxiv" else "Uploaded PDF")
            assert ("Summarized" in facts) is (paper.summary is not None)
            count = len(paper.questions)
            assert (f"{count} question" in facts) is (count > 0)

    @invariant()
    def the_database_matches_the_model(self):
        assert db.count_papers() == len(self.model)
        for paper_id, paper in self.model.items():
            stored = db.get_paper(paper_id)
            assert stored["summary"] == paper.summary
            assert stored["full_text"].strip()
            assert stored["arxiv_id"] == paper.arxiv_id
            assert [item["question"] for item in db.list_questions(paper_id)] == paper.questions

    @invariant()
    def no_stray_or_missing_files_and_no_orphan_questions(self):
        expected = {db.get_paper(paper_id)["pdf_path"] for paper_id in self.model}
        assert {path.name for path in config.settings.pdf_dir.iterdir()} == expected
        with db.connect() as conn:
            orphans = conn.execute("SELECT COUNT(*) FROM questions WHERE paper_id NOT IN (SELECT id FROM papers)").fetchone()[0]
        assert orphans == 0

    @invariant()
    def saved_papers_are_recognized_by_every_id_form(self):
        for paper_id, paper in self.model.items():
            if paper.arxiv_id:
                assert db.find_by_arxiv_id(paper.arxiv_id) == paper_id
                assert arxiv_client.normalize_id(f"{paper.arxiv_id}v9") == paper.arxiv_id


LibraryMachine.TestCase.settings = settings(
    max_examples=max(5, settings.default.max_examples // 4),
    stateful_step_count=20,
)
TestLibrary = LibraryMachine.TestCase
