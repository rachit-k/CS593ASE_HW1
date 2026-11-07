"""Gemini calls: upload metadata, paper summaries, and questions about a paper.

Everything goes through Google's official `google-genai` SDK. Each function
sends the paper's text (whole, no retrieval step) and returns plain text or
validated data. Any failure becomes an LLMError whose message can be shown to
the user as is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from app import config

MAX_PAPER_CHARS = 400_000  # about 100k tokens: one request stays well under free-tier per-minute limits
HISTORY_TURNS = 6  # earlier Q&A pairs sent along with a new question
REQUEST_TIMEOUT_MS = 120_000


class LLMError(Exception):
    """Talking to Gemini failed. The message is written for the user."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class PaperMetadata(BaseModel):
    title: str
    authors: list[str]
    year: int | None = None
    abstract: str | None = None


@dataclass
class LLMResult:
    text: str  # Markdown
    truncated: bool  # True if the paper was cut to MAX_PAPER_CHARS before sending


# --- Client and request plumbing -------------------------------------------------

_clients: dict[str, genai.Client] = {}


def get_client() -> genai.Client:
    """Return a Gemini client for the configured key. Tests can monkeypatch this."""
    key = config.settings.gemini_api_key
    if not key:
        raise LLMError(
            "No Gemini API key is set. Add GEMINI_API_KEY to the .env file "
            "(see the README), then restart the app.",
            503,
        )
    if key not in _clients:
        _clients[key] = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        )
    return _clients[key]


def thinking_config(model: str) -> types.ThinkingConfig | None:
    """Gemini 3 models take a thinking *level*; older models reject it, so they get their default."""
    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    return None


def _generate(contents: str, *, system: str, schema: type[BaseModel] | None = None) -> Any:
    model = config.settings.gemini_model
    client = get_client()
    options: dict[str, Any] = {"system_instruction": system}
    if schema is not None:
        options.update(response_mime_type="application/json", response_schema=schema)
    thinking = thinking_config(model)
    try:
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**options, thinking_config=thinking),
            )
        except errors.ClientError as exc:
            # A model that doesn't support the thinking setting: retry once without it.
            if thinking is not None and exc.code == 400 and "thinking" in str(exc.message).lower():
                return client.models.generate_content(
                    model=model, contents=contents, config=types.GenerateContentConfig(**options)
                )
            raise
    except errors.APIError as exc:
        raise _friendly_error(exc, model) from exc
    except httpx.TimeoutException as exc:
        raise LLMError("Gemini took too long to respond. Try again.", 504) from exc
    except httpx.HTTPError as exc:
        raise LLMError(
            "Couldn't reach Gemini. Check your internet connection and try again.", 502
        ) from exc
    except Exception as exc:  # the SDK raises several types for a reply it can't parse
        raise LLMError("Gemini sent a reply the app couldn't read. Try again.", 502) from exc


def _friendly_error(exc: errors.APIError, model: str) -> LLMError:
    code = exc.code or 0
    detail = " ".join(str(exc.message or exc.status or "").split())[:300]
    if code == 429:
        return LLMError(
            f"Gemini's free-tier quota for {model} is used up for now. Wait a minute and try "
            "again. If it keeps happening, today's quota may be spent: set GEMINI_MODEL in .env "
            "to another model (for example a Flash-Lite model) and restart the app.",
            429,
        )
    if code in (400, 401, 403) and "api key" in detail.lower():
        return LLMError("Gemini rejected the API key. Check GEMINI_API_KEY in .env, then restart.", 502)
    if code == 404:
        return LLMError(f"Gemini doesn't recognize the model '{model}'. Check GEMINI_MODEL in .env.", 502)
    if code >= 500:
        return LLMError(f"Gemini is having trouble right now (HTTP {code}). Try again in a minute.", 503)
    return LLMError(f"Gemini returned an error (HTTP {code}): {detail}", 502)


def _response_text(response: Any) -> str:
    text = (response.text or "").strip()
    if text:
        return text
    reason = None
    if getattr(response, "prompt_feedback", None) and response.prompt_feedback.block_reason:
        reason = response.prompt_feedback.block_reason
    elif response.candidates:
        reason = response.candidates[0].finish_reason
    reason_text = f" (reason: {getattr(reason, 'value', reason)})" if reason else ""
    raise LLMError(f"Gemini returned an empty answer{reason_text}. Try again or rephrase.", 502)


def _paper_block(paper: dict[str, Any]) -> tuple[str, bool]:
    text = paper["full_text"]
    truncated = len(text) > MAX_PAPER_CHARS
    note = "\n(Only the first part of this long document is included.)" if truncated else ""
    return f"Title: {paper['title']}{note}\n\n<paper>\n{text[:MAX_PAPER_CHARS]}\n</paper>", truncated


# --- The three operations --------------------------------------------------------

METADATA_SYSTEM = (
    "You extract bibliographic metadata from the first pages of an academic paper. "
    "Copy the title and the author names exactly as they appear (names only, without "
    "affiliations, emails, or footnote marks). Give the publication year only if the text "
    "states it, for example in a venue line, copyright notice, or arXiv stamp; otherwise null. "
    "Copy the abstract if the paper has one; otherwise null. Never invent values."
)


def extract_metadata(first_pages_text: str) -> PaperMetadata:
    """Read title, authors, year, and abstract from the first pages of a paper."""
    response = _generate(
        f"First pages of the paper:\n\n<paper>\n{first_pages_text}\n</paper>",
        system=METADATA_SYSTEM,
        schema=PaperMetadata,
    )
    meta = response.parsed if isinstance(response.parsed, PaperMetadata) else None
    if meta is None:
        try:
            meta = PaperMetadata.model_validate_json(_response_text(response))
        except ValidationError as exc:
            raise LLMError("Gemini's metadata wasn't in the expected format.", 502) from exc

    meta.title = " ".join(meta.title.split())
    meta.authors = [" ".join(name.split()) for name in meta.authors if name.strip()]
    meta.abstract = " ".join((meta.abstract or "").split()) or None
    if meta.year is not None and not 1900 <= meta.year <= 2100:
        meta.year = None
    if not meta.title:
        raise LLMError("Gemini couldn't find the paper's title.", 502)
    return meta


SUMMARY_SYSTEM = (
    "You are a research assistant who explains academic papers clearly and accurately. "
    "Work only from the paper text you are given. Do not add facts, numbers, or claims "
    "that are not in it."
)

SUMMARY_INSTRUCTIONS = """Summarize this paper in Markdown, using exactly this structure:

**TL;DR:** one or two sentences on what the paper does and why it matters.

### Problem
What problem the paper addresses and why it matters.

### Approach
The main idea of the method, in plain language.

### Key results
The most important findings, with key numbers and the datasets or benchmarks they come from.

### Limitations
The limitations the authors acknowledge. If they state none, say so, then note at most two \
limitations that are evident from the paper itself, labeled as your own observation.

Use short paragraphs or bullet points. Keep the whole summary under about 400 words."""


def summarize(paper: dict[str, Any]) -> LLMResult:
    """Write a structured Markdown summary from the paper's full text."""
    block, truncated = _paper_block(paper)
    response = _generate(f"{block}\n\n{SUMMARY_INSTRUCTIONS}", system=SUMMARY_SYSTEM)
    return LLMResult(_response_text(response), truncated)


QA_SYSTEM = (
    "You answer questions about one academic paper, using only the paper text you are given. "
    "If the paper does not contain the answer, say so plainly instead of guessing. "
    "When you can, mention the section, table, or figure the answer comes from. "
    "Answer in Markdown. Be concise: a short paragraph or a few bullet points, unless the "
    "question asks for detail."
)


def answer(paper: dict[str, Any], question: str, history: list[dict[str, Any]]) -> LLMResult:
    """Answer a question about the paper, with recent Q&A pairs as conversational context."""
    block, truncated = _paper_block(paper)
    parts = [block]
    recent = history[-HISTORY_TURNS:]
    if recent:
        earlier = "\n\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in recent)
        parts.append(f"Earlier questions and answers about this paper:\n\n<history>\n{earlier}\n</history>")
    parts.append(f"Question: {question}")
    response = _generate("\n\n".join(parts), system=QA_SYSTEM)
    return LLMResult(_response_text(response), truncated)
