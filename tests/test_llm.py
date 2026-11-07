"""llm.py: what is sent to Gemini and how its replies and errors are handled (design section 6).

The `gemini` fixture is a fake REST endpoint behind the real google-genai SDK, so these tests
check the actual request bodies and the SDK's real error types.
"""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from google.genai import errors

from app import config, llm
from app.llm import LLMError
from helpers import BODY_FACT, PAPER_TEXT, PAPER_TITLE

PAPER = {"title": PAPER_TITLE, "full_text": PAPER_TEXT}
METADATA = {"title": "A Paper", "authors": ["Alice Smith"], "year": 2024, "abstract": "An abstract."}


def call_summarize():
    return llm.summarize(PAPER)


def call_answer():
    return llm.answer(PAPER, "What?", [])


def call_extract_metadata():
    return llm.extract_metadata("First pages")


ALL_THREE = [call_summarize, call_answer, call_extract_metadata]
ALL_THREE_IDS = ["summarize", "answer", "extract_metadata"]


def qa_history(count: int) -> list[dict[str, str]]:
    return [{"question": f"question-{n}?", "answer": f"answer-{n}."} for n in range(1, count + 1)]


# --- Summaries ----------------------------------------------------------------------


def test_summary_is_built_from_the_whole_paper(gemini):
    gemini.reply_text("**TL;DR:** Agents help.")

    result = llm.summarize(PAPER)

    assert result.text == "**TL;DR:** Agents help."
    assert result.truncated is False
    prompt = gemini.prompt()
    assert PAPER_TITLE in prompt
    assert BODY_FACT in prompt
    assert gemini.requests[0].url.path.endswith("/models/test-model:generateContent")


def test_summary_asks_for_the_fixed_sections_and_only_the_papers_facts(gemini):
    gemini.reply_text("Summary")
    llm.summarize(PAPER)
    for section in ("TL;DR", "### Problem", "### Approach", "### Key results", "### Limitations"):
        assert section in gemini.prompt()
    assert "Work only from the paper text" in gemini.system()


def test_replies_are_trimmed(gemini):
    gemini.reply_text("  \n A summary. \n ")
    assert llm.summarize(PAPER).text == "A summary."


def test_a_long_paper_is_cut_at_400000_characters_and_says_so(gemini):
    gemini.reply_text("Summary")
    result = llm.summarize({"title": "Thesis", "full_text": "a" * 400_000 + "TAIL-MARKER"})

    assert result.truncated is True
    assert "a" * 400_000 in gemini.prompt()
    assert "TAIL-MARKER" not in gemini.prompt()
    assert "Only the first part" in gemini.prompt()


def test_a_paper_of_exactly_the_limit_is_sent_whole(gemini):
    gemini.reply_text("Summary")
    result = llm.summarize({"title": "Thesis", "full_text": "a" * 400_000})

    assert result.truncated is False
    assert "Only the first part" not in gemini.prompt()


# --- Questions ----------------------------------------------------------------------


def test_a_question_is_asked_about_the_whole_paper(gemini):
    gemini.reply_text("The Zephyr benchmark.")

    result = llm.answer(PAPER, "What datasets are used?", [])

    assert result.text == "The Zephyr benchmark."
    prompt = gemini.prompt()
    assert BODY_FACT in prompt
    assert prompt.endswith("Question: What datasets are used?")
    assert "<history>" not in prompt


def test_instructions_keep_answers_to_the_paper(gemini):
    gemini.reply_text("Answer")
    llm.answer(PAPER, "What?", [])
    system = gemini.system()
    assert "using only the paper text" in system
    assert "say so plainly" in system
    assert "section" in system


def test_earlier_questions_are_sent_as_context_in_order(gemini):
    gemini.reply_text("Answer")
    llm.answer(PAPER, "A follow-up?", qa_history(2))

    prompt = gemini.prompt()
    assert "Q: question-1?\nA: answer-1." in prompt
    assert prompt.index("question-1?") < prompt.index("question-2?") < prompt.index("A follow-up?")


def test_only_the_last_six_pairs_are_sent(gemini):
    gemini.reply_text("Answer")
    llm.answer(PAPER, "A follow-up?", qa_history(7))

    prompt = gemini.prompt()
    assert "question-1?" not in prompt
    for n in range(2, 8):
        assert f"question-{n}?" in prompt


# --- Upload metadata ----------------------------------------------------------------


def test_metadata_is_read_from_the_first_pages_and_requested_as_json(gemini):
    gemini.reply_text(json.dumps(METADATA))

    meta = llm.extract_metadata("The first pages of the paper.")

    assert meta.title == "A Paper"
    assert meta.authors == ["Alice Smith"]
    assert meta.year == 2024
    assert meta.abstract == "An abstract."
    assert "The first pages of the paper." in gemini.prompt()
    assert "Never invent values" in gemini.system()
    generation = gemini.generation_config()
    assert generation["responseMimeType"] == "application/json"
    schema = generation.get("responseSchema") or generation["responseJsonSchema"]
    assert set(schema["properties"]) == {"title", "authors", "year", "abstract"}


def test_metadata_whitespace_is_cleaned_up(gemini):
    gemini.reply_text(
        json.dumps({"title": "  A   Spaced\nTitle ", "authors": [" Alice   Smith ", "", "   "], "abstract": " Some\n text. "})
    )
    meta = llm.extract_metadata("First pages")
    assert meta.title == "A Spaced Title"
    assert meta.authors == ["Alice Smith"]
    assert meta.abstract == "Some text."
    assert meta.year is None


def test_a_blank_abstract_becomes_none(gemini):
    gemini.reply_text(json.dumps(METADATA | {"abstract": "   "}))
    assert llm.extract_metadata("First pages").abstract is None


@pytest.mark.parametrize(
    ("year", "expected"),
    [(1899, None), (1900, 1900), (2024, 2024), (2100, 2100), (2101, None), (None, None)],
)
def test_implausible_years_are_dropped(gemini, year, expected):
    gemini.reply_text(json.dumps(METADATA | {"year": year}))
    assert llm.extract_metadata("First pages").year == expected


def test_a_blank_title_is_an_error(gemini):
    gemini.reply_text(json.dumps(METADATA | {"title": "   "}))
    with pytest.raises(LLMError, match="couldn't find the paper's title"):
        llm.extract_metadata("First pages")


@pytest.mark.parametrize(
    "reply",
    ["this is not json", json.dumps({"authors": ["No title"]}), json.dumps({"title": "T", "authors": "not a list"})],
    ids=["not-json", "missing-title", "wrong-type"],
)
def test_metadata_in_the_wrong_format_is_an_error(gemini, reply):
    gemini.reply_text(reply)
    with pytest.raises(LLMError, match="expected format") as caught:
        llm.extract_metadata("First pages")
    assert caught.value.status_code == 502


# --- The thinking setting -----------------------------------------------------------


@pytest.mark.parametrize("call", ALL_THREE, ids=ALL_THREE_IDS)
def test_gemini_3_models_get_a_low_thinking_level(gemini, use_model, call):
    use_model("gemini-3.8-flash")
    gemini.reply_text(json.dumps(METADATA))
    call()
    # The SDK decides how the key is spelled (it currently sends thinking_level); the API takes both.
    thinking = gemini.generation_config()["thinkingConfig"]
    assert "LOW" in (thinking.get("thinking_level"), thinking.get("thinkingLevel"))
    assert "/models/gemini-3.8-flash:" in gemini.requests[0].url.path


@pytest.mark.parametrize("call", ALL_THREE, ids=ALL_THREE_IDS)
def test_older_models_get_no_thinking_setting(gemini, use_model, call):
    use_model("gemini-2.5-flash")
    gemini.reply_text(json.dumps(METADATA))
    call()
    assert "thinkingConfig" not in gemini.generation_config()


def test_a_model_that_rejects_the_thinking_setting_is_retried_without_it(gemini, use_model):
    use_model("gemini-3.0-preview")
    gemini.reply_error(400, "Thinking level is not supported for this model.", "INVALID_ARGUMENT")
    gemini.reply_text("Summary")

    assert llm.summarize(PAPER).text == "Summary"

    assert len(gemini.requests) == 2
    assert "thinkingConfig" in gemini.generation_config(0)
    assert "thinkingConfig" not in gemini.generation_config(1)


def test_the_retry_without_thinking_keeps_the_json_format(gemini, use_model):
    use_model("gemini-3.0-preview")
    gemini.reply_error(400, "Thinking level is not supported for this model.", "INVALID_ARGUMENT")
    gemini.reply_text(json.dumps(METADATA))

    assert llm.extract_metadata("First pages").title == "A Paper"
    assert gemini.generation_config(1)["responseMimeType"] == "application/json"


def test_other_bad_requests_are_not_retried(gemini, use_model):
    use_model("gemini-3.8-flash")
    gemini.reply_error(400, "The request was malformed.", "INVALID_ARGUMENT")

    with pytest.raises(LLMError, match=r"HTTP 400\): The request was malformed"):
        llm.summarize(PAPER)
    assert len(gemini.requests) == 1


# --- Errors -------------------------------------------------------------------------


def test_no_api_key_explains_how_to_set_one():
    with pytest.raises(LLMError, match="No Gemini API key") as caught:
        llm.summarize(PAPER)
    assert caught.value.status_code == 503
    assert "GEMINI_API_KEY" in str(caught.value)


def test_quota_errors_name_the_model_and_the_escape_hatch(gemini, use_model):
    use_model("gemini-2.5-flash")
    gemini.reply_error(429, "Resource has been exhausted.", "RESOURCE_EXHAUSTED")

    with pytest.raises(LLMError) as caught:
        llm.summarize(PAPER)

    assert caught.value.status_code == 429
    assert "gemini-2.5-flash" in str(caught.value)
    assert "GEMINI_MODEL" in str(caught.value)
    assert len(gemini.requests) == 1


@pytest.mark.parametrize("code", [400, 401, 403])
def test_a_rejected_key_says_to_check_the_key(gemini, code):
    gemini.reply_error(code, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
    with pytest.raises(LLMError, match="rejected the API key") as caught:
        llm.summarize(PAPER)
    assert caught.value.status_code == 502


def test_an_unknown_model_is_named(gemini):
    gemini.reply_error(404, "models/test-model is not found.", "NOT_FOUND")
    with pytest.raises(LLMError, match="doesn't recognize the model 'test-model'") as caught:
        llm.summarize(PAPER)
    assert caught.value.status_code == 502


@pytest.mark.parametrize("code", [500, 503])
def test_gemini_server_errors_suggest_trying_later(gemini, code):
    gemini.reply_error(code, "Internal error.", "INTERNAL")
    with pytest.raises(LLMError, match=f"having trouble right now \\(HTTP {code}\\)") as caught:
        llm.summarize(PAPER)
    assert caught.value.status_code == 503


def test_a_server_error_without_a_json_body_is_still_friendly(gemini):
    gemini.reply(502, text="<html>Bad Gateway</html>")
    with pytest.raises(LLMError, match="having trouble"):
        llm.summarize(PAPER)


def test_a_timeout_says_so(gemini):
    gemini.fail(httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError, match="took too long") as caught:
        llm.summarize(PAPER)
    assert caught.value.status_code == 504


def test_a_connection_failure_says_so(gemini):
    gemini.fail(httpx.ConnectError("no route"))
    with pytest.raises(LLMError, match="Couldn't reach Gemini") as caught:
        llm.summarize(PAPER)
    assert caught.value.status_code == 502


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "SAFETY"}]}, "(reason: SAFETY)"),
        ({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}, "(reason: PROHIBITED_CONTENT)"),
        ({"candidates": []}, "empty answer."),
    ],
    ids=["finish-reason", "blocked-prompt", "no-reason"],
)
def test_an_empty_answer_is_an_error_with_the_reason(gemini, body, reason):
    gemini.reply(200, json=body)
    with pytest.raises(LLMError, match="empty answer") as caught:
        llm.answer(PAPER, "What?", [])
    assert reason in str(caught.value)
    assert caught.value.status_code == 502


# --- The client ---------------------------------------------------------------------


def test_the_client_is_created_once_per_key(monkeypatch):
    monkeypatch.setattr(config, "settings", replace(config.settings, gemini_api_key="key-one"))
    first = llm.get_client()
    assert llm.get_client() is first

    monkeypatch.setattr(config, "settings", replace(config.settings, gemini_api_key="key-two"))
    assert llm.get_client() is not first


# --- Rare shapes of a reply or an error --------------------------------------------------


def test_a_quota_error_on_a_gemini_3_model_is_not_retried_without_thinking(gemini, use_model):
    use_model("gemini-3.8-flash")
    gemini.reply_error(429, "Thinking about quota: resource has been exhausted.", "RESOURCE_EXHAUSTED")

    with pytest.raises(LLMError, match="quota") as caught:
        llm.summarize(PAPER)

    assert caught.value.status_code == 429
    assert len(gemini.requests) == 1


def test_a_bad_request_mentioning_thinking_on_an_older_model_is_not_retried(gemini, use_model):
    use_model("gemini-2.5-flash")
    gemini.reply_error(400, "Thinking is not supported.", "INVALID_ARGUMENT")

    with pytest.raises(LLMError, match="Thinking is not supported"):
        llm.summarize(PAPER)

    assert len(gemini.requests) == 1


@pytest.mark.parametrize(
    ("error", "detail"),
    [({"code": 422, "status": "UNPROCESSABLE"}, "UNPROCESSABLE"), ({"code": 422}, "")],
    ids=["status-only", "nothing"],
)
def test_an_error_with_no_message_falls_back_to_what_it_has(gemini, error, detail):
    gemini.reply(422, json={"error": error})
    with pytest.raises(LLMError) as caught:
        llm.summarize(PAPER)
    assert str(caught.value) == f"Gemini returned an error (HTTP 422): {detail}"
    assert caught.value.status_code == 502


def test_an_error_with_no_code_is_reported_as_code_0():
    friendly = llm._friendly_error(errors.APIError(0, {"error": {"message": "Something odd."}}), "test-model")
    assert str(friendly) == "Gemini returned an error (HTTP 0): Something odd."
    assert friendly.status_code == 502


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            {
                "promptFeedback": {"safetyRatings": []},
                "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "MAX_TOKENS"}],
            },
            "(reason: MAX_TOKENS)",
        ),
        ({"candidates": [{"content": {"role": "model", "parts": []}}]}, "empty answer."),
    ],
    ids=["feedback-without-a-block", "candidate-without-a-reason"],
)
def test_an_empty_answer_reports_the_finish_reason_if_there_is_one(gemini, body, reason):
    gemini.reply(200, json=body)
    with pytest.raises(LLMError) as caught:
        llm.answer(PAPER, "What?", [])
    assert reason in str(caught.value)


# --- Found by mutation testing: the exact text sent to Gemini and shown to people ------------------

SHORT_PAPER = {"title": "A Title", "full_text": "The body."}
PAPER_BLOCK = "Title: A Title\n\n<paper>\nThe body.\n</paper>"


def test_the_summary_prompt_is_the_paper_then_the_instructions(gemini):
    gemini.reply_text("A summary.")
    llm.summarize(SHORT_PAPER)
    assert gemini.prompt() == f"{PAPER_BLOCK}\n\n{llm.SUMMARY_INSTRUCTIONS}"


def test_the_question_prompt_is_the_paper_then_the_question(gemini):
    gemini.reply_text("An answer.")
    llm.answer(SHORT_PAPER, "Why?", [])
    assert gemini.prompt() == f"{PAPER_BLOCK}\n\nQuestion: Why?"


def test_a_follow_up_prompt_puts_the_history_between_the_paper_and_the_question(gemini):
    gemini.reply_text("An answer.")
    llm.answer(SHORT_PAPER, "Why?", [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}])
    assert gemini.prompt() == (
        f"{PAPER_BLOCK}\n\n"
        "Earlier questions and answers about this paper:\n\n"
        "<history>\nQ: Q1\nA: A1\n\nQ: Q2\nA: A2\n</history>\n\n"
        "Question: Why?"
    )


def test_the_metadata_prompt_is_only_the_first_pages(gemini):
    gemini.reply_text(json.dumps(METADATA))
    llm.extract_metadata("The first pages.")
    assert gemini.prompt() == "First pages of the paper:\n\n<paper>\nThe first pages.\n</paper>"


def test_the_quota_message_is_complete(gemini, use_model):
    use_model("gemini-3.8-flash")
    gemini.reply_error(429, "Quota exceeded.", "RESOURCE_EXHAUSTED")
    with pytest.raises(LLMError) as caught:
        llm.answer(SHORT_PAPER, "Why?", [])
    assert str(caught.value) == (
        "Gemini's free-tier quota for gemini-3.8-flash is used up for now. Wait a minute and try again. "
        "If it keeps happening, today's quota may be spent: set GEMINI_MODEL in .env to another model "
        "(for example a Flash-Lite model) and restart the app."
    )


def test_a_long_error_detail_is_cut_at_300_characters(gemini):
    gemini.reply_error(422, "x" * 500)
    with pytest.raises(LLMError) as caught:
        llm.answer(SHORT_PAPER, "Why?", [])
    assert str(caught.value) == "Gemini returned an error (HTTP 422): " + "x" * 300


def test_metadata_with_a_blank_title_is_an_error(gemini):
    gemini.reply_text(json.dumps(METADATA | {"title": "   "}))
    with pytest.raises(LLMError, match="couldn't find the paper's title") as caught:
        llm.extract_metadata("The first pages.")
    assert caught.value.status_code == 502
