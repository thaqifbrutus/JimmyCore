"""
Tests for app/services/ai_service.py — the daily-quota short-circuit,
history trimming in answer_dataset_question, and the language instruction
in SYSTEM_PROMPT.

Network is never touched: the model call is mocked at whichever boundary
is appropriate for each test.
"""
import pandas as pd
import pytest

from app.services import ai_service


# ---------------------------------------------------------------------------
# Daily-quota short-circuit
# ---------------------------------------------------------------------------

def test_daily_quota_error_aborts_model_chain(monkeypatch):
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        raise RuntimeError(
            "Error code: 429 - Too Many Requests. "
            "limit_source: openrouter_free_tier_daily; "
            "reason: free-models-per-day exceeded"
        )

    monkeypatch.setattr(ai_service.client.chat.completions, "create", fake_create)
    # Force a non-trivial chain so we can prove we don't walk it.
    monkeypatch.setattr(ai_service, "AI_FALLBACK_MODELS", ["model_b", "model_c"])

    result = ai_service._call_ai_model(
        messages=[{"role": "user", "content": "hi"}],
        context_label="test",
        max_tokens=100,
    )

    # Only the primary model was tried — the chain was aborted.
    assert call_count["n"] == 1
    assert result["status"] == "error"
    assert result["error_type"] == "daily_quota_exhausted"


def test_non_quota_error_walks_the_full_chain(monkeypatch):
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        raise RuntimeError("network error")

    monkeypatch.setattr(ai_service.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(ai_service, "AI_FALLBACK_MODELS", ["model_b", "model_c"])

    result = ai_service._call_ai_model(
        messages=[{"role": "user", "content": "hi"}],
        context_label="test",
        max_tokens=100,
    )

    # primary + 2 fallbacks = 3 attempts — the pre-existing behavior.
    assert call_count["n"] == 3
    assert result["status"] == "error"
    # Not a quota error, so no special error_type.
    assert result["error_type"] is None


def test_quota_error_detection_helper(monkeypatch):
    """
    Direct unit test of the pattern-match helper, since it's load-bearing
    for the short-circuit above.
    """
    assert ai_service._is_daily_quota_error(
        RuntimeError("429 free-models-per-day exceeded")
    )
    assert ai_service._is_daily_quota_error(
        RuntimeError("HTTP 429: openrouter_free_tier_daily")
    )
    # 429 without the free-tier markers — not a daily-quota error.
    assert not ai_service._is_daily_quota_error(RuntimeError("429 too many requests"))
    # Free-tier markers without a 429 — not a daily-quota error.
    assert not ai_service._is_daily_quota_error(RuntimeError("free-models-per-day"))
    assert not ai_service._is_daily_quota_error(RuntimeError("network error"))


# ---------------------------------------------------------------------------
# History trimming (Task 3)
# ---------------------------------------------------------------------------

def test_answer_dataset_question_trims_history(monkeypatch):
    """
    answer_dataset_question's job is to build the message list and hand
    it to tool_runner.run_tool_loop. We mock that boundary and inspect
    what it received — the tool loop itself has its own tests.

    With 20 turns of history and MAX_HISTORY_TURNS=12, the model should
    only see the last 12 turns (i.e. messages from turn 8 onward).
    """
    from app.services import tool_runner

    captured = {}

    def fake_run_tool_loop(df, messages, system_prompt, context_label, max_tokens=4000):
        captured["messages"] = messages
        return {
            "status": "ok",
            "content": "answer",
            "reason": None,
            "tool_calls_log": [],
        }

    monkeypatch.setattr(tool_runner, "run_tool_loop", fake_run_tool_loop)

    # 20 turns = 40 messages, alternating user/assistant.
    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"Q{i}"})
        history.append({"role": "assistant", "content": f"A{i}"})

    ai_service.answer_dataset_question(
        df=pd.DataFrame({"x": [1, 2, 3]}),
        profile_data={"overview": {}},
        original_filename="test.csv",
        question="final?",
        conversation_history=history,
    )

    msgs = captured["messages"]
    # [context_user, context_assistant] + last 24 history entries + final question.
    assert len(msgs) == 2 + (ai_service.MAX_HISTORY_TURNS * 2) + 1

    # The trimmed history should start at turn 8 (20 - 12) and end at turn 19.
    history_slice = msgs[2:-1]
    assert history_slice[0]["content"] == "Q8"
    assert history_slice[-1]["content"] == "A19"
    # And the final question is still last.
    assert msgs[-1]["content"] == "final?"


def test_answer_dataset_question_handles_short_history(monkeypatch):
    """Under the cap, nothing gets dropped."""
    from app.services import tool_runner

    captured = {}

    def fake_run_tool_loop(df, messages, system_prompt, context_label, max_tokens=4000):
        captured["messages"] = messages
        return {"status": "ok", "content": "x", "reason": None, "tool_calls_log": []}

    monkeypatch.setattr(tool_runner, "run_tool_loop", fake_run_tool_loop)

    history = [
        {"role": "user", "content": "Q0"},
        {"role": "assistant", "content": "A0"},
    ]

    ai_service.answer_dataset_question(
        df=pd.DataFrame({"x": [1]}),
        profile_data={},
        original_filename="test.csv",
        question="next?",
        conversation_history=history,
    )

    msgs = captured["messages"]
    # 2 context + 2 history + 1 question = 5
    assert len(msgs) == 5
    assert msgs[2]["content"] == "Q0"
    assert msgs[3]["content"] == "A0"


# ---------------------------------------------------------------------------
# Language instruction (Task 4)
# ---------------------------------------------------------------------------

def test_system_prompt_instructs_same_language():
    assert "respond in the same language" in ai_service.SYSTEM_PROMPT