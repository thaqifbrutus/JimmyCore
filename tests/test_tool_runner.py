"""
Tests for app.services.tool_runner.

Only the OpenRouter call is mocked. Fake response objects mirror the
OpenAI SDK shape (choices[0].message.content / .tool_calls for the
non-streaming path; choices[0].delta.* for the streaming path).
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from app.services import tool_runner


@pytest.fixture()
def df():
    return pd.DataFrame({
        "state": ["Selangor", "Johor", "Selangor"],
        "accidents": [120, 85, 130],
    })


# ── Non-streaming helpers ──────────────────────────────────────────────────

def _make_tool_call(id, name, arguments):
    return SimpleNamespace(
        id=id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _make_response(content=None, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _patch_client(monkeypatch, responses):
    """responses: list of response objects or exceptions, consumed in order."""
    iterator = iter(responses)
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        nxt = next(iterator)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(
        tool_runner.client.chat.completions, "create", fake_create
    )
    return captured


# ── Non-streaming tests (unchanged) ────────────────────────────────────────

def test_no_tool_calls_returns_content_immediately(df, monkeypatch):
    _patch_client(monkeypatch, [_make_response(content="Hello!")])
    result = tool_runner.run_tool_loop(
        df=df,
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="You are helpful.",
        context_label="test",
    )
    assert result["status"] == "ok"
    assert result["content"] == "Hello!"
    assert result["tool_calls_log"] == []


def test_tool_call_appends_result_and_continues(df, monkeypatch):
    tool_call = _make_tool_call("call_1", "get_schema", {})
    responses = [
        _make_response(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
        _make_response(content="There are 3 rows."),
    ]
    captured = _patch_client(monkeypatch, responses)
    result = tool_runner.run_tool_loop(
        df=df,
        messages=[{"role": "user", "content": "how many rows?"}],
        system_prompt="You are helpful.",
        context_label="test",
    )
    assert result["status"] == "ok"
    assert result["content"] == "There are 3 rows."
    assert len(result["tool_calls_log"]) == 1
    assert result["tool_calls_log"][0]["name"] == "get_schema"
    second_call_messages = captured[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"


def test_iteration_cap_returns_failure(df, monkeypatch):
    tool_call = _make_tool_call("call_x", "get_schema", {})
    responses = [
        _make_response(content=None, tool_calls=[tool_call], finish_reason="tool_calls")
        for _ in range(tool_runner.MAX_TOOL_ITERATIONS)
    ]
    _patch_client(monkeypatch, responses)
    result = tool_runner.run_tool_loop(
        df=df,
        messages=[{"role": "user", "content": "go"}],
        system_prompt="You are helpful.",
        context_label="test",
    )
    assert result["status"] == "failed"
    assert "iteration limit" in result["reason"]


def test_unknown_tool_error_goes_back_to_model(df, monkeypatch):
    tool_call = _make_tool_call("call_1", "not_a_real_tool", {})
    responses = [
        _make_response(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
        _make_response(content="OK."),
    ]
    captured = _patch_client(monkeypatch, responses)
    result = tool_runner.run_tool_loop(
        df=df,
        messages=[{"role": "user", "content": "go"}],
        system_prompt="You are helpful.",
        context_label="test",
    )
    assert result["status"] == "ok"
    tool_msg = next(m for m in captured[1]["messages"] if m.get("role") == "tool")
    content = json.loads(tool_msg["content"])
    assert "unknown tool" in content["error"]


def test_malformed_tool_arguments_go_back_to_model(df, monkeypatch):
    bad_call = SimpleNamespace(
        id="call_bad",
        type="function",
        function=SimpleNamespace(name="get_schema", arguments="{not valid json"),
    )
    responses = [
        _make_response(content=None, tool_calls=[bad_call], finish_reason="tool_calls"),
        _make_response(content="Recovered."),
    ]
    captured = _patch_client(monkeypatch, responses)
    result = tool_runner.run_tool_loop(
        df=df,
        messages=[{"role": "user", "content": "go"}],
        system_prompt="You are helpful.",
        context_label="test",
    )
    assert result["status"] == "ok"
    tool_msg = next(m for m in captured[1]["messages"] if m.get("role") == "tool")
    content = json.loads(tool_msg["content"])
    assert "could not parse arguments" in content["error"]


def test_tools_unsupported_error_skips_to_next_model(df, monkeypatch):
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("This model does not support tools")
        return _make_response(content="Answer from fallback model.")

    monkeypatch.setattr(tool_runner.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(tool_runner, "AI_MODEL", "model_a")
    monkeypatch.setattr(tool_runner, "AI_FALLBACK_MODELS", ["model_b"])
    result = tool_runner.run_tool_loop(
        df=df,
        messages=[{"role": "user", "content": "go"}],
        system_prompt="You are helpful.",
        context_label="test",
    )
    assert result["status"] == "ok"
    assert result["content"] == "Answer from fallback model."
    assert call_count["n"] == 2


def test_all_models_reject_tools_falls_back_to_context_mode(df, monkeypatch):
    def fake_create(**kwargs):
        if "tools" in kwargs:
            raise RuntimeError("model does not support tools")
        return _make_response(content="Answer in context mode.")

    monkeypatch.setattr(tool_runner.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(tool_runner, "AI_MODEL", "model_a")
    monkeypatch.setattr(tool_runner, "AI_FALLBACK_MODELS", ["model_b"])
    result = tool_runner.run_tool_loop(
        df=df,
        messages=[{"role": "user", "content": "go"}],
        system_prompt="You are helpful.",
        context_label="test",
    )
    assert result["status"] == "ok"
    assert result["content"] == "Answer in context mode."
    assert result.get("context_mode") is True


# ── Streaming tests ────────────────────────────────────────────────────────

def _make_chunk(content=None, tool_calls=None, finish_reason=None):
    """One fake SSE chunk shaped like the OpenAI SDK stream chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _make_tool_call_delta(index, id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


def _patch_client_stream(monkeypatch, chunks, non_streaming_response=None):
    """
    Mock client.chat.completions.create so stream=True returns an iterator
    over chunks, and stream=False returns non_streaming_response (used by
    the wrap-up path). Returns the list of calls captured.
    """
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        if kwargs.get("stream"):
            return iter(chunks)
        if non_streaming_response is None:
            raise AssertionError(
                "non-streaming create called but no response was provided"
            )
        return non_streaming_response

    monkeypatch.setattr(
        tool_runner.client.chat.completions, "create", fake_create
    )
    return captured


def test_streaming_content_only_yields_tokens_then_done(df, monkeypatch):
    """A pure content stream yields each token, then a done event."""
    chunks = [
        _make_chunk(content="Hello "),
        _make_chunk(content="world"),
        _make_chunk(finish_reason="stop"),
    ]
    _patch_client_stream(monkeypatch, chunks)

    events = list(tool_runner.run_tool_loop_streaming(
        df=df,
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="You are helpful.",
        context_label="test",
    ))

    token_events = [e for e in events if e["type"] == "token"]
    done_events = [e for e in events if e["type"] == "done"]

    assert [e["content"] for e in token_events] == ["Hello ", "world"]
    assert len(done_events) == 1
    assert done_events[0]["content"] == "Hello world"
    assert done_events[0]["tool_calls_log"] == []


def test_streaming_tool_call_does_not_yield_tokens_then_loops(df, monkeypatch):
    """
    A tool-call stream accumulates silently, executes the tool, and the
    next iteration's content stream yields the answer.
    """
    # Iteration 1: model calls get_schema. No content.
    tc1 = _make_tool_call_delta(0, id="call_1", name="get_schema")
    # Iteration 2: model answers.
    chunks_iter1 = [
        _make_chunk(tool_calls=[tc1], finish_reason="tool_calls"),
    ]
    chunks_iter2 = [
        _make_chunk(content="There are 3 rows."),
        _make_chunk(finish_reason="stop"),
    ]

    # Because the streaming client is called twice (once per iteration),
    # we need to return different iterators per call.
    iterator_by_call = iter([iter(chunks_iter1), iter(chunks_iter2)])

    def fake_create(**kwargs):
        if not kwargs.get("stream"):
            raise AssertionError("did not expect a non-streaming call")
        return next(iterator_by_call)

    monkeypatch.setattr(
        tool_runner.client.chat.completions, "create", fake_create
    )

    events = list(tool_runner.run_tool_loop_streaming(
        df=df,
        messages=[{"role": "user", "content": "how many rows?"}],
        system_prompt="You are helpful.",
        context_label="test",
    ))

    token_events = [e for e in events if e["type"] == "token"]
    done_events = [e for e in events if e["type"] == "done"]

    # No tokens from the tool-call iteration; the answer's tokens come
    # from iteration 2.
    assert [e["content"] for e in token_events] == ["There are 3 rows."]
    assert len(done_events) == 1
    assert done_events[0]["content"] == "There are 3 rows."
    assert len(done_events[0]["tool_calls_log"]) == 1
    assert done_events[0]["tool_calls_log"][0]["name"] == "get_schema"


def test_streaming_mid_iteration_error_yields_error_event(df, monkeypatch):
    """A stream that raises after some tokens yields an error event."""
    class ExplodingStream:
        def __iter__(self):
            yield _make_chunk(content="Hello")
            raise RuntimeError("connection dropped")

    def fake_create(**kwargs):
        if not kwargs.get("stream"):
            raise AssertionError("did not expect a non-streaming call")
        return ExplodingStream()

    monkeypatch.setattr(
        tool_runner.client.chat.completions, "create", fake_create
    )

    events = list(tool_runner.run_tool_loop_streaming(
        df=df,
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="You are helpful.",
        context_label="test",
    ))

    token_events = [e for e in events if e["type"] == "token"]
    error_events = [e for e in events if e["type"] == "error"]

    # The token before the exception was still yielded (documented
    # trade-off: we can't retract already-streamed content).
    assert [e["content"] for e in token_events] == ["Hello"]
    assert len(error_events) == 1
    assert "connection dropped" in error_events[0]["message"]