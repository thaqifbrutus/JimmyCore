"""
Tests for app.services.tool_runner.run_tool_loop.

Only the OpenRouter call is mocked. Fake response objects mirror the
OpenAI SDK shape (choices[0].message.content / .tool_calls).
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
        _make_response(content="OK, I'll use another approach."),
    ]
    captured = _patch_client(monkeypatch, responses)

    result = tool_runner.run_tool_loop(
        df=df,
        messages=[{"role": "user", "content": "go"}],
        system_prompt="You are helpful.",
        context_label="test",
    )

    assert result["status"] == "ok"
    second_messages = captured[1]["messages"]
    tool_msg = next(m for m in second_messages if m.get("role") == "tool")
    content = json.loads(tool_msg["content"])
    assert "error" in content
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
    tool_msg = next(
        m for m in captured[1]["messages"] if m.get("role") == "tool"
    )
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