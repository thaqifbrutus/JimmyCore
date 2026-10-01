"""
Tool-calling loop for the conversational analyst.

## Future work (deliberately NOT implemented in this revamp)
- Chat history persistence (needs its own table/column + retention design)
- Renaming QualityReport -> AnalysisSession, quality_reports -> analysis_sessions
- Dropping the overall_status column
- Document (PDF/text) ingestion
- Agentic multi-dataset queries ("compare this to last year's dataset")
- Streaming chat responses  (DONE — see run_tool_loop_streaming)
- Tool-call result caching within a single chat session
"""
from __future__ import annotations

import json

import pandas as pd

from app.config import AI_FALLBACK_MODELS, AI_MODEL
from app.services import data_tools
from app.services.ai_service import (
    _OPENROUTER_HEADERS,
    _is_daily_quota_error,
    _is_degenerate_output,
    client,
)


MAX_TOOL_ITERATIONS = 6

# Cap on how many chart data points we store per tool call for the
# frontend to render inline. Rendering sanity, not a data limit.
_CHART_DATA_MAX = 20

# Parameters that do not meaningfully change what a tool call "is" for
# repeat-detection purposes. `limit` is here because the model routinely
# varies it (limit=10 vs limit=20) between otherwise-identical calls.
_SIGNATURE_IGNORED_ARGS = {"limit"}


def _is_tools_unsupported_error(exc: Exception) -> bool:
    """Same pattern-matching approach as _is_structured_output_unsupported_error."""
    message = str(exc).lower()
    mentions_tools = "tool" in message or "function" in message
    mentions_unsupported = any(
        phrase in message
        for phrase in (
            "does not support",
            "not supported",
            "unsupported",
            "no endpoints found",
        )
    )
    return mentions_tools and mentions_unsupported


def _summarize_result(result: dict) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    if "rows" in result:
        return f"{len(result.get('rows', []))} rows"
    if "values" in result:
        return f"{len(result.get('values', []))} values"
    if "results" in result:
        return f"{len(result.get('results', []))} groups"
    if "columns" in result:
        return f"{len(result.get('columns', []))} columns"
    if "column" in result:
        return f"column '{result.get('column')}'"
    return "ok"


def _signature(name: str, args: dict) -> tuple[str, str]:
    """
    Build a signature for repeat-detection. Normalizes away parameters
    that don't change the semantic identity of the call.
    """
    normalized = {
        k: v for k, v in args.items() if k not in _SIGNATURE_IGNORED_ARGS
    }
    return (name, json.dumps(normalized, sort_keys=True))


def _build_chart_data(name: str, result: dict) -> list[dict] | None:
    """
    Extract a small, chart-ready list from a value_counts or aggregate
    result. Returns None for anything else (including errors).
    """
    if "error" in result:
        return None
    if name == "value_counts" and "values" in result:
        return [
            {"value": v.get("value"), "count": v.get("count")}
            for v in result["values"][:_CHART_DATA_MAX]
        ]
    if name == "aggregate" and "results" in result:
        return [
            {"group": r.get("group"), "value": r.get("value")}
            for r in result["results"][:_CHART_DATA_MAX]
        ]
    return None


def _build_truncation_warning(df: pd.DataFrame) -> str:
    """Empty string when the DataFrame is fully loaded, or has no tag."""
    if not df.attrs.get("may_be_truncated"):
        return ""
    limit = df.attrs.get("fetch_limit", "?")
    return (
        f"\n\nIMPORTANT: This dataset was fetched with a limit of {limit} rows, "
        f"and the fetch returned exactly that many — so the dataset may have more "
        f"rows than you can currently see. If a user asks about totals, "
        f"percentages, or \"how many,\" mention that your view may be partial."
    )


def _log_tool_result(iteration: int, name: str, result: dict) -> None:
    """Single-line result summary, printed after the tool_call line."""
    if "error" in result:
        summary = f"ERROR: {result['error']}"
    elif "rows" in result:
        summary = (
            f"rows={len(result['rows'])} matched={result.get('matched')} "
            f"truncated={result.get('truncated')}"
        )
    elif "results" in result:
        summary = (
            f"groups={len(result['results'])} truncated={result.get('truncated')}"
        )
    else:
        summary = f"keys={list(result.keys())}"
    print(f"INFO: tool_result | iter={iteration} | tool={name} | {summary}")


def _execute_tool_call(
    df: pd.DataFrame,
    name: str,
    raw_args: str,
    seen_signatures: set[tuple[str, str]],
    iteration: int,
) -> tuple[dict, dict]:
    """
    Runs one tool call. Returns (result, log_entry). Shared between the
    non-streaming and streaming loops so the semantics of parse / dedup /
    execute / log can't drift between them.

    raw_args is the raw JSON string from the model — parsing is part of
    this helper so both callers treat parse failures identically.
    """
    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        args = {}
        result = {"error": f"could not parse arguments: {exc}"}
    else:
        signature = _signature(name, args)
        if signature in seen_signatures:
            result = {
                "error": (
                    f"You ALREADY called {name} with these exact "
                    f"arguments in this turn. The result has not "
                    f"changed and will not change. DO NOT call "
                    f"{name} with these arguments again — either "
                    f"(a) use the result you already have to answer "
                    f"the question, or (b) call a genuinely "
                    f"DIFFERENT tool, or the same tool with "
                    f"DIFFERENT arguments. If you have enough "
                    f"information to answer, answer now."
                )
            }
        else:
            fn = data_tools.TOOL_REGISTRY.get(name)
            if fn is None:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = fn(df, **args)
                except Exception as exc:
                    result = {"error": str(exc)}
            seen_signatures.add(signature)

    print(f"INFO: tool_call | iter={iteration} | tool={name} | args={args}")
    _log_tool_result(iteration, name, result)

    log_entry = {
        "name": name,
        "arguments": args,
        "result_summary": _summarize_result(result),
        "chart_data": _build_chart_data(name, result),
    }
    return result, log_entry


def _serialize_assistant_turn(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (message.tool_calls or [])
        ],
    }


def _call_model(
    messages: list[dict],
    model: str,
    tools_enabled: bool,
    max_tokens: int,
):
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "extra_headers": _OPENROUTER_HEADERS,
    }
    if tools_enabled:
        kwargs["tools"] = data_tools.TOOL_SCHEMAS
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


def _call_model_streaming(
    messages: list[dict],
    model: str,
    tools_enabled: bool,
    max_tokens: int,
):
    """Streaming variant of _call_model. Returns an iterator of chunks."""
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "extra_headers": _OPENROUTER_HEADERS,
        "stream": True,
    }
    if tools_enabled:
        kwargs["tools"] = data_tools.TOOL_SCHEMAS
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


def _run_wrap_up_call(
    messages: list[dict],
    model: str,
    max_tokens: int,
) -> str | None:
    """
    One last call with tools REMOVED, asking the model to answer using the
    tool results already in the conversation history.

    Called when the loop can't make further progress. Returns the text, or
    None if the call failed / produced nothing.
    """
    wrap_up_messages = list(messages) + [{
        "role": "user",
        "content": (
            "You have used your tool-calling budget for this turn. "
            "Do not call any more tools. Using ONLY the information "
            "already gathered above (including the tool results you "
            "received earlier in this conversation), answer the user's "
            "original question now. If the gathered information only "
            "partially answers it, say so plainly and give the best "
            "partial answer you can from what you have."
        ),
    }]
    try:
        response = _call_model(
            wrap_up_messages, model, tools_enabled=False, max_tokens=max_tokens
        )
    except Exception as exc:
        print(f"WARNING: wrap-up call failed for model={model}: {exc}")
        return None

    choice = response.choices[0] if response.choices else None
    text = choice.message.content if choice and choice.message else None
    if not text:
        return None
    if _is_degenerate_output(text):
        print(f"WARNING: wrap-up output from model={model} was degenerate.")
        return None
    return text


def _run_loop_with_model(
    df: pd.DataFrame,
    messages: list[dict],
    model: str,
    tools_enabled: bool,
    max_tokens: int,
    context_label: str,
    seen_signatures: set[tuple[str, str]],
) -> dict:
    """
    Non-streaming tool loop. See run_tool_loop for the driving logic.
    """
    working_messages = list(messages)
    tool_calls_log: list[dict] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = _call_model(working_messages, model, tools_enabled, max_tokens)
        except Exception as exc:
            error_type = None
            if _is_daily_quota_error(exc):
                error_type = "daily_quota_exhausted"
            return {
                "status": "error",
                "content": None,
                "reason": str(exc),
                "error_type": error_type,
                "tool_calls_log": tool_calls_log,
            }

        choice = response.choices[0] if response.choices else None
        if choice is None:
            return {
                "status": "failed",
                "content": None,
                "reason": "Model returned no choices.",
                "tool_calls_log": tool_calls_log,
            }

        message = choice.message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return {
                "status": "ok",
                "content": message.content,
                "reason": None,
                "tool_calls_log": tool_calls_log,
            }

        working_messages.append(_serialize_assistant_turn(message))

        for tc in tool_calls:
            result, log_entry = _execute_tool_call(
                df, tc.function.name, tc.function.arguments or "{}",
                seen_signatures, iteration,
            )
            tool_calls_log.append(log_entry)
            working_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return {
        "status": "failed",
        "content": None,
        "reason": "tool loop exceeded iteration limit",
        "failure_type": "iteration_limit",
        "tool_calls_log": tool_calls_log,
        "_final_messages": working_messages,
    }


def _build_context_mode_prompt(df: pd.DataFrame) -> str:
    schema = data_tools.get_schema(df)
    sample = data_tools.get_sample_rows(df, n=30)
    return (
        "\n\n--- DATASET CONTEXT (tool-calling unavailable) ---\n"
        "You do not have tool access on this call. Use the schema and "
        "sample rows below to answer the user's question as best you can. "
        "If the question needs computation the sample doesn't support, say "
        "so plainly and offer what you can answer.\n\n"
        f"SCHEMA:\n{json.dumps(schema, indent=2, default=str)}\n\n"
        f"SAMPLE ROWS (up to 30):\n{json.dumps(sample, indent=2, default=str)}\n"
        "--- END DATASET CONTEXT ---\n"
    )


def run_tool_loop(
    df: pd.DataFrame,
    messages: list[dict],
    system_prompt: str,
    context_label: str,
    max_tokens: int = 4000,
) -> dict:
    """
    Non-streaming tool loop. `messages` should NOT include the system
    prompt — this function prepends it.
    """
    effective_system_prompt = system_prompt + _build_truncation_warning(df)
    full_messages = [{"role": "system", "content": effective_system_prompt}] + list(messages)

    models = [AI_MODEL] + [m for m in AI_FALLBACK_MODELS if m != AI_MODEL]

    seen_signatures: set[tuple[str, str]] = set()

    last_error: str | None = None
    all_rejected_tools = True

    for model in models:
        result = _run_loop_with_model(
            df, full_messages, model,
            tools_enabled=True, max_tokens=max_tokens,
            context_label=context_label,
            seen_signatures=seen_signatures,
        )

        if result["status"] == "ok":
            content = result.get("content")
            if content and _is_degenerate_output(content):
                return {
                    "status": "failed",
                    "content": None,
                    "reason": "Model produced degenerate output.",
                    "tool_calls_log": result.get("tool_calls_log", []),
                }
            return result

        if result.get("error_type") == "daily_quota_exhausted":
            print(
                f"WARNING: OpenRouter daily quota exhausted for "
                f"{context_label} — aborting model chain."
            )
            return {
                "status": "failed",
                "content": None,
                "reason": (
                    "OpenRouter daily free-tier quota exhausted. Try again "
                    "after the daily reset (see X-RateLimit-Reset in the error)."
                ),
                "error_type": "daily_quota_exhausted",
                "tool_calls_log": result.get("tool_calls_log", []),
            }

        if result.get("failure_type") == "iteration_limit":
            print(
                f"WARNING: Tool loop hit iteration cap for {context_label} "
                f"(model={model}). Attempting wrap-up call."
            )
            final_messages = result.get("_final_messages")
            if final_messages:
                wrap_up_text = _run_wrap_up_call(
                    final_messages, model=model, max_tokens=max_tokens
                )
                if wrap_up_text:
                    print(
                        f"INFO: Wrap-up call succeeded for {context_label} — "
                        f"returning synthesized answer from collected tool results."
                    )
                    return {
                        "status": "ok",
                        "content": wrap_up_text,
                        "reason": None,
                        "tool_calls_log": result.get("tool_calls_log", []),
                        "recovered_from_cap": True,
                    }
                print(
                    f"WARNING: Wrap-up call also failed for {context_label}. "
                    f"Returning original cap failure."
                )
            return result

        error_text = result.get("reason") or ""

        if _is_tools_unsupported_error(Exception(error_text)) or (
            "tool" in error_text.lower()
            and any(
                p in error_text.lower()
                for p in ("does not support", "not supported", "unsupported", "no endpoints found")
            )
        ):
            print(
                f"WARNING: Model {model} rejected tools for {context_label}. "
                f"Trying next model."
            )
            last_error = error_text
            continue

        all_rejected_tools = False
        last_error = error_text

    if all_rejected_tools:
        print(
            f"WARNING: All models rejected tools for {context_label}. "
            f"Falling back to context mode."
        )
        context_system = effective_system_prompt + _build_context_mode_prompt(df)
        context_messages = [
            {"role": "system", "content": context_system}
        ] + [m for m in messages if m.get("role") != "system"]

        for model in models:
            result = _run_loop_with_model(
                df, context_messages, model,
                tools_enabled=False, max_tokens=max_tokens,
                context_label=f"{context_label}:context_mode",
                seen_signatures=seen_signatures,
            )
            if result["status"] == "ok":
                result["context_mode"] = True
                return result
            last_error = result.get("reason") or last_error

    return {
        "status": "failed",
        "content": None,
        "reason": last_error or "All configured models failed.",
        "tool_calls_log": [],
    }


# ---------------------------------------------------------------------------
# Streaming tool loop (Round 1, Task 7a)
#
# Design notes, because the streaming path has sharp edges that the
# non-streaming path does not:
#
# 1. We only know whether a turn is "the answer" or "a tool-call round"
#    after we've seen the first delta. We use a `mode` state machine:
#    the first delta that carries content or tool_calls pins the mode for
#    the rest of that turn. Content deltas in "answer" mode are yielded
#    immediately; tool-call deltas are always accumulated silently.
#
# 2. If the model emits content and THEN a tool_call in the same turn
#    (which OpenAI-compatible APIs do not do in practice — it picks one
#    mode per turn), we've already yielded the content. Retracting is
#    impossible in SSE. Documented assumption.
#
# 3. If a model fails mid-stream — after tokens have already been yielded
#    — we do NOT fall through to the next model. The user saw partial
#    output; switching models and restarting would produce a garbled
#    answer. Emit an error and stop.
#
# 4. Context-mode fallback is deliberately NOT implemented for streaming.
#    The non-streaming run_tool_loop still has it. If every model rejects
#    tools on the streaming path, the caller gets an error event and can
#    retry through the non-streaming endpoint if desired.
#
# 5. Degenerate-output detection is not applied to streamed content —
#    by the time we could measure it, the tokens are already on the
#    wire. Applied to the wrap-up path, which is non-streaming.
# ---------------------------------------------------------------------------


def _stream_with_model(
    df: pd.DataFrame,
    messages: list[dict],
    model: str,
    tools_enabled: bool,
    max_tokens: int,
    context_label: str,
    seen_signatures: set[tuple[str, str]],
):
    """
    Generator. Yields events for a single model's run:

      {"type": "token", "content": "<str>"}
      {"type": "done", "content": "<full text>", "tool_calls_log": [...]}
      {"type": "error", "message": "<str>", "error_type": "<str|None>"}

    Returns (falls off the end) after yielding "done" or "error".
    """
    working_messages = list(messages)
    tool_calls_log: list[dict] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            stream = _call_model_streaming(
                working_messages, model, tools_enabled, max_tokens
            )
        except Exception as exc:
            error_type = None
            if _is_daily_quota_error(exc):
                error_type = "daily_quota_exhausted"
            yield {
                "type": "error",
                "message": str(exc),
                "error_type": error_type,
            }
            return

        # mode: None until first content/tool-call delta, then "answer" or "tools"
        mode: str | None = None
        content_buffer = ""
        tool_calls_by_index: dict[int, dict] = {}
        finish_reason = None

        try:
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                has_tool_calls = bool(getattr(delta, "tool_calls", None))
                content = getattr(delta, "content", None)

                if has_tool_calls:
                    if mode is None:
                        mode = "tools"
                    if mode == "tools":
                        for tc in delta.tool_calls:
                            idx = getattr(tc, "index", 0)
                            if idx not in tool_calls_by_index:
                                tool_calls_by_index[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            slot = tool_calls_by_index[idx]
                            tc_id = getattr(tc, "id", None)
                            if tc_id:
                                slot["id"] = tc_id
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                fn_name = getattr(fn, "name", None)
                                fn_args = getattr(fn, "arguments", None)
                                if fn_name:
                                    slot["function"]["name"] = fn_name
                                if fn_args:
                                    slot["function"]["arguments"] += fn_args

                if content:
                    content_buffer += content
                    if mode is None:
                        mode = "answer"
                    if mode == "answer":
                        yield {"type": "token", "content": content}

        except Exception as exc:
            yield {
                "type": "error",
                "message": str(exc),
                "error_type": None,
            }
            return

        # End of stream for this iteration. Two cases:
        #   - We were in "tools" mode → execute, append results, loop.
        #   - We were in "answer" mode (or stayed None → empty answer) → done.
        if mode == "tools" and tool_calls_by_index:
            sorted_calls = [
                tool_calls_by_index[i] for i in sorted(tool_calls_by_index.keys())
            ]
            working_messages.append({
                "role": "assistant",
                "content": content_buffer or None,
                "tool_calls": sorted_calls,
            })
            for tc in sorted_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"] or "{}"
                result, log_entry = _execute_tool_call(
                    df, name, raw_args, seen_signatures, iteration,
                )
                tool_calls_log.append(log_entry)
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, default=str),
                })
            continue

        # "answer" mode — this turn was the response.
        yield {
            "type": "done",
            "content": content_buffer,
            "tool_calls_log": tool_calls_log,
        }
        return

    # Iteration cap. Try a non-streaming wrap-up, yield its content as a
    # single token so the frontend still gets something usable.
    print(
        f"WARNING: Streaming tool loop hit iteration cap for {context_label} "
        f"(model={model}). Attempting wrap-up."
    )
    wrap_up_text = _run_wrap_up_call(working_messages, model=model, max_tokens=max_tokens)
    if wrap_up_text:
        yield {"type": "token", "content": wrap_up_text}
        yield {
            "type": "done",
            "content": wrap_up_text,
            "tool_calls_log": tool_calls_log,
        }
        return

    yield {
        "type": "error",
        "message": "tool loop exceeded iteration limit and wrap-up failed",
        "error_type": None,
    }


def run_tool_loop_streaming(
    df: pd.DataFrame,
    messages: list[dict],
    system_prompt: str,
    context_label: str,
    max_tokens: int = 4000,
):
    """
    Generator. Same control flow as run_tool_loop, but the final answer's
    tokens stream to the caller. Yields the event dicts documented on
    _stream_with_model.

    Model chain: try the primary model first. If it fails with
    tools-unsupported before any token has been yielded, try each fallback
    in order. If a model fails after tokens have been yielded, emit the
    error and stop — do not fall through.
    """
    effective_system_prompt = system_prompt + _build_truncation_warning(df)
    full_messages = [{"role": "system", "content": effective_system_prompt}] + list(messages)

    models = [AI_MODEL] + [m for m in AI_FALLBACK_MODELS if m != AI_MODEL]

    seen_signatures: set[tuple[str, str]] = set()
    any_token_yielded = False
    last_error: dict | None = None

    for model in models:
        try:
            generator = _stream_with_model(
                df, full_messages, model,
                tools_enabled=True, max_tokens=max_tokens,
                context_label=context_label,
                seen_signatures=seen_signatures,
            )
            for event in generator:
                if event["type"] == "token":
                    any_token_yielded = True
                    yield event
                elif event["type"] == "done":
                    yield event
                    return
                elif event["type"] == "error":
                    if any_token_yielded:
                        # Mid-stream failure. Do not fall through —
                        # the user already saw partial output.
                        print(
                            f"WARNING: Streaming failed mid-turn for "
                            f"{context_label} (model={model}). Emitting error "
                            f"without retrying on next model."
                        )
                        yield event
                        return
                    # Pre-stream failure. Remember and try next model.
                    print(
                        f"WARNING: Streaming failed for {context_label} "
                        f"(model={model}) before any tokens: "
                        f"{event.get('message')}. Trying next model."
                    )
                    last_error = event
                    break  # out of the event loop; continue to next model
        except Exception as exc:
            # An uncaught exception in the generator itself.
            if any_token_yielded:
                yield {
                    "type": "error",
                    "message": str(exc),
                    "error_type": None,
                }
                return
            last_error = {
                "type": "error",
                "message": str(exc),
                "error_type": None,
            }
            continue

    # All models exhausted.
    if last_error:
        yield last_error
    else:
        yield {
            "type": "error",
            "message": "All configured models failed.",
            "error_type": None,
        }