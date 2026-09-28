"""
Tool-calling loop for the conversational analyst.

## Future work (deliberately NOT implemented in this revamp)
- Chat history persistence (needs its own table/column + retention design)
- Renaming QualityReport -> AnalysisSession, quality_reports -> analysis_sessions
- Dropping the overall_status column
- Document (PDF/text) ingestion
- Agentic multi-dataset queries ("compare this to last year's dataset")
- Streaming chat responses
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
# frontend to render inline. Rendering sanity, not a data limit — the
# tools themselves already cap at 50. 20 is what a bar chart can show
# legibly in the chat panel width.
_CHART_DATA_MAX = 20


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

# Parameters that do not meaningfully change what a tool call "is" for
# repeat-detection purposes. `limit` is here because the model routinely
# varies it (limit=10 vs limit=20) between otherwise-identical calls,
# trying to "get more rows" of the same filter — which defeats exact-match
# deduplication without changing the underlying query. The row cap is 50
# regardless, so lowering the limit can never produce a new result.
_SIGNATURE_IGNORED_ARGS = {"limit"}


def _signature(name: str, args: dict) -> tuple[str, str]:
    """
    Build a signature for repeat-detection. Normalizes away parameters
    that don't change the semantic identity of the call (currently: limit)
    so that back-to-back calls like filter_rows(state=='X', limit=10) and
    filter_rows(state=='X', limit=20) collide and the second is rejected.
    """
    normalized = {
        k: v for k, v in args.items() if k not in _SIGNATURE_IGNORED_ARGS
    }
    return (name, json.dumps(normalized, sort_keys=True))


def _build_chart_data(name: str, result: dict) -> list[dict] | None:
    """
    Extract a small, chart-ready list from a value_counts or aggregate
    result. Returns None for anything else (including errors) — the
    frontend treats None as "no chart".

    Capped at _CHART_DATA_MAX entries. Doesn't reuse the response's own
    truncated flag — the chart is a presentation nicety, and a 50-bar
    chart is unusable in a chat panel regardless of whether the data was
    truncated upstream.
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
    """
    Empty string when the DataFrame is fully loaded, or when it doesn't
    carry the tag (e.g. a DataFrame constructed in tests without going
    through gov_data_client). Only fires when the fetch hit its limit
    exactly, which is the only signal we have that there may be more rows
    upstream that we didn't retrieve.
    """
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
    """
    Single-line result summary for a tool call — printed immediately after
    the tool_call line, so the log shows call-then-result as a pair.

    Special-cases the shape we care about during debugging (rows / groups /
    errors) rather than dumping the whole dict, which would flood the log.
    """
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


def _run_wrap_up_call(
    messages: list[dict],
    model: str,
    max_tokens: int,
) -> str | None:
    """
    One last call with tools REMOVED, asking the model to answer using the
    tool results already in the conversation history.

    Called when the loop can't make further progress (iteration cap, or
    the model is stuck calling the same tool). Better to hand the user a
    partial answer built from real tool output than a raw "tool loop
    exceeded iteration limit" failure — the results are already in the
    conversation; we just need the model to synthesize them.

    Returns the text, or None if the call failed / produced nothing.
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
    Runs the tool loop against a single model. Returns a dict with at least
    {"status": "ok"|"failed"|"error", "content", "reason", "tool_calls_log"}.

    failure_type="iteration_limit" is set when the cap is hit (deterministic
    across models). error_type="daily_quota_exhausted" is set when
    OpenRouter reports the account-wide free-tier limit (also deterministic
    across models — they all share it).

    On iteration_limit, the returned dict also carries "_final_messages" —
    the full working conversation including tool results — so the caller
    can attempt a wrap-up call without re-running the tool loop.

    seen_signatures is owned by the caller (run_tool_loop) and shared across
    the whole model chain within a single user turn — so a repeat call of
    the same tool with the same arguments gets a repeated-call error
    instead of being executed again, even after a fallback handoff.
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
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"

            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                args = {}
                signature = None
                result = {"error": f"could not parse arguments: {exc}"}
            else:
                # sort_keys=True is load-bearing: the model serializes the
                # same conceptual call with different key order between
                # turns ({'column': 'date', 'operator': '=='} vs
                # {'value': ..., 'operator': '=='}), so unsorted JSON would
                # never collide and repeat detection would silently miss.
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

            print(
                f"INFO: tool_call | iter={iteration} | tool={name} | args={args}"
            )
            _log_tool_result(iteration, name, result)

            tool_calls_log.append({
                "name": name,
                "arguments": args,
                "result_summary": _summarize_result(result),
                "chart_data": _build_chart_data(name, result),
            })

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
        # Hand the working conversation back to the caller so it can run a
        # wrap-up call that has access to the tool results collected so far.
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
    `messages` should NOT include the system prompt — this function
    prepends it (plus a truncation warning when the DataFrame's fetch
    hit its limit). Returns:
        {"status": "ok"|"failed", "content": str|None,
         "reason": str|None, "error_type": str|None, "tool_calls_log": [...]}
    """
    effective_system_prompt = system_prompt + _build_truncation_warning(df)
    full_messages = [{"role": "system", "content": effective_system_prompt}] + list(messages)

    models = [AI_MODEL] + [m for m in AI_FALLBACK_MODELS if m != AI_MODEL]

    # Scoped per run_tool_loop invocation, reset on every call. Deliberately
    # not a module global — the "same call twice" rule applies within a
    # single user turn, not across separate questions.
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

        # Daily quota exhaustion is account-wide — every model in the chain
        # shares the same budget. Break now, don't burn the remaining calls.
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

        # Iteration cap: the model is stuck. Rather than handing the user a
        # raw "tool loop exceeded iteration limit", try a wrap-up call with
        # the SAME model — no tools, no cap, just "synthesize what you
        # already have." The tool results are already in _final_messages.
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

        # A non-tools failure (API error, degenerate output) — try the next
        # model too, but note it for the fallback decision.
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