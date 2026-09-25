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
    _is_degenerate_output,
    client,
)


MAX_TOOL_ITERATIONS = 6


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

    When the iteration cap is hit, the returned dict also carries
    "failure_type": "iteration_limit" so the caller can distinguish a
    deterministic cap failure from a transient API error.

    seen_signatures is owned by the caller (run_tool_loop) and shared across
    the whole model chain within a single user turn — so a repeat call of
    the same tool with the same arguments gets a repeated-call error
    instead of being executed again, even if the primary model failed over
    to a fallback.
    """
    working_messages = list(messages)
    tool_calls_log: list[dict] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = _call_model(working_messages, model, tools_enabled, max_tokens)
        except Exception as exc:
            return {
                "status": "error",
                "content": None,
                "reason": str(exc),
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
                signature = (name, json.dumps(args, sort_keys=True))
                if signature in seen_signatures:
                    result = {
                        "error": (
                            f"You already called {name} with these exact "
                            f"arguments in this turn. The result has not "
                            f"changed. Try different arguments, or answer "
                            f"the question with what you already have."
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
    prepends it. Returns:
        {"status": "ok"|"failed", "content": str|None,
         "reason": str|None, "tool_calls_log": [...]}
    """
    full_messages = [{"role": "system", "content": system_prompt}] + list(messages)

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

        # An iteration-cap hit is deterministic: every model will hit the
        # same cap on the same DataFrame. Falling through to the next model
        # would just burn more tokens on an identical failure. Return now.
        if result.get("failure_type") == "iteration_limit":
            print(
                f"WARNING: Tool loop hit iteration cap for {context_label} "
                f"(model={model}). Returning immediately — this failure is "
                f"not model-specific."
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
        context_system = system_prompt + _build_context_mode_prompt(df)
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