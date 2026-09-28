import json
from collections import Counter
from openai import OpenAI
from pydantic import ValidationError

from app.config import OPENROUTER_API_KEY, AI_MODEL, OPENROUTER_APP_NAME, OPENROUTER_APP_URL, AI_FALLBACK_MODELS
from app.schemas import DatasetOverview, DATASET_OVERVIEW_JSON_SCHEMA

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

_OPENROUTER_HEADERS = {
    "HTTP-Referer": OPENROUTER_APP_URL,
    "X-Title": OPENROUTER_APP_NAME,
}

SYSTEM_PROMPT = """

You are JimmyCore AI, a data analyst embedded in a platform for exploring
official government datasets and user-uploaded data. You help users
understand what a dataset contains and answer questions about it by
querying the actual data with the tools provided.

When the profiling results include a "source" field, this dataset comes
from an official government open-data catalog. In that case:
- Name the source agency and category plainly when relevant.
- Stick to what the data shows. Do not add outside knowledge, draw legal,
  policy, or causal conclusions the data doesn't support, or speculate.
- Cite the source when the user is making a decision based on the answer.

Tool use:
- You have tools to inspect and query the dataset. Use them liberally —
  don't guess at numbers when you can compute them.
- Prefer computing over recalling. If a user asks "how many rows have X,"
  call filter_rows and report the actual count.
- When a tool returns truncated: true, tell the user the result was capped
  and offer to narrow the query.
- Once you have the information needed to answer, STOP calling tools and
  write the answer. Do not keep calling tools just to be thorough. A
  single good aggregate is usually enough to answer a "which / how many /
  top N" question — you don't need to also filter the underlying rows.

When a tool errors:
- If the error says you already called that tool with those arguments:
  DO NOT call it again. You already have that result. This is a signal
  to STOP, not to retry. Either use what you have already gathered to
  answer the question, or call a genuinely DIFFERENT tool with DIFFERENT
  arguments. If neither is needed, answer with what you have.
- For any other error (unknown column, invalid operator, wrong value
  type), read the error message, fix your arguments, and try once more.
  Do not give up after one failure.

When you cannot answer: if the data doesn't contain what's needed, say
so plainly. Do not fabricate. Offer what you can answer.

Tone: professional, conversational, precise. Write for mixed audiences —
developers, analysts, project managers. Avoid jargon where possible.

"""


# ---------------------------------------------------------------------------
# Source context — the "source" field mentioned in SYSTEM_PROMPT above.
# Pure function, no I/O, so it's directly unit-testable without a database
# or an API call: given a catalog dataset's fields, build the dict that
# gets embedded into profile_data before it's sent to the model.
# ---------------------------------------------------------------------------

def build_source_context(catalog_dataset) -> dict:
    """
    Builds the "source" dict injected into profile_data for a
    catalog-sourced report, so the model can cite it per SYSTEM_PROMPT's
    instructions above. Takes any object with the relevant attributes
    (a real CatalogDataset row, or a stand-in in tests) rather than
    importing the model directly — keeps this module decoupled from the
    DB layer, same as everything else in ai_service.py.
    """
    coverage = None
    if catalog_dataset.dataset_begin and catalog_dataset.dataset_end:
        coverage = f"{catalog_dataset.dataset_begin}\u2013{catalog_dataset.dataset_end}"

    return {
        "type": "official_government_dataset",
        "agency": catalog_dataset.source,
        "category": catalog_dataset.category_en,
        "subcategory": catalog_dataset.subcategory_en,
        "coverage": coverage,
    }


def _is_degenerate_output(text: str, min_length: int = 400) -> bool:
    if not text or len(text) < min_length:
        return False

    for chunk_size in (1, 2, 4, 8, 16):
        chunk = text[:chunk_size]
        if not chunk.strip():
            if chunk_size == 1 and text.strip() == "":
                return True
            continue
        repeated = chunk * (min_length // max(chunk_size, 1) + 1)
        if text.startswith(repeated[:min_length]):
            return True

    words = text.split()
    if len(words) < 20:
        return False

    unique_word_ratio = len(set(words)) / len(words)
    if unique_word_ratio < 0.15:
        return True

    most_common_count = max(Counter(words).values())
    if (most_common_count / len(words)) > 0.4:
        return True

    return False


def _is_structured_output_unsupported_error(exc: Exception) -> bool:
    message = str(exc).lower()
    mentions_response_format = "response_format" in message or "json_schema" in message
    mentions_unsupported = any(
        phrase in message
        for phrase in ("does not support", "not supported", "unsupported", "no endpoints found")
    )
    return mentions_response_format and mentions_unsupported


def _is_daily_quota_error(exc: Exception) -> bool:
    """
    True when OpenRouter reports the account-wide free-tier daily limit is
    exhausted. This is a per-ACCOUNT budget shared by every free model, so
    when it fires, no fallback model has any chance of succeeding — the
    only correct response is to stop walking the chain and surface the
    error to the user.

    Matched by inspecting the error text because OpenRouter surfaces this
    as a 429 with a limit_source field; there's no dedicated exception
    class from the OpenAI SDK for it.
    """
    text = str(exc).lower()
    return (
        "429" in text
        and ("free-models-per-day" in text or "openrouter_free_tier_daily" in text)
    )


def _call_ai_model(
    messages: list,
    context_label: str,
    max_tokens: int,
    model: str = AI_MODEL,
    response_format: dict | None = None,
) -> dict:
    """Call OpenRouter, trying the configured model chain when needed."""
    model_chain = [model] + [
        fallback_model
        for fallback_model in AI_FALLBACK_MODELS
        if fallback_model != model
    ]
    last_error = None
    last_error_type = None

    for attempt, current_model in enumerate(model_chain):
        try:
            kwargs = {
                "model": current_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "extra_headers": _OPENROUTER_HEADERS,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)

        except Exception as exc:
            # Daily quota exhaustion is account-wide: no fallback can help.
            # Break immediately instead of walking the chain (~15s of
            # guaranteed-fail calls + per-model log noise).
            if _is_daily_quota_error(exc):
                print(
                    "WARNING: OpenRouter daily quota exhausted — "
                    "aborting model chain."
                )
                last_error = str(exc)
                last_error_type = "daily_quota_exhausted"
                break

            error_type = None
            if response_format is not None and _is_structured_output_unsupported_error(exc):
                error_type = "structured_output_unsupported"
                print(
                    f"WARNING: Model {current_model} does not support structured output "
                    f"(response_format) for {context_label}. Trying the next model."
                )
            else:
                print(f"ERROR: OpenRouter call failed for {context_label} (model={current_model}): {exc}")

            last_error = str(exc)
            last_error_type = error_type
            continue

        choice = response.choices[0] if response.choices else None
        finish_reason = choice.finish_reason if choice else None
        text = choice.message.content if choice and choice.message else None

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        print(
            f"INFO: AI call complete | context={context_label} | model={current_model} | "
            f"finish_reason={finish_reason} | usage={usage}"
        )

        result = {
            "status": "ok",
            "text": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "error": None,
            "error_type": None,
        }

        if not _needs_retry(result):
            return result

        print(
            f"INFO: AI output for {context_label} from {current_model} was unusable "
            f"(empty, truncated, or degenerate). Trying the next model."
        )

    return {
        "status": "error",
        "text": None,
        "finish_reason": None,
        "usage": None,
        "error": last_error or "All configured models failed.",
        "error_type": last_error_type,
    }


def _needs_retry(call_result: dict) -> bool:
    if call_result["status"] == "error":
        return True

    text = call_result["text"]

    if call_result["finish_reason"] == "length" and not text:
        return True

    if not text:
        return True

    if _is_degenerate_output(text):
        return True

    return False


def _failure_result(reason: str) -> dict:
    return {
        "status": "failed",
        "reason": reason,
        "content": None,
    }


def _success_result(content) -> dict:
    return {
        "status": "ok",
        "reason": None,
        "content": content,
    }


# ---------------------------------------------------------------------------
# Dataset overview — produces a short orientation, suggested questions,
# and a hint (primary_column + primary_metric) about what to chart.
# Structured output via json_schema, with the same unsupported-model
# fallback path the old technical-context generator had.
# ---------------------------------------------------------------------------

_DATASET_OVERVIEW_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "dataset_overview",
        "strict": True,
        "schema": DATASET_OVERVIEW_JSON_SCHEMA,
    },
}


def _build_dataset_overview_prompt(profile_data: dict, original_filename: str, inline_schema: bool = False) -> str:
    schema_instructions = f"""
You MUST respond with ONLY a single JSON object — no markdown fences, no
commentary before or after — that strictly matches this JSON Schema:

{json.dumps(DATASET_OVERVIEW_JSON_SCHEMA, indent=2)}
""" if inline_schema else ""

    return f"""

You are writing an overview of the following dataset for a user who just
selected it and is about to ask questions about it.

--- DATASET PROFILE ---
{json.dumps(profile_data, indent=2)}
--- END DATASET PROFILE ---
{schema_instructions}
Write 2-3 short paragraphs (no more) covering:
1. What the dataset appears to cover — the domain, the entities, the key
   dimensions (time, geography, categories).
2. What's notable — coverage period, granularity, important columns,
   anything a new user should know before asking questions.

Do NOT produce an issue list, severity ratings, or a "ready for use"
verdict. This is an orientation, not an audit.

Then produce 3-5 suggested starter questions the user might ask. They
must be concrete, answerable from the data shown above, and varied in
difficulty — mix simple lookups and counts with more analytical ones
("What's the trend over time?", "Which state has the highest X?", "Are
there missing values in the Y column?").

Finally, choose what to chart. Two related fields:

- primary_column: a categorical, time, or low-cardinality dimension
  to plot on the x-axis — a state column, a category column, a year
  column, a type column. Do NOT pick a free-text identifier, a
  mostly-null column, or a high-cardinality column (more than ~50
  distinct values). If no column is chart-worthy, respond with
  "primary_column": null.

- primary_metric: if the dataset contains a NUMERIC column whose sum
  by primary_column would tell the reader something interesting — a
  count of cases, a total, a measured quantity — name it here. The
  chart will then show "sum of {{primary_metric}} by {{primary_column}}"
  (for example, "sum of total_cases by state"), which is usually more
  informative than a raw frequency count. If there is no numeric
  column, or none of them makes sense summed by the chosen
  primary_column, respond with "primary_metric": null. A dataset that
  is purely categorical (IDs and labels, no measures) should have
  "primary_metric": null, and the chart will fall back to a count of
  rows per category.

Respond with JSON matching the provided schema. The "overview" field is
the prose; the "suggested_questions" field is the list;
"primary_column" and "primary_metric" are the chart hints (either may
be null).
"""


def _parse_dataset_overview(raw_text: str) -> DatasetOverview:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    parsed = json.loads(cleaned)
    return DatasetOverview.model_validate(parsed)


def generate_dataset_overview(profile_data: dict, original_filename: str) -> dict:
    prompt = _build_dataset_overview_prompt(profile_data, original_filename)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    result = _call_ai_model(
        messages,
        context_label=f"overview:{original_filename}",
        max_tokens=4000,
        response_format=_DATASET_OVERVIEW_RESPONSE_FORMAT,
    )

    parsed = None
    parse_error = None
    if not _needs_retry(result):
        try:
            parsed = _parse_dataset_overview(result["text"])
        except (json.JSONDecodeError, ValidationError) as exc:
            parse_error = str(exc)
            print(f"WARNING: Dataset overview JSON failed validation for {original_filename}: {exc}")

    if parsed is not None:
        return _success_result(parsed.model_dump())

    use_inline_schema_fallback = result.get("error_type") == "structured_output_unsupported"

    if use_inline_schema_fallback:
        print(
            f"INFO: Retrying generate_dataset_overview for {original_filename} "
            f"WITHOUT response_format (model does not support structured output)"
        )
    else:
        print(f"INFO: Retrying generate_dataset_overview for {original_filename}")

    fallback_prompt = _build_dataset_overview_prompt(
        profile_data, original_filename, inline_schema=use_inline_schema_fallback
    )
    fallback_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": fallback_prompt},
    ]

    retry_result = _call_ai_model(
        fallback_messages,
        context_label=f"overview:{original_filename}:retry",
        max_tokens=3000,
        response_format=None if use_inline_schema_fallback else _DATASET_OVERVIEW_RESPONSE_FORMAT,
    )

    if not _needs_retry(retry_result):
        try:
            retry_parsed = _parse_dataset_overview(retry_result["text"])
            return _success_result(retry_parsed.model_dump())
        except (json.JSONDecodeError, ValidationError) as exc:
            print(f"WARNING: Retry dataset overview JSON also failed validation for {original_filename}: {exc}")
            return _failure_result(f"Model output failed schema validation on retry: {exc}")

    return _failure_result(
        retry_result.get("error")
        or parse_error
        or "Model produced empty, truncated, or degenerate output twice in a row."
    )


# ---------------------------------------------------------------------------
# Q&A — tool-driven. Delegates the loop to tool_runner, which owns the
# model chain, iteration cap, and context-mode fallback.
# ---------------------------------------------------------------------------

def answer_dataset_question(
    df,
    profile_data: dict,
    original_filename: str,
    question: str,
    conversation_history: list = None,
) -> dict:
    """
    Delegates to tool_runner.run_tool_loop. Returns
    {"status", "content", "reason", "error_type", "tool_calls_log"}.

    Local import of tool_runner to avoid a module-level import cycle
    (tool_runner imports from this module).
    """
    from app.services import tool_runner

    context_message = f"""
The user is asking questions about a dataset called "{original_filename}".
Here is a profile of the data for context:

{json.dumps(profile_data, indent=2)}

Use the tools available to you to inspect and query the actual data when
answering. Prefer computing over recalling.
"""

    messages = [
        {"role": "user", "content": context_message},
        {
            "role": "assistant",
            "content": "Understood. I have the profile and access to the data. Ready to answer.",
        },
    ]

    if conversation_history:
        for turn in conversation_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": question})

    return tool_runner.run_tool_loop(
        df=df,
        messages=messages,
        system_prompt=SYSTEM_PROMPT,
        context_label=f"qa:{original_filename}",
    )