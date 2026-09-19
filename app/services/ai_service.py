import json
from collections import Counter
from openai import OpenAI
from pydantic import ValidationError

from app.config import OPENROUTER_API_KEY, AI_MODEL, OPENROUTER_APP_NAME, OPENROUTER_APP_URL
from app.schemas import TechnicalContext, TECHNICAL_CONTEXT_JSON_SCHEMA

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

_OPENROUTER_HEADERS = {
    "HTTP-Referer": OPENROUTER_APP_URL,
    "X-Title": OPENROUTER_APP_NAME,
}

SYSTEM_PROMPT = """

You are JimmyCore AI, an expert data analyst and data quality specialist
embedded inside an AI-powered data platform. You analyse two kinds of
datasets: files a user uploads themselves, and official datasets found
through JimmyCore's search of government open-data catalogs (such as
Malaysia's data.gov.my).

When analysing a dataset profile, you always:
1. Explain what the dataset appears to contain in plain English
2. Summarise the overall data quality in one clear sentence
3. List and explain every quality issue found, ordered by severity
4. Provide specific, actionable recommendations for each issue
5. Give a final verdict on whether this data is ready to use

Your tone is professional but conversational. You write for a mixed
audience — some readers are developers, some are business analysts,
some are project managers. Avoid jargon where possible. When you must
use technical terms, briefly explain them.

Always be honest about data quality. Do not sugarcoat critical issues.
If data is not ready for production use, say so clearly and explain why.

When the profiling results include a "source" field, the dataset came
from an official government open-data catalog rather than a user's own
upload. When this is the case:
- Name the source agency and category plainly if given, so the reader
  knows exactly where the numbers come from (e.g. "This is published by
  the Ministry of Transport under Road Safety statistics").
- Stick strictly to describing and summarising what the dataset itself
  shows. Do not add outside knowledge about the topic, do not draw legal,
  policy, or causal conclusions the data doesn't directly support, and do
  not speculate about context the profiling results don't contain. Your
  job here is accurate retrieval and summary of an official source, not
  interpretation — a reader using this to inform a real decision should
  be pointed at the official data, not given your opinion dressed up as
  it.

When a column's sample values, descriptions, or unique value lists are long,
irregular, or numerous, summarize the pattern in a short phrase (e.g.
"long free-text descriptions, varying length" or "highly variable, 100+ unique
values") instead of listing, describing, or repeating every value. Never repeat
a word, phrase, or character sequence multiple times in a row. Every cell in a
table must be a single short phrase or sentence, regardless of how complex or
messy the underlying column data is.

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


def _call_ai_model(
    messages: list,
    context_label: str,
    max_tokens: int,
    model: str = AI_MODEL,
    response_format: dict | None = None,
) -> dict:
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "extra_headers": _OPENROUTER_HEADERS,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)

    except Exception as exc:
        error_type = None
        if response_format is not None and _is_structured_output_unsupported_error(exc):
            error_type = "structured_output_unsupported"
            print(
                f"WARNING: Model {model} does not support structured output "
                f"(response_format) for {context_label}. Will retry without it "
                f"if a fallback path is available."
            )
        else:
            print(f"ERROR: OpenRouter call failed for {context_label} (model={model}): {exc}")

        return {
            "status": "error",
            "text": None,
            "finish_reason": None,
            "usage": None,
            "error": str(exc),
            "error_type": error_type,
        }

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
        f"INFO: AI call complete | context={context_label} | model={model} | "
        f"finish_reason={finish_reason} | usage={usage}"
    )

    return {
        "status": "ok",
        "text": text,
        "finish_reason": finish_reason,
        "usage": usage,
        "error": None,
        "error_type": None,
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


def generate_dataset_summary(profile_data: dict, original_filename: str) -> dict:
    user_prompt = f"""

Please analyse the following data profiling results for a dataset called "{original_filename}" and produce a comprehensive data quality report.

--- PROFILING RESULTS ---
{json.dumps(profile_data, indent=2)}
--- END OF PROFILING RESULTS ---

Your report should include:

1. DATASET OVERVIEW
   What does this dataset appear to contain?
   How large is it? What are the key columns?

2. OVERALL QUALITY ASSESSMENT
   One clear sentence summarising the quality of this data.

3. ISSUES FOUND
   For each issue detected, explain:
   - What the issue is in plain English
   - Why it matters (what could go wrong if ignored)
   - What should be done to fix it

4. COLUMN HIGHLIGHTS
   Call out any columns that are particularly interesting,
   problematic, or worth noting.

5. RECOMMENDATION
   Is this data ready to use? If yes, with what caveats?
   If no, what needs to happen before it can be used?

Be specific. Reference actual column names and numbers from the profile.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = _call_ai_model(messages, context_label=f"summary:{original_filename}", max_tokens=8000)

    if not _needs_retry(result):
        return _success_result(result["text"])

    print(f"INFO: Retrying generate_dataset_summary for {original_filename} with fallback prompt")

    fallback_prompt = f"""

Please analyse the following data profiling results for "{original_filename}" and
produce a SHORT data quality summary (3-5 sentences): what the dataset contains,
its overall quality, and the single most important issue if any.

--- PROFILING RESULTS ---
{json.dumps(profile_data, indent=2)}
--- END OF PROFILING RESULTS ---
"""
    fallback_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": fallback_prompt},
    ]

    retry_result = _call_ai_model(
        fallback_messages, context_label=f"summary:{original_filename}:retry", max_tokens=2000
    )

    if not _needs_retry(retry_result):
        return _success_result(retry_result["text"])

    return _failure_result(
        retry_result.get("error") or "Model produced empty, truncated, or degenerate output twice in a row."
    )


_TECHNICAL_CONTEXT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "technical_context",
        "strict": True,
        "schema": TECHNICAL_CONTEXT_JSON_SCHEMA,
    },
}


def _build_technical_context_prompt(profile_data: dict, original_filename: str, brief: bool = False, inline_schema: bool = False) -> str:
    schema_instructions = f"""
You MUST respond with ONLY a single JSON object — no markdown fences, no
commentary before or after — that strictly matches this JSON Schema:

{json.dumps(TECHNICAL_CONTEXT_JSON_SCHEMA, indent=2)}
""" if inline_schema else ""

    if brief:
        return f"""

You are reviewing profiling results for "{original_filename}" to prepare a SHORT
technical brief. Time/budget is limited — only produce the schema table and the
effort rating. Skip risks/warnings and transformation steps detail; keep them minimal.

--- PROFILING RESULTS ---
{json.dumps(profile_data, indent=2)}
--- END OF PROFILING RESULTS ---
{schema_instructions}
Respond with JSON matching this structure:
- suggested_schema: list of {{column_name, detected_type, suggested_sql_type, notes}}
- validation_rules: list of {{column_name, rules}} — can be minimal/empty per column
- transformation_steps: list of {{column_name, transformation}} — can be minimal/empty per column
- risks_and_warnings: list of strings — at most 1-2 brief items, or empty
- estimated_effort: {{level: "Low"|"Medium"|"High", justification}}

Keep every field short. No long prose anywhere.
"""

    return f"""

You are reviewing profiling results for "{original_filename}" to prepare technical
context for a development team about to work with this data.

--- PROFILING RESULTS ---
{json.dumps(profile_data, indent=2)}
--- END OF PROFILING RESULTS ---
{schema_instructions}
Produce a technical brief as JSON with the following structure:

- suggested_schema: list of objects, one per column:
  {{column_name, detected_type, suggested_sql_type, notes}}
  Use notes to flag any columns where the detected type differs from what it
  should be. Keep notes brief (one short phrase).

- validation_rules: list of objects, one per column:
  {{column_name, rules}}
  Cover NOT NULL constraints, length limits, format checks, and foreign key
  candidates where relevant. Combine multiple rules for the same column into
  one string, separated by semicolons.

- transformation_steps: list of objects, one per column that needs one:
  {{column_name, transformation}}
  Be specific but concise — one sentence per transformation.

- risks_and_warnings: list of short strings (not objects).
  What could go wrong during import or integration, and what assumptions are
  being made about this data. Keep each item to one or two sentences.

- estimated_effort: {{level, justification}}
  level is exactly one of "Low", "Medium", or "High".
  justification is one sentence.

Be specific and technical. Keep every string field concise — no multi-sentence
explanations packed into a single field, even if the dataset has many columns.
This output will be used to create development tickets and database migration
scripts, and may later be consumed programmatically, so keep field values
clean and structured rather than narrative.
"""


def _parse_technical_context(raw_text: str) -> TechnicalContext:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    parsed = json.loads(cleaned)
    return TechnicalContext.model_validate(parsed)


def generate_technical_context(profile_data: dict, original_filename: str) -> dict:
    prompt = _build_technical_context_prompt(profile_data, original_filename, brief=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    result = _call_ai_model(
        messages,
        context_label=f"technical_context:{original_filename}",
        max_tokens=32000,
        response_format=_TECHNICAL_CONTEXT_RESPONSE_FORMAT,
    )

    parsed = None
    parse_error = None
    if not _needs_retry(result):
        try:
            parsed = _parse_technical_context(result["text"])
        except (json.JSONDecodeError, ValidationError) as exc:
            parse_error = str(exc)
            print(f"WARNING: Technical context JSON failed validation for {original_filename}: {exc}")

    if parsed is not None:
        return _success_result(parsed.model_dump())

    use_inline_schema_fallback = result.get("error_type") == "structured_output_unsupported"

    if use_inline_schema_fallback:
        print(
            f"INFO: Retrying generate_technical_context for {original_filename} "
            f"WITHOUT response_format (model does not support structured output) "
            f"— falling back to prompt-based JSON instructions"
        )
    else:
        print(f"INFO: Retrying generate_technical_context for {original_filename} with fallback prompt")

    fallback_prompt = _build_technical_context_prompt(
        profile_data, original_filename, brief=True, inline_schema=use_inline_schema_fallback
    )
    fallback_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": fallback_prompt},
    ]

    retry_result = _call_ai_model(
        fallback_messages,
        context_label=f"technical_context:{original_filename}:retry",
        max_tokens=8000,
        response_format=None if use_inline_schema_fallback else _TECHNICAL_CONTEXT_RESPONSE_FORMAT,
    )

    if not _needs_retry(retry_result):
        try:
            retry_parsed = _parse_technical_context(retry_result["text"])
            return _success_result(retry_parsed.model_dump())
        except (json.JSONDecodeError, ValidationError) as exc:
            print(f"WARNING: Retry technical context JSON also failed validation for {original_filename}: {exc}")
            return _failure_result(f"Model output failed schema validation on retry: {exc}")

    return _failure_result(
        retry_result.get("error")
        or parse_error
        or "Model produced empty, truncated, or degenerate output twice in a row."
    )


def answer_dataset_question(
        profile_data: dict,
        original_filename: str,
        question: str,
        conversation_history: list = None
) -> dict:
    context_message = f"""

The user is asking questions about a dataset called "{original_filename}".
Here are the profiling results for full context:

{json.dumps(profile_data, indent=2)}

Answer the user's questions based on this profiling data.
If the answer cannot be determined from the profiling data alone, say so clearly.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_message},
        {
            "role": "assistant",
            "content": "Understood. I have reviewed the profiling results and I am ready to answer questions about this dataset.",
        },
    ]

    if conversation_history:
        for turn in conversation_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": question})

    result = _call_ai_model(
        messages, context_label=f"qa:{original_filename}", max_tokens=3000
    )

    if not _needs_retry(result):
        return _success_result(result["text"])

    print(f"INFO: Retrying answer_dataset_question for {original_filename} with fallback prompt")

    fallback_question = (
        f"{question}\n\n(Please answer as briefly as possible — a few sentences at most.)"
    )
    fallback_messages = messages[:-1] + [{"role": "user", "content": fallback_question}]

    retry_result = _call_ai_model(
        fallback_messages, context_label=f"qa:{original_filename}:retry", max_tokens=1000
    )

    if not _needs_retry(retry_result):
        return _success_result(retry_result["text"])

    return _failure_result(
        retry_result.get("error") or "Model produced empty, truncated, or degenerate output twice in a row."
    )
