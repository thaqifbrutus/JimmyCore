"""
Pydantic models for the structured `generate_technical_context` output.

These exist for two reasons:
1. Validation — if the model's JSON output doesn't conform, that's a
   detectable failure mode that feeds into the retry-once policy rather
   than silently shipping malformed data to the caller.
2. Forward shape — these fields are written with the future agent-
   orchestration consumer in mind (CSV auto-cleaning, autonomous schema
   changes), per the spec. Nothing in this pass *uses* that, but the shape
   should already be programmatically consumable rather than just display-
   friendly.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

# Strict JSON-schema mode (OpenAI/OpenRouter response_format with
# "strict": true) requires every object to set additionalProperties: false,
# AND requires every property to appear in "required" — optional fields
# must be expressed as nullable types, not as Python-side defaults, or the
# provider will reject the schema outright before the model ever runs.
# ConfigDict(extra="forbid") is what produces additionalProperties: false;
# the "notes" field below is typed as `str` (not given a default) so it
# stays in `required`, with prompt instructions telling the model to send
# an empty string when there's nothing to note.
_STRICT = ConfigDict(extra="forbid")


class SchemaColumn(BaseModel):
    model_config = _STRICT
    column_name: str
    detected_type: str
    suggested_sql_type: str
    notes: str = Field(description="Brief flag or note, one short phrase. Empty string if none.")


class ValidationRule(BaseModel):
    model_config = _STRICT
    column_name: str
    rules: str = Field(
        description="Validation rule(s) for this column, semicolon-separated if multiple"
    )


class TransformationStep(BaseModel):
    model_config = _STRICT
    column_name: str
    transformation: str = Field(description="One-sentence description of the transformation needed")


class EstimatedEffort(BaseModel):
    model_config = _STRICT
    level: Literal["Low", "Medium", "High"]
    justification: str


class TechnicalContext(BaseModel):
    model_config = _STRICT
    suggested_schema: list[SchemaColumn]
    validation_rules: list[ValidationRule]
    transformation_steps: list[TransformationStep]
    risks_and_warnings: list[str] = Field(
        description="Short prose strings — this section is inherently written "
                     "explanation, not tabular data. Empty list if none."
    )
    estimated_effort: EstimatedEffort


# JSON schema dict, used for OpenRouter/OpenAI response_format when the
# underlying model supports structured-output mode. Built from the Pydantic
# model so the schema and the validator can never drift out of sync.
TECHNICAL_CONTEXT_JSON_SCHEMA = TechnicalContext.model_json_schema()