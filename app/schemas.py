"""
Pydantic models for structured AI outputs.

Strict JSON-schema mode (OpenAI/OpenRouter response_format with
"strict": true) requires every object to set additionalProperties: false,
AND requires every property to appear in "required" — optional fields
must be expressed as nullable types, not as Python-side defaults, or the
provider rejects the schema outright before the model ever runs.
ConfigDict(extra="forbid") is what produces additionalProperties: false.
"""
from pydantic import BaseModel, ConfigDict


_STRICT = ConfigDict(extra="forbid")


class DatasetOverview(BaseModel):
    model_config = _STRICT
    overview: str
    suggested_questions: list[str]


DATASET_OVERVIEW_JSON_SCHEMA = DatasetOverview.model_json_schema()