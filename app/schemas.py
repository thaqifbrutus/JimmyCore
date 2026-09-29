"""
Pydantic model for the structured dataset-overview output.

Strict JSON-schema mode (OpenAI/OpenRouter response_format with
"strict": true) requires every object to set additionalProperties: false,
AND requires every property to appear in "required" — optional fields
must be expressed as nullable types, not as Python-side defaults, or the
provider rejects the schema outright before the model ever runs.
ConfigDict(extra="forbid") is what produces additionalProperties: false.

The two chart-hint fields (primary_column / primary_metric) are read by
the analyze/profile endpoints to pick an overview chart; see
data_tools.chart_data_for_overview for how they're consumed.
"""
from pydantic import BaseModel, ConfigDict


_STRICT = ConfigDict(extra="forbid")


class DatasetOverview(BaseModel):
    model_config = _STRICT
    overview: str
    suggested_questions: list[str]

    # primary_column: which dimension to chart on the x-axis (state,
    # category, year, etc). null when no column is chart-worthy.
    primary_column: str | None

    # primary_metric: a numeric column to SUM by primary_column, so the
    # chart reads "sum of total_cases by state" rather than "how many
    # times each state name appears". null when the dataset has no
    # natural metric (purely categorical) — the chart then falls back
    # to a row count per category.
    primary_metric: str | None


DATASET_OVERVIEW_JSON_SCHEMA = DatasetOverview.model_json_schema()