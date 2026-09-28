"""
Pure tool functions operating on a pandas DataFrame.

This module must have ZERO imports from app.models, app.routers, or db.
Everything is pure: given a DataFrame and arguments, return a
JSON-serializable dict. On error, return {"error": "<message>"} — never
raise — so the model sees the failure and can self-correct.

Each tool that returns rows caps its output and sets truncated: true when
it did. A runaway tool call must not be able to blow the context window.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JSON-serialization helpers — numpy/pandas types leak into everything, and
# the OpenAI tool-calling protocol needs plain JSON.
# ---------------------------------------------------------------------------

def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, (pd.Timestamp, datetime)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {str(k): _to_jsonable(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _column_names(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns]


def _column_error(df: pd.DataFrame, column: str, role: str = "column") -> dict:
    return {
        "error": (
            f"{role} '{column}' not found. Available columns: "
            f"{_column_names(df)}"
        )
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_schema(df: pd.DataFrame) -> dict:
    """Column names, dtypes, null counts, unique counts, plus row/column totals."""
    try:
        columns = []
        for col in df.columns:
            series = df[col]
            columns.append({
                "name": str(col),
                "dtype": str(series.dtype),
                "null_count": int(series.isnull().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            })
        return {
            "columns": columns,
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
        }
    except Exception as e:
        return {"error": str(e)}


def get_sample_rows(df: pd.DataFrame, n: int = 5) -> dict:
    """First N rows. n capped at 20."""
    try:
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 5
        n = max(1, min(n, 20))
        total = int(df.shape[0])
        return {
            "rows": _to_records(df.head(n)),
            "truncated": total > n,
            "total_rows": total,
        }
    except Exception as e:
        return {"error": str(e)}


def describe_column(df: pd.DataFrame, column: str) -> dict:
    """Numeric or string summary of one column."""
    try:
        if column not in df.columns:
            return _column_error(df, column)

        series = df[column]
        result = {
            "column": str(column),
            "dtype": str(series.dtype),
            "count": int(series.notnull().sum()),
            "null_count": int(series.isnull().sum()),
            "null_percentage": round(float(series.isnull().mean() * 100), 2)
                if len(series) > 0 else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                result["min"] = _to_jsonable(clean.min())
                result["max"] = _to_jsonable(clean.max())
                result["mean"] = _to_jsonable(clean.mean())
                result["median"] = _to_jsonable(clean.median())
                result["std_dev"] = _to_jsonable(clean.std())
            else:
                result["min"] = result["max"] = result["mean"] = None
                result["median"] = result["std_dev"] = None
        else:
            clean = series.dropna().astype(str)
            if len(clean) > 0:
                result["avg_length"] = round(float(clean.str.len().mean()), 2)
            else:
                result["avg_length"] = 0.0
            top = series.value_counts(dropna=True).head(10)
            result["top_values"] = [
                {"value": _to_jsonable(v), "count": int(c)}
                for v, c in top.items()
            ]

        return result
    except Exception as e:
        return {"error": str(e)}


def value_counts(df: pd.DataFrame, column: str, top_n: int = 10) -> dict:
    """
    Frequency counts for one column. top_n capped at 50.

    Each value includes a 'share' field — the percentage of non-null rows
    that hold that value, rounded to 2 decimals. This exists so the model
    doesn't have to do the percentage arithmetic itself (and can't
    hallucinate it).
    """
    try:
        if column not in df.columns:
            return _column_error(df, column)
        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            top_n = 10
        top_n = max(1, min(top_n, 50))

        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        vc = series.value_counts(dropna=True)
        truncated = len(vc) > top_n

        total_non_null = int(series.notna().sum())

        values = []
        for v, c in vc.head(top_n).items():
            count = int(c)
            share = round(count / total_non_null * 100, 2) if total_non_null else 0.0
            values.append({
                "value": _to_jsonable(v),
                "count": count,
                "share": share,
            })

        return {
            "column": str(column),
            "values": values,
            "truncated": truncated,
            "unique_count": unique_count,
        }
    except Exception as e:
        return {"error": str(e)}


_VALID_FILTER_OPS = {"==", "!=", ">", "<", ">=", "<=", "contains"}


def filter_rows(df: pd.DataFrame, column: str, operator: str, value: Any, limit: int = 20) -> dict:
    """Filter rows by a single comparison. limit capped at 50."""
    try:
        if column not in df.columns:
            return _column_error(df, column)
        if operator not in _VALID_FILTER_OPS:
            return {
                "error": (
                    f"invalid operator '{operator}'. Must be one of "
                    f"{sorted(_VALID_FILTER_OPS)}"
                )
            }
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 50))

        series = df[column]

        if operator == "contains":
            mask = series.astype(str).str.contains(
                str(value), case=False, na=False, regex=False
            )
        else:
            target = value
            if pd.api.types.is_numeric_dtype(series):
                try:
                    target = float(value)
                except (TypeError, ValueError):
                    return {
                        "error": (
                            f"value {value!r} cannot be compared to numeric "
                            f"column '{column}'"
                        )
                    }

            if operator == "==":
                mask = series == target
            elif operator == "!=":
                # Match SQL NULL semantics: NULL != x is unknown, not true.
                # pandas' != returns True for NaN != x on object columns,
                # so we explicitly exclude nulls to match the model's
                # expectation and the contains branch's na=False.
                mask = (series != target) & series.notna()
            elif operator == ">":
                mask = series > target
            elif operator == "<":
                mask = series < target
            elif operator == ">=":
                mask = series >= target
            else:  # "<="
                mask = series <= target

        matched = int(mask.sum())
        filtered = df[mask].head(limit)
        returned = int(filtered.shape[0])

        result = {
            "rows": _to_records(filtered),
            "matched": matched,
            "returned": returned,
            "truncated": matched > limit,
        }
        if matched > limit:
            # Direct guidance for the model — the raw `truncated: true` flag
            # alone isn't enough. Without this, models routinely re-call
            # filter_rows with a larger limit trying to "see more", which
            # (a) hits the same 50-row cap and (b) burns tool-loop
            # iterations. Telling them to switch to aggregate() closes
            # that failure mode at the source.
            result["guidance"] = (
                f"This response was truncated to {limit} rows (of "
                f"{matched} total matching). If you need a total, a trend "
                f"over time, or a per-group breakdown, DO NOT call "
                f"filter_rows again with a larger limit (the cap is 50). "
                f"Instead call aggregate() with the appropriate group_by "
                f"column and agg_func='count' or 'sum'."
            )
        return result
    except Exception as e:
        return {"error": str(e)}


_VALID_AGG_FUNCS = {"sum", "mean", "count", "min", "max", "median"}
_AGG_MAX_GROUPS = 50


def aggregate(df: pd.DataFrame, group_by: str, agg_column: str, agg_func: str) -> dict:
    """
    Group by one column, aggregate another. Results capped at 50 groups.

    When agg_func == 'count', each result also includes a 'share' field —
    the percentage of the total count, rounded to 2 decimals. Share is
    deliberately omitted for other agg funcs: percentages of a mean or a
    median aren't meaningful.
    """
    try:
        if group_by not in df.columns:
            return _column_error(df, group_by, role="group_by column")
        if agg_column not in df.columns:
            return _column_error(df, agg_column, role="agg_column")
        if agg_func not in _VALID_AGG_FUNCS:
            return {
                "error": (
                    f"invalid agg_func '{agg_func}'. Must be one of "
                    f"{sorted(_VALID_AGG_FUNCS)}"
                )
            }

        grouped = df.groupby(group_by, dropna=False)[agg_column].agg(agg_func)
        try:
            grouped = grouped.sort_values(ascending=False)
        except Exception:
            pass

        truncated = len(grouped) > _AGG_MAX_GROUPS

        include_share = agg_func == "count"
        total = grouped.sum() if include_share else None

        results = []
        for g, v in grouped.head(_AGG_MAX_GROUPS).items():
            entry = {"group": _to_jsonable(g), "value": _to_jsonable(v)}
            if include_share:
                try:
                    entry["share"] = (
                        round(float(v) / float(total) * 100, 2) if total else 0.0
                    )
                except (TypeError, ValueError, ZeroDivisionError):
                    entry["share"] = 0.0
            results.append(entry)

        return {
            "group_by": str(group_by),
            "agg_column": str(agg_column),
            "agg_func": agg_func,
            "results": results,
            "truncated": truncated,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Chart data helpers — separate from the tools above on purpose.
#
# The tools are the model-facing surface (schemas, dedup, iteration budget
# apply). These helpers are UI-facing: they produce a simpler shape the
# frontend can hand straight to st.bar_chart. Keeping the two paths
# separate means a change to one cannot ripple into the other.
#
# The "count" field on each value entry is a historical name — it holds
# the bar's height, which is a row count for kind="count" charts and a
# sum for kind="sum" charts. The frontend keys off "kind" and "metric"
# to label the axis correctly.
# ---------------------------------------------------------------------------

def _column_is_chartable(
    series: pd.Series,
    min_distinct: int = 2,
    max_distinct: int = 50,
) -> bool:
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return False
    n_unique = int(non_null.nunique())
    if n_unique < min_distinct or n_unique > max_distinct:
        return False
    # A fully-unique column is an identifier, not a dimension — a bar
    # chart of one-count bars is noise.
    if n_unique == n:
        return False
    return True


def _auto_pick_chart_column(df: pd.DataFrame) -> str | None:
    """
    Two-pass auto-pick. Pass 1 uses [2, 20] distinct values, pass 2
    relaxes to [2, 50]. Within each pass, string/object columns are
    preferred, then datetime, then numeric. First match wins — columns
    are iterated in their natural DataFrame order within each pass.
    """
    predicates = (
        lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
        lambda s: pd.api.types.is_datetime64_any_dtype(s),
        lambda s: pd.api.types.is_numeric_dtype(s),
    )
    for max_distinct in (20, 50):
        for predicate in predicates:
            for col in df.columns:
                series = df[col]
                try:
                    if not predicate(series):
                        continue
                except Exception:
                    continue
                if _column_is_chartable(series, 2, max_distinct):
                    return col
    return None


def chart_data_for_column(
    df: pd.DataFrame,
    column: str | None = None,
    top_n: int = 15,
) -> dict | None:
    """
    Count-based chart: how often each value of `column` appears.

    Returns a chart-ready breakdown for a single column, or None if no
    suitable column exists. If `column` is provided, validated, and
    chartable, use it. Otherwise fall back to an auto-picked column.

    Return shape:
        {"column": str, "metric": None, "kind": "count",
         "values": [{"value": ..., "count": int}, ...], "truncated": bool}

    Never raises — chart is presentation, and a broken chart must never
    break the page it renders on.
    """
    try:
        selected: str | None = None

        if column is not None and column in df.columns:
            if _column_is_chartable(df[column]):
                selected = column

        if selected is None:
            selected = _auto_pick_chart_column(df)

        if selected is None:
            return None

        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            top_n = 15
        top_n = max(1, top_n)

        series = df[selected]
        counts = series.value_counts(dropna=True)
        truncated = len(counts) > top_n

        values = [
            {"value": _to_jsonable(v), "count": int(c)}
            for v, c in counts.head(top_n).items()
        ]

        return {
            "column": str(selected),
            "metric": None,
            "kind": "count",
            "values": values,
            "truncated": truncated,
        }
    except Exception:
        return None


def chart_data_for_metric(
    df: pd.DataFrame,
    group_column: str,
    metric_column: str,
    top_n: int = 15,
) -> dict | None:
    """
    Sum-based chart: for each distinct value of `group_column`, sum
    `metric_column`. E.g. sum of total_cases by state.

    Returns None if the columns don't exist, the metric isn't numeric,
    or the group column isn't chartable (too few / too many distinct
    values, all-unique identifier, all nulls). Never raises.

    Return shape:
        {"column": str, "metric": str, "kind": "sum",
         "values": [{"value": ..., "count": <sum>}, ...], "truncated": bool}
    """
    try:
        if group_column not in df.columns:
            return None
        if metric_column not in df.columns:
            return None

        group_series = df[group_column]
        metric_series = df[metric_column]

        if not pd.api.types.is_numeric_dtype(metric_series):
            return None

        if not _column_is_chartable(group_series):
            return None

        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            top_n = 15
        top_n = max(1, top_n)

        # dropna=True: a "None" bar on the x-axis is noise, not insight.
        # pandas .sum() skips NaN in the metric by default, so groups
        # with all-null metrics come back as 0.0 rather than NaN.
        grouped = df.groupby(group_column, dropna=True)[metric_column].sum()
        grouped = grouped.sort_values(ascending=False)
        truncated = len(grouped) > top_n

        values = [
            {"value": _to_jsonable(g), "count": _to_jsonable(v)}
            for g, v in grouped.head(top_n).items()
        ]

        return {
            "column": str(group_column),
            "metric": str(metric_column),
            "kind": "sum",
            "values": values,
            "truncated": truncated,
        }
    except Exception:
        return None


def chart_data_for_overview(
    df: pd.DataFrame,
    primary_column: str | None = None,
    primary_metric: str | None = None,
    top_n: int = 15,
) -> dict | None:
    """
    Dispatcher used by the /catalog/{id}/analyze and /reports/datasets/{id}/profile
    endpoints to pick the right chart for a dataset overview.

    Preference order:
      1. Sum of primary_metric by primary_column — if both hints are set,
         both columns exist, and the metric is numeric.
      2. Count of rows per primary_column — the count-based fallback.
      3. Auto-picked column, count of rows — when primary_column is null
         or fails validation.

    Returns None only when nothing in the DataFrame is chartable.
    """
    if primary_column and primary_metric and primary_column != primary_metric:
        metric_chart = chart_data_for_metric(
            df, primary_column, primary_metric, top_n=top_n
        )
        if metric_chart is not None:
            return metric_chart
    return chart_data_for_column(df, column=primary_column, top_n=top_n)


# ---------------------------------------------------------------------------
# Registry + OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "get_schema": get_schema,
    "get_sample_rows": get_sample_rows,
    "describe_column": describe_column,
    "value_counts": value_counts,
    "filter_rows": filter_rows,
    "aggregate": aggregate,
}


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Return the dataset's columns with their dtypes, null counts, "
                "and unique counts, plus total row and column counts. Call "
                "this first if you don't already know the shape of the data."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sample_rows",
            "description": (
                "Return the first N rows of the dataset (max 20, default 5). "
                "Useful for eyeballing column semantics and value formats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to return (1-20). Defaults to 5.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": (
                "Return a numeric or string summary of a single column. "
                "Numeric columns give min/max/mean/median/std_dev; string "
                "columns give avg_length and the top 10 most-frequent values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name."},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "value_counts",
            "description": (
                "Frequency counts for one column. Top N values (max 50, "
                "default 10). Sets truncated: true when there were more "
                "distinct values than returned. Each value includes a share "
                "field (percentage of non-null rows)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "top_n": {"type": "integer", "description": "1-50. Defaults to 10."},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_rows",
            "description": (
                "Return rows matching a single comparison. Results are "
                "capped at 50 (default 20). Do NOT use this to compute "
                "totals, trends, or per-group counts — use aggregate() for "
                "those. When the response includes truncated: true, the "
                "result was capped; increasing the limit will not help."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {
                        "type": "string",
                        "enum": ["==", "!=", ">", "<", ">=", "<=", "contains"],
                        "description": "'contains' is a case-insensitive substring match.",
                    },
                    "value": {
                        "description": "Comparison value. Numbers for numeric ops, strings for 'contains'.",
                    },
                    "limit": {"type": "integer", "description": "1-50. Defaults to 20."},
                },
                "required": ["column", "operator", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate",
            "description": (
                "Group the data by one column and aggregate another. This is "
                "the tool to use for totals, trends over time, per-category "
                "sums, and any 'how many' question with a 'by X' or 'over "
                "time' flavor. Results capped at 50 groups. When agg_func "
                "is 'count', each result includes a share field (percentage "
                "of the total count)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string"},
                    "agg_column": {"type": "string"},
                    "agg_func": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "min", "max", "median"],
                    },
                },
                "required": ["group_by", "agg_column", "agg_func"],
            },
        },
    },
]