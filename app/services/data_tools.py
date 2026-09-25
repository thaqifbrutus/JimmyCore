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
    """Frequency counts for one column. top_n capped at 50."""
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

        return {
            "column": str(column),
            "values": [
                {"value": _to_jsonable(v), "count": int(c)}
                for v, c in vc.head(top_n).items()
            ],
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

        return {
            "rows": _to_records(filtered),
            "matched": matched,
            "returned": returned,
            "truncated": matched > limit,
        }
    except Exception as e:
        return {"error": str(e)}


_VALID_AGG_FUNCS = {"sum", "mean", "count", "min", "max", "median"}
_AGG_MAX_GROUPS = 50


def aggregate(df: pd.DataFrame, group_by: str, agg_column: str, agg_func: str) -> dict:
    """Group by one column, aggregate another. Results capped at 50 groups."""
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
        results = [
            {"group": _to_jsonable(g), "value": _to_jsonable(v)}
            for g, v in grouped.head(_AGG_MAX_GROUPS).items()
        ]

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
                "distinct values than returned."
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
                "Return rows matching a single comparison. Results capped at "
                "50 (default 20) and truncated: true is set when the cap was "
                "hit — tell the user and offer to narrow the query."
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
                "Group the data by one column and aggregate another. Results "
                "capped at 50 groups. Use this for 'highest/lowest/most' "
                "questions and any grouped totals."
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