import json

import numpy as np
import pandas as pd
import pytest

from app.services import data_tools


@pytest.fixture()
def df():
    return pd.DataFrame({
        "state": ["Selangor", "Johor", "Selangor", "Penang", "Johor", None],
        "accidents": [120, 85, 130, 60, 90, 15],
        "rate": [1.5, 2.0, np.nan, 3.1, 2.2, 1.0],
        "date": pd.to_datetime([
            "2023-01-01", "2023-02-01", "2023-03-01",
            "2023-04-01", "2023-05-01", "2023-06-01",
        ]),
    })


# ── get_schema ──────────────────────────────────────────────────────────────

def test_get_schema_returns_column_metadata(df):
    result = data_tools.get_schema(df)
    assert result["row_count"] == 6
    assert result["column_count"] == 4
    assert [c["name"] for c in result["columns"]] == ["state", "accidents", "rate", "date"]
    state_col = next(c for c in result["columns"] if c["name"] == "state")
    assert state_col["null_count"] == 1


def test_get_schema_is_json_serializable(df):
    json.dumps(data_tools.get_schema(df))


# ── get_sample_rows ─────────────────────────────────────────────────────────

def test_get_sample_rows_caps_at_20():
    big = pd.DataFrame({"x": list(range(100))})
    result = data_tools.get_sample_rows(big, n=50)
    assert len(result["rows"]) == 20
    assert result["truncated"] is True
    assert result["total_rows"] == 100


def test_get_sample_rows_all_when_n_exceeds_total():
    small = pd.DataFrame({"x": [1, 2]})
    result = data_tools.get_sample_rows(small, n=5)
    assert len(result["rows"]) == 2
    assert result["truncated"] is False


def test_get_sample_rows_nan_becomes_none(df):
    result = data_tools.get_sample_rows(df, n=20)
    assert any(row["rate"] is None for row in result["rows"])


# ── describe_column ─────────────────────────────────────────────────────────

def test_describe_numeric_column(df):
    result = data_tools.describe_column(df, "accidents")
    assert result["min"] == 15
    assert result["max"] == 130
    assert result["null_count"] == 0
    assert result["mean"] == pytest.approx(83.3333, abs=0.001)


def test_describe_string_column_has_top_values_and_avg_length(df):
    result = data_tools.describe_column(df, "state")
    assert result["null_count"] == 1
    assert "top_values" in result
    assert "avg_length" in result


def test_describe_missing_column_returns_error(df):
    result = data_tools.describe_column(df, "nonexistent")
    assert "error" in result


# ── value_counts ────────────────────────────────────────────────────────────

def test_value_counts_groups_correctly(df):
    result = data_tools.value_counts(df, "state", top_n=10)
    assert result["unique_count"] == 3
    assert result["truncated"] is False
    values = {v["value"]: v for v in result["values"]}
    assert values["Selangor"]["count"] == 2
    assert values["Johor"]["count"] == 2
    # Fixture has 5 non-null state rows; Selangor and Johor each appear
    # twice → 2/5 * 100 = 40.0. Penang once → 20.0.
    assert values["Selangor"]["share"] == 40.0
    assert values["Johor"]["share"] == 40.0
    assert values["Penang"]["share"] == 20.0


def test_value_counts_includes_share(df):
    """A4: every entry carries a share field (percentage of non-null rows)."""
    result = data_tools.value_counts(df, "state", top_n=10)
    for entry in result["values"]:
        assert "share" in entry
        assert isinstance(entry["share"], float)
    # Shares of non-null rows should sum to ~100 (float rounding aside).
    total_share = sum(e["share"] for e in result["values"])
    assert total_share == pytest.approx(100.0, abs=0.1)


def test_value_counts_truncation_flag():
    big = pd.DataFrame({"x": list(range(100))})
    result = data_tools.value_counts(big, "x", top_n=10)
    assert result["truncated"] is True
    assert len(result["values"]) == 10


def test_value_counts_top_n_capped_at_50(df):
    result = data_tools.value_counts(df, "state", top_n=500)
    # Cap is 50 but only 3 unique — all 3 returned.
    assert len(result["values"]) == 3


def test_value_counts_missing_column(df):
    assert "error" in data_tools.value_counts(df, "nonexistent")


# ── filter_rows ─────────────────────────────────────────────────────────────

def test_filter_rows_equality(df):
    result = data_tools.filter_rows(df, "state", "==", "Selangor")
    assert result["matched"] == 2
    assert result["returned"] == 2
    assert result["truncated"] is False


def test_filter_rows_numeric_greater_than(df):
    result = data_tools.filter_rows(df, "accidents", ">", 100)
    assert result["matched"] == 2


def test_filter_rows_numeric_less_than_or_equal(df):
    result = data_tools.filter_rows(df, "accidents", "<=", 85)
    assert result["matched"] == 3


def test_filter_rows_not_equal(df):
    # Johor x2 and Penang x1 — the null row is excluded (SQL semantics).
    result = data_tools.filter_rows(df, "state", "!=", "Selangor")
    assert result["matched"] == 3


def test_filter_rows_contains_case_insensitive(df):
    # "JO" matches Johor twice (case-insensitive); Selangor does not
    # contain "jo", so it should not match.
    result = data_tools.filter_rows(df, "state", "contains", "JO")
    assert result["matched"] == 2


def test_filter_rows_truncation():
    big = pd.DataFrame({"x": list(range(100))})
    result = data_tools.filter_rows(big, "x", ">=", 0, limit=10)
    assert result["matched"] == 100
    assert result["returned"] == 10
    assert result["truncated"] is True


def test_filter_rows_invalid_operator(df):
    result = data_tools.filter_rows(df, "state", "like", "Sel")
    assert "error" in result


def test_filter_rows_missing_column(df):
    assert "error" in data_tools.filter_rows(df, "nonexistent", "==", "x")


# ── aggregate ───────────────────────────────────────────────────────────────

def test_aggregate_sum(df):
    result = data_tools.aggregate(df, "state", "accidents", "sum")
    assert result["agg_func"] == "sum"
    values = {r["group"]: r["value"] for r in result["results"]}
    assert values["Selangor"] == 250
    assert values["Johor"] == 175


def test_aggregate_mean(df):
    values = {
        r["group"]: r["value"]
        for r in data_tools.aggregate(df, "state", "accidents", "mean")["results"]
    }
    assert values["Selangor"] == 125


def test_aggregate_count(df):
    values = {
        r["group"]: r["value"]
        for r in data_tools.aggregate(df, "state", "accidents", "count")["results"]
    }
    assert values["Selangor"] == 2


def test_aggregate_count_includes_share(df):
    """A4: agg_func='count' results carry share (percentage of total count)."""
    result = data_tools.aggregate(df, "state", "accidents", "count")
    by_group = {r["group"]: r for r in result["results"]}
    # 6 total rows (dropna=False: the None state is its own group with
    # count 1). Selangor has count 2 → 2/6 * 100 = 33.33.
    assert by_group["Selangor"]["value"] == 2
    assert by_group["Selangor"]["share"] == 33.33
    for entry in result["results"]:
        assert "share" in entry


def test_aggregate_sum_does_not_include_share(df):
    """A4 regression guard: share is only meaningful for count."""
    result = data_tools.aggregate(df, "state", "accidents", "sum")
    for entry in result["results"]:
        assert "share" not in entry


def test_aggregate_min_max_median_run(df):
    for fn in ("min", "max", "median"):
        result = data_tools.aggregate(df, "state", "accidents", fn)
        assert "error" not in result
        for entry in result["results"]:
            assert "share" not in entry


def test_aggregate_invalid_func(df):
    assert "error" in data_tools.aggregate(df, "state", "accidents", "stddev")


def test_aggregate_missing_group_by(df):
    assert "error" in data_tools.aggregate(df, "nope", "accidents", "sum")


# ── registry / schema sanity ────────────────────────────────────────────────

def test_registry_and_schemas_agree():
    schema_names = {s["function"]["name"] for s in data_tools.TOOL_SCHEMAS}
    assert schema_names == set(data_tools.TOOL_REGISTRY.keys())
    assert schema_names == {
        "get_schema", "get_sample_rows", "describe_column",
        "value_counts", "filter_rows", "aggregate",
    }


def test_schemas_have_valid_shape():
    for s in data_tools.TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        assert fn["parameters"]["type"] == "object"