"""
Unit tests for app.services.profiler — unchanged from the pre-refactor
version. Running them unmodified against the refactored profiler.py
(profile_dataset now a thin wrapper around profile_dataframe) is the
actual regression check: if these still all pass, the refactor changed
nothing about observable behavior.
"""
import pandas as pd
import pytest

from app.services.profiler import (
    profile_dataset,
    profile_dataframe,
    determine_overall_status,
    _get_top_values,
    _truncate_value,
    _should_skip_value_sampling,
    TOP_VALUE_MAX_LENGTH,
)


def _write_csv(tmp_path, df: pd.DataFrame, name: str = "data.csv") -> str:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_overview_counts_rows_columns_and_nulls(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "email": ["a@x.com", "b@x.com", None, "d@x.com"],
    })
    path = _write_csv(tmp_path, df)

    profile = profile_dataset(path)
    overview = profile["overview"]

    assert overview["row_count"] == 4
    assert overview["column_count"] == 2
    assert overview["total_null_count"] == 1
    assert overview["column_names"] == ["id", "email"]
    assert overview["null_percentage"] == 12.5


def test_overview_detects_duplicate_rows(tmp_path):
    df = pd.DataFrame({"id": [1, 1, 2], "name": ["same", "same", "different"]})
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    assert profile["overview"]["duplicate_row_count"] == 1


def test_numeric_column_stats_and_negative_zero_counts(tmp_path):
    df = pd.DataFrame({"value": [10, -5, 0, 20, -1]})
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    col = next(c for c in profile["columns"] if c["name"] == "value")
    assert col["stats"]["min"] == -5
    assert col["stats"]["max"] == 20
    assert col["stats"]["negative_count"] == 2
    assert col["stats"]["zero_count"] == 1


def test_negative_values_raise_an_info_issue(tmp_path):
    df = pd.DataFrame({"balance": [10, -5, 20]})
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    negative_issues = [i for i in profile["issues"] if i["type"] == "negative_values"]
    assert len(negative_issues) == 1
    assert negative_issues[0]["affected"] == "balance"


def test_string_column_detects_email_pattern(tmp_path):
    df = pd.DataFrame({"contact": ["a@example.com", "b@example.com", "not-an-email"]})
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    col = next(c for c in profile["columns"] if c["name"] == "contact")
    assert col["patterns"]["looks_like_email"] is True


def test_string_column_flags_whitespace_issues(tmp_path):
    df = pd.DataFrame({"name": [" Alice", "Bob ", "Carol"]})
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    col = next(c for c in profile["columns"] if c["name"] == "name")
    assert col["patterns"]["has_whitespace_issues"] is True


def test_high_null_column_flagged_critical(tmp_path):
    df = pd.DataFrame({
        "sparse": [None, None, None, None, None, None, 1, 2, 3, 4],
        "id": range(10),
    })
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    high_null_issues = [i for i in profile["issues"] if i["type"] == "high_nulls"]
    assert len(high_null_issues) == 1
    assert high_null_issues[0]["severity"] == "critical"


def test_all_unique_string_column_flagged_as_likely_identifier(tmp_path):
    df = pd.DataFrame({"code": [f"CODE-{i}" for i in range(15)]})
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    id_issues = [i for i in profile["issues"] if i["type"] == "all_unique_strings"]
    assert len(id_issues) == 1


def test_determine_overall_status_prioritizes_critical_over_warning():
    issues = [{"severity": "info"}, {"severity": "warning"}, {"severity": "critical"}]
    assert determine_overall_status(issues) == "critical"


def test_determine_overall_status_is_good_when_no_issues():
    assert determine_overall_status([]) == "good"


def test_truncate_value_leaves_short_values_untouched():
    short = "hello"
    assert _truncate_value(short) == short


def test_truncate_value_truncates_and_marks_long_values():
    long_value = "x" * (TOP_VALUE_MAX_LENGTH + 50)
    result = _truncate_value(long_value)
    assert len(result) < len(long_value)
    assert result.endswith("...(truncated)")


def test_get_top_values_truncates_long_narrative_text():
    long_text = "This is a very long narrative description. " * 10
    series = pd.Series([long_text, long_text, "short"])
    top = _get_top_values(series, n=5)
    assert all(len(entry["value"]) <= TOP_VALUE_MAX_LENGTH + len("...(truncated)") for entry in top)


def test_high_cardinality_long_text_column_skips_sampling(tmp_path):
    rows = [f"This is a fairly long, mostly-unique narrative note number {i} with detail." for i in range(20)]
    df = pd.DataFrame({"notes": rows})
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    col = next(c for c in profile["columns"] if c["name"] == "notes")
    assert col["top_values"] == []
    assert "sampling_note" in col


def test_low_cardinality_column_does_not_skip_sampling(tmp_path):
    df = pd.DataFrame({"category": ["Enterprise Customer Segment"] * 20})
    path = _write_csv(tmp_path, df)
    profile = profile_dataset(path)
    col = next(c for c in profile["columns"] if c["name"] == "category")
    assert col["top_values"] != []
    assert "sampling_note" not in col


def test_should_skip_value_sampling_requires_both_conditions():
    patterns_short = {"avg_length": 10}
    assert _should_skip_value_sampling(pd.Series(["a"] * 100), 99, 100, patterns_short) is False
    patterns_long = {"avg_length": 80}
    assert _should_skip_value_sampling(pd.Series(["a"] * 100), 2, 100, patterns_long) is False
    assert _should_skip_value_sampling(pd.Series(["a"] * 100), 98, 100, patterns_long) is True


# ---------------------------------------------------------------------------
# New: profile_dataframe used directly (the government-data-fetch flow's
# actual entry point — no CSV file involved at all)
# ---------------------------------------------------------------------------

def test_profile_dataframe_called_directly_matches_profile_dataset(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10, -5, 20]})
    path = _write_csv(tmp_path, df)

    via_file = profile_dataset(path)
    via_dataframe = profile_dataframe(df)

    assert via_file["overview"] == via_dataframe["overview"]
    assert via_file["issues"] == via_dataframe["issues"]


def test_profile_dataframe_works_on_data_that_never_touched_disk():
    # Simulates the gov-data-fetch flow: a DataFrame built directly from
    # an API response, never written to or read from a CSV file.
    df = pd.DataFrame({
        "state": ["Selangor", "Johor", "Selangor"],
        "accidents": [120, 85, 130],
    })

    profile = profile_dataframe(df)

    assert profile["overview"]["row_count"] == 3
    assert profile["overview"]["column_count"] == 2
