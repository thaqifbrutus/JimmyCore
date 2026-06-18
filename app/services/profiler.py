import pandas as pd
import numpy as np
from datetime import datetime


# Threshold for truncating individual string values inside top_values output.
# Long narrative text dumped wholesale into the AI prompt's JSON payload was
# the root cause of the repetition/looping bug — capping length here keeps
# the profile output bounded regardless of what's in the source data.
TOP_VALUE_MAX_LENGTH = 80

# A column is considered "high-cardinality" for sampling-skip purposes when
# its unique-value count is at or above this fraction of total row count.
HIGH_CARDINALITY_RATIO = 0.95

# A column is considered "long-form text" for sampling-skip purposes when
# its average string length (from _get_string_patterns) is at or above this.
LONG_FORM_TEXT_AVG_LENGTH = 50


def profile_dataset(file_path: str) -> dict:
    """
    Reads a CSV file and returns a comprehensive profile of its contents.
    This is the core intelligence of JimmyLens.
    """
    df = pd.read_csv(file_path)

    profile = {
        "profiled_at": datetime.utcnow().isoformat(),
        "overview": _get_overview(df),
        "columns": _get_column_profiles(df),
        "issues": _get_issues(df)
    }

    return profile


def _get_overview(df: pd.DataFrame) -> dict:
    """
    High level summary of the entire dataset.
    """
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = df.isnull().sum().sum()

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_row_count": int(df.duplicated().sum()),
        "total_null_count": int(total_nulls),
        "null_percentage": round((total_nulls / total_cells) * 100, 2) if total_cells > 0 else 0,
        "column_names": list(df.columns)
    }


def _get_column_profiles(df: pd.DataFrame) -> list:
    """
    Per-column breakdown. This is the most detailed section.
    Every column tells its own story.
    """
    column_profiles = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        total = len(series)
        unique_count = int(series.nunique())

        col_profile = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_percentage": round((null_count / total) * 100, 2) if total > 0 else 0,
            "unique_count": unique_count,
        }

        # Numeric columns get extra statistical analysis
        if pd.api.types.is_numeric_dtype(series):
            col_profile["stats"] = _get_numeric_stats(series)
            col_profile["top_values"] = _get_top_values(series)

        # String columns get pattern analysis. Pattern analysis is computed
        # first because the resulting avg_length feeds the high-cardinality
        # long-text check that decides whether top_values sampling runs at
        # all for this column.
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            patterns = _get_string_patterns(series)
            col_profile["patterns"] = patterns

            if _should_skip_value_sampling(series, unique_count, total, patterns):
                col_profile["top_values"] = []
                col_profile["sampling_note"] = (
                    "high cardinality, long-form text — value sampling skipped"
                )
            else:
                col_profile["top_values"] = _get_top_values(series)

        else:
            col_profile["top_values"] = _get_top_values(series)

        column_profiles.append(col_profile)

    return column_profiles


def _should_skip_value_sampling(series: pd.Series, unique_count: int, total: int, patterns: dict) -> bool:
    """
    Decides whether "top values" sampling is meaningless for this column.

    A column qualifies when it is both:
    1. High-cardinality: nearly every value is unique (close to or equal to
       total row count), so there's no real repetition to report.
    2. Long-form text: the average string length (already computed by
       _get_string_patterns) is long enough that dumping raw values into a
       prompt risks bloating the payload with narrative text.

    Reuses fields already computed elsewhere in the profiler — no new
    statistical machinery, per the scope of this fix.
    """
    if total == 0:
        return False

    if not patterns:
        return False

    avg_length = patterns.get("avg_length")
    if avg_length is None:
        return False

    is_high_cardinality = (unique_count / total) >= HIGH_CARDINALITY_RATIO
    is_long_form = avg_length >= LONG_FORM_TEXT_AVG_LENGTH

    return is_high_cardinality and is_long_form


def _get_numeric_stats(series: pd.Series) -> dict:
    """
    Statistical summary for numeric columns.
    These are the signals that tell you if your numbers make sense.
    """
    clean = series.dropna()

    return {
        "min": round(float(clean.min()), 4) if len(clean) > 0 else None,
        "max": round(float(clean.max()), 4) if len(clean) > 0 else None,
        "mean": round(float(clean.mean()), 4) if len(clean) > 0 else None,
        "median": round(float(clean.median()), 4) if len(clean) > 0 else None,
        "std_dev": round(float(clean.std()), 4) if len(clean) > 0 else None,
        "negative_count": int((clean < 0).sum()),
        "zero_count": int((clean == 0).sum())
    }


def _get_string_patterns(series: pd.Series) -> dict:
    """
    Pattern analysis for text columns.
    Detects things like inconsistent casing, whitespace issues,
    and whether a column might actually be something typed as text.
    """
    clean = series.dropna().astype(str)

    if len(clean) == 0:
        return {}

    avg_length = round(float(clean.str.len().mean()), 2)
    has_whitespace_issues = bool((clean != clean.str.strip()).any())
    has_mixed_case = bool(
        clean.str.lower().nunique() < clean.nunique()
    )

    return {
        "avg_length": avg_length,
        "has_whitespace_issues": has_whitespace_issues,
        "has_mixed_case": has_mixed_case,
        "looks_like_email": bool(clean.str.contains(r'^[\w\.-]+@[\w\.-]+\.\w+$', regex=True).any()),
        "looks_like_date": bool(clean.str.contains(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,4}', regex=True).any())
    }


def _get_top_values(series: pd.Series, n: int = 5) -> list:
    """
    Returns the most frequently occurring values in a column.
    Useful for spotting dominant categories or suspicious repetition.

    Individual values are truncated to TOP_VALUE_MAX_LENGTH characters
    before being included — long narrative text values were previously
    dumped wholesale into the AI prompt's JSON payload, which is what
    triggered the repetition loop in the AI layer.
    """
    top = series.value_counts().head(n)

    return [
        {"value": _truncate_value(str(val)), "count": int(count)}
        for val, count in top.items()
    ]


def _truncate_value(value: str, max_length: int = TOP_VALUE_MAX_LENGTH) -> str:
    """
    Truncates a string value to max_length characters, appending a marker
    so it's clear in the output that truncation occurred.
    """
    if len(value) <= max_length:
        return value
    return value[:max_length] + "...(truncated)"


def _get_issues(df: pd.DataFrame) -> list:
    """
    Automatically detects and flags data quality issues.
    This is JimmyLens being proactive — not just describing data,
    but telling you what's wrong with it.
    """
    issues = []

    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({
            "type": "duplicate_rows",
            "severity": "warning",
            "message": f"{dup_count} duplicate rows detected",
            "affected": "entire dataset"
        })

    # Check each column for issues
    for col in df.columns:
        series = df[col]
        null_pct = (series.isnull().sum() / len(series)) * 100

        # Flag high null columns
        if null_pct > 50:
            issues.append({
                "type": "high_nulls",
                "severity": "critical",
                "message": f"Column '{col}' is {null_pct:.1f}% empty",
                "affected": col
            })
        elif null_pct > 20:
            issues.append({
                "type": "moderate_nulls",
                "severity": "warning",
                "message": f"Column '{col}' has {null_pct:.1f}% missing values",
                "affected": col
            })

        elif null_pct > 0:
            issues.append({
                "type": "minimal_nulls",
                "severity": "info",
                "message": f"Column '{col}' has {null_pct:.1f}% missing values",
                "affected": col
            })

        # Flag columns that are entirely unique (potential ID columns stored wrong)
        if series.nunique() == len(series) and len(series) > 10:
            if not pd.api.types.is_numeric_dtype(series):
                issues.append({
                    "type": "all_unique_strings",
                    "severity": "info",
                    "message": f"Column '{col}' has all unique values — may be an identifier column",
                    "affected": col
                })

        # Flag numeric columns with negative values (context dependent)
        if pd.api.types.is_numeric_dtype(series):
            neg_count = (series < 0).sum()
            if neg_count > 0:
                issues.append({
                    "type": "negative_values",
                    "severity": "info",
                    "message": f"Column '{col}' contains {neg_count} negative values",
                    "affected": col
                })

    # Determine overall health status
    severities = [i["severity"] for i in issues]
    if "critical" in severities:
        overall = "critical"
    elif "warning" in severities:
        overall = "needs_attention"
    elif "info" in severities:
        overall = "good_with_notes"
    else:
        overall = "good"

    return issues


def determine_overall_status(issues: list) -> str:
    """
    Converts the issues list into a single status string.
    Used to update the quality_reports table.
    """
    severities = [i["severity"] for i in issues]
    if "critical" in severities:
        return "critical"
    elif "warning" in severities:
        return "needs_attention"
    elif "info" in severities:
        return "good_with_notes"
    return "good"