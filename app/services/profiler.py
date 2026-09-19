import pandas as pd
import numpy as np
from datetime import datetime


TOP_VALUE_MAX_LENGTH = 80
HIGH_CARDINALITY_RATIO = 0.95
LONG_FORM_TEXT_AVG_LENGTH = 50


def profile_dataset(file_path: str) -> dict:
    """
    Reads a CSV file and returns a comprehensive profile of its contents.
    Thin wrapper around profile_dataframe — kept for the upload flow,
    which has an actual file on disk. The government-catalog flow has no
    file (data arrives as JSON from an API, converted to a DataFrame by
    gov_data_client.py), so it calls profile_dataframe directly instead.
    """
    df = pd.read_csv(file_path)
    return profile_dataframe(df)


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    The actual profiling logic, operating on an in-memory DataFrame
    regardless of where it came from — an uploaded CSV or a fetched
    government dataset. This is the real entry point; profile_dataset()
    above is just a CSV-reading convenience wrapper around it.
    """
    profile = {
        "profiled_at": datetime.utcnow().isoformat(),
        "overview": _get_overview(df),
        "columns": _get_column_profiles(df),
        "issues": _get_issues(df)
    }

    return profile


def _get_overview(df: pd.DataFrame) -> dict:
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

        if pd.api.types.is_numeric_dtype(series):
            col_profile["stats"] = _get_numeric_stats(series)
            col_profile["top_values"] = _get_top_values(series)

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
    top = series.value_counts().head(n)

    return [
        {"value": _truncate_value(str(val)), "count": int(count)}
        for val, count in top.items()
    ]


def _truncate_value(value: str, max_length: int = TOP_VALUE_MAX_LENGTH) -> str:
    if len(value) <= max_length:
        return value
    return value[:max_length] + "...(truncated)"


def _get_issues(df: pd.DataFrame) -> list:
    issues = []

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({
            "type": "duplicate_rows",
            "severity": "warning",
            "message": f"{dup_count} duplicate rows detected",
            "affected": "entire dataset"
        })

    for col in df.columns:
        series = df[col]
        null_pct = (series.isnull().sum() / len(series)) * 100

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

        if series.nunique() == len(series) and len(series) > 10:
            if not pd.api.types.is_numeric_dtype(series):
                issues.append({
                    "type": "all_unique_strings",
                    "severity": "info",
                    "message": f"Column '{col}' has all unique values — may be an identifier column",
                    "affected": col
                })

        if pd.api.types.is_numeric_dtype(series):
            neg_count = (series < 0).sum()
            if neg_count > 0:
                issues.append({
                    "type": "negative_values",
                    "severity": "info",
                    "message": f"Column '{col}' contains {neg_count} negative values",
                    "affected": col
                })

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
    severities = [i["severity"] for i in issues]
    if "critical" in severities:
        return "critical"
    elif "warning" in severities:
        return "needs_attention"
    elif "info" in severities:
        return "good_with_notes"
    return "good"
