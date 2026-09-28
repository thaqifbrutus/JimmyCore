"""
Tests for frontend.py using Streamlit's AppTest framework — this runs the
REAL app script and simulates REAL user interaction (typing into the
search box, clicking buttons), not just checking that functions exist.
Only requests.get/requests.post are mocked, at the exact same boundary
every other test suite in this codebase mocks external calls at.

This is meaningfully stronger verification than "the Python syntax is
valid" — it proves the actual Streamlit widget tree renders correctly and
responds to interaction the way a real user's clicks would.
"""
from unittest.mock import patch, MagicMock

import pytest
from streamlit.testing.v1 import AppTest


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = {"content-type": "application/json"}
    resp.text = str(json_data)
    return resp


def _fake_search_results():
    return {
        "results": [
            {
                "id": "roadaccidents",
                "title_en": "Road Accidents by State",
                "category_en": "Transport",
                "subcategory_en": "Safety",
                "source": "PDRM",
                "frequency": "Yearly",
                "dataset_begin": 2010,
                "dataset_end": 2024,
                "score": 0.87,
            }
        ]
    }


def _fake_analysis_result():
    return {
        "report_id": "abc-123",
        "catalog_dataset_id": "roadaccidents",
        "dataset_stats": {
            "row_count": 500,
            "column_count": 4,
            "null_percentage": 0.0,
            "duplicate_row_count": 0,
        },
        "columns": [],
        "overview": {
            "status": "ok",
            "reason": None,
            "content": {
                "overview": "This dataset tracks road accidents by state.",
                "suggested_questions": [
                    "Which state has the most accidents?",
                    "What's the trend over time?",
                    "Are there missing values?",
                ],
            },
        },
        "source": {"type": "official_government_dataset"},
    }


def _fake_post_dispatcher(analysis=None, ask=None):
    """
    Returns a side_effect function for requests.post that routes /analyze
    and /ask to different fake responses. Required because both endpoints
    are POSTs and a single return_value can't distinguish them.
    """
    def _dispatch(url, **kwargs):
        if "/analyze" in url:
            return _mock_response(200, analysis or _fake_analysis_result())
        if "/ask" in url:
            if ask is None:
                return _mock_response(500, {"detail": "unmocked /ask"})
            return _mock_response(200, ask)
        return _mock_response(404, {"detail": "unhandled POST"})
    return _dispatch


def _has_bar_chart(at):
    """Streamlit AppTest exposes bar charts under one of several names
    depending on version. Check the plausible ones — return True on any."""
    for key in ("bar_chart", "arrow_vega_lite_chart", "vega_lite_chart"):
        try:
            if len(at.get(key)) > 0:
                return True
        except Exception:
            continue
    return False


def _drive_to_detail_view(at, fake_results=None):
    """Search → click Analyze → land on the detail page. Returns the
    AppTest instance, mutated in place."""
    fake_results = fake_results or _fake_search_results()
    with patch("frontend.requests.get", return_value=_mock_response(200, fake_results)):
        at.text_input[0].set_value("road accidents")
        next(b for b in at.button if b.label == "\U0001f50d Search").click()
        at.run(timeout=15)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher()):
        next(b for b in at.button if b.label == "Analyze").click()
        at.run(timeout=15)
    return at


# ── Existing behaviour, unchanged ──────────────────────────────────────────

def test_app_loads_with_search_mode_selected_by_default():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=60)

    assert not at.exception
    assert at.radio[0].value == "Search government data"


def test_switching_to_upload_mode_shows_file_uploader():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    at.radio[0].set_value("Upload a CSV")
    at.run(timeout=15)

    assert not at.exception
    assert len(at.get("file_uploader")) == 1


def test_search_with_no_results_shows_info_message():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    with patch("frontend.requests.get", return_value=_mock_response(200, {"results": []})):
        at.text_input[0].set_value("some very specific query with no matches")
        search_button = next(b for b in at.button if b.label == "\U0001f50d Search")
        search_button.click()
        at.run(timeout=15)

    assert not at.exception
    assert any("No matching datasets found" in i.value for i in at.info)


def test_search_with_results_renders_dataset_cards():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    with patch("frontend.requests.get", return_value=_mock_response(200, _fake_search_results())):
        at.text_input[0].set_value("drunk driving accidents")
        next(b for b in at.button if b.label == "\U0001f50d Search").click()
        at.run(timeout=15)

    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Road Accidents by State" in markdown_text
    assert any(b.label == "Analyze" for b in at.button)


def test_analyze_button_populates_detail_view_with_overview_and_suggested_questions():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    _drive_to_detail_view(at)

    assert not at.exception
    assert at.session_state["report_id"] == "abc-123"
    assert at.session_state["source_label"] == "Road Accidents by State"
    assert at.session_state["source_kind"] == "government"
    assert "Which state has the most accidents?" in at.session_state["suggested_questions"]

    markdown_text = " ".join(m.value for m in at.markdown)
    assert "road accidents by state" in markdown_text.lower()

    button_labels = [b.label for b in at.button]
    assert "Which state has the most accidents?" in button_labels
    assert not any("Technical brief" in m for m in markdown_text)
    assert not any(b.label == "Generate technical brief" for b in at.button)


def test_suggested_question_button_submits_question_to_chat():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    fake_ask = {
        "question": "Which state has the most accidents?",
        "answer": {
            "status": "ok",
            "reason": None,
            "content": "Selangor has the most with 1,234.",
        },
        "report_id": "abc-123",
        "tool_calls_log": [],
    }

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(ask=fake_ask)):
        suggested = next(
            b for b in at.button if b.label == "Which state has the most accidents?"
        )
        suggested.click()
        at.run(timeout=15)

    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Selangor has the most with 1,234." in markdown_text


def test_back_to_search_clears_state():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    _drive_to_detail_view(at)

    assert at.session_state["report_id"] == "abc-123"

    back_button = next(b for b in at.button if "Back to search" in b.label)
    back_button.click()
    at.run(timeout=15)

    assert not at.exception
    assert at.session_state["report_id"] is None
    assert at.session_state["profile_result"] is None
    assert at.session_state["source_label"] is None
    assert at.session_state["suggested_questions"] == []
    assert at.session_state["messages"] == []


def test_search_api_error_shows_error_message():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    with patch("frontend.requests.get", side_effect=__import__("requests").exceptions.ConnectionError("refused")):
        at.text_input[0].set_value("anything")
        next(b for b in at.button if b.label == "\U0001f50d Search").click()
        at.run(timeout=15)

    assert not at.exception
    assert any("Search failed" in e.value for e in at.error)


def test_enter_in_search_submits_the_form():
    """Pressing Enter inside the search box triggers the search."""
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    with patch("frontend.requests.get", return_value=_mock_response(200, _fake_search_results())):
        at.text_input[0].set_value("road accidents")
        at.run(timeout=15)

    assert not at.exception
    assert at.session_state["search_error"] is None
    assert at.session_state["search_results"] == _fake_search_results()["results"]
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Road Accidents by State" in markdown_text


# ── A1: provenance expander ────────────────────────────────────────────────

def test_provenance_expander_renders_for_assistant_message():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    fake_ask = {
        "question": "Which state had the most road accidents?",
        "answer": {
            "status": "ok",
            "reason": None,
            "content": "Selangor had the most, with 1,234 incidents.",
        },
        "report_id": "abc-123",
        "tool_calls_log": [
            {
                "name": "filter_rows",
                "arguments": {
                    "column": "state",
                    "operator": "==",
                    "value": "Selangor",
                },
                "result_summary": "17 rows matched",
                "chart_data": None,
            }
        ],
    }

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(ask=fake_ask)):
        suggested = next(
            b for b in at.button if b.label == "Which state has the most accidents?"
        )
        suggested.click()
        at.run(timeout=15)

    assert not at.exception

    # The expander LABEL is a widget attribute, not markdown content.
    expander_labels = [e.label for e in at.expander]
    assert any(
        "How I got this answer" in label for label in expander_labels
    ), f"expected a 'How I got this answer' expander; got labels: {expander_labels}"

    # Tool call detail is rendered by THREE different Streamlit primitives:
    #   st.markdown  → tool name
    #   st.code      → arguments JSON
    #   st.caption   → result_summary
    # AppTest surfaces each of these as a SEPARATE collection. The previous
    # version of this test checked only at.markdown, which is why the
    # result_summary assertion kept failing even though the UI renders it.
    markdown_text = " ".join(m.value for m in at.markdown)
    caption_text = " ".join(c.value for c in at.caption)
    code_text = " ".join(c.value for c in at.code)

    assert "filter_rows" in markdown_text
    assert "17 rows matched" in caption_text or "17 rows matched" in markdown_text, (
        f"expected result_summary in captions or markdown; "
        f"captions={[c.value for c in at.caption]} "
        f"markdown={[m.value for m in at.markdown]}"
    )


# ── A5: inline bar chart inside the provenance expander ────────────────────

def test_bar_chart_renders_for_value_counts_tool_call():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    fake_ask = {
        "question": "Break down accidents by state",
        "answer": {
            "status": "ok",
            "reason": None,
            "content": "Selangor and Johor lead; here's the full breakdown.",
        },
        "report_id": "abc-123",
        "tool_calls_log": [
            {
                "name": "value_counts",
                "arguments": {"column": "state"},
                "result_summary": "3 values",
                "chart_data": [
                    {"value": "Selangor", "count": 12},
                    {"value": "Johor", "count": 8},
                    {"value": "Penang", "count": 4},
                ],
            }
        ],
    }

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(ask=fake_ask)):
        suggested = next(
            b for b in at.button if b.label == "Which state has the most accidents?"
        )
        suggested.click()
        at.run(timeout=15)

    assert not at.exception
    assert _has_bar_chart(at), "expected a bar chart element in the provenance expander"


# ── A3: quota-exhausted error message ──────────────────────────────────────

def test_quota_exhausted_error_shows_friendly_message():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    fake_ask = {
        "question": "Which state had the most?",
        "answer": {
            "status": "failed",
            "reason": "429 free-models-per-day",
            "error_type": "daily_quota_exhausted",
            "content": None,
            "tool_calls_log": [],
        },
        "report_id": "abc-123",
        "tool_calls_log": [],
    }

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(ask=fake_ask)):
        suggested = next(
            b for b in at.button if b.label == "Which state has the most accidents?"
        )
        suggested.click()
        at.run(timeout=15)

    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "AI quota exhausted for today" in markdown_text