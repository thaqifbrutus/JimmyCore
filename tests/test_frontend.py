"""
Tests for frontend.py using Streamlit's AppTest framework — this runs the
REAL app script and simulates REAL user interaction. Only requests.get
and requests.post are mocked, at the same boundary every other test suite
in this codebase mocks at.

Round 1 streaming: /ask/stream is now the endpoint the frontend calls.
Tests that exercise the chat flow construct a fake stream response whose
iter_lines() yields SSE data lines.
"""
import json
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


def _fake_stream_response(events):
    """
    Mimics requests.Response for a streamed request. iter_lines() yields
    the SSE data lines: b"data: <json>" plus a blank b"" between events.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}

    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}".encode("utf-8"))
        lines.append(b"")

    resp.iter_lines.return_value = iter(lines)
    return resp


def _default_stream_events(content="OK."):
    """Minimal stream: one token, one done with empty tool_calls_log."""
    return [
        {"type": "token", "content": content},
        {"type": "done", "content": content, "tool_calls_log": []},
    ]


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


def _fake_analysis_result(overview_chart=None):
    if overview_chart is None:
        overview_chart = {
            "column": "state",
            "metric": None,
            "kind": "count",
            "values": [
                {"value": "Selangor", "count": 12},
                {"value": "Johor", "count": 8},
            ],
            "truncated": False,
        }
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
                "primary_column": "state",
                "primary_metric": None,
            },
            "chart": overview_chart,
        },
        "source": {"type": "official_government_dataset"},
    }


def _fake_post_dispatcher(analysis=None, stream_events=None):
    """
    routes /analyze, /ask/stream, and /reset.  /ask/stream returns a
    fake SSE response; other endpoints return the standard mock response.
    """
    def _dispatch(url, **kwargs):
        if "/analyze" in url:
            return _mock_response(200, analysis or _fake_analysis_result())
        if "/ask/stream" in url:
            return _fake_stream_response(stream_events or _default_stream_events())
        if "/reset" in url:
            return _mock_response(200, {"report_id": "abc-123", "reset": True})
        return _mock_response(404, {"detail": "unhandled POST"})
    return _dispatch


def _has_bar_chart(at):
    for key in ("bar_chart", "arrow_vega_lite_chart", "vega_lite_chart"):
        try:
            if len(at.get(key)) > 0:
                return True
        except Exception:
            continue
    return False


def _drive_to_detail_view(at, fake_results=None, analysis=None):
    fake_results = fake_results or _fake_search_results()
    with patch("frontend.requests.get", return_value=_mock_response(200, fake_results)):
        at.text_input[0].set_value("road accidents")
        next(b for b in at.button if b.label == "\U0001f50d Search").click()
        at.run(timeout=15)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(analysis=analysis)):
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
        at.text_input[0].set_value("nope")
        next(b for b in at.button if b.label == "\U0001f50d Search").click()
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

    assert _has_bar_chart(at)
    caption_text = " ".join(c.value for c in at.caption)
    assert "values of `state`" in caption_text


def test_overview_chart_shows_sum_caption_when_metric_provided():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    sum_chart = {
        "column": "state",
        "metric": "total_cases",
        "kind": "sum",
        "values": [
            {"value": "Johor", "count": 3191},
            {"value": "Kedah", "count": 2975},
        ],
        "truncated": False,
    }
    analysis = _fake_analysis_result(overview_chart=sum_chart)
    analysis["overview"]["content"]["primary_metric"] = "total_cases"

    _drive_to_detail_view(at, analysis=analysis)

    assert not at.exception
    assert _has_bar_chart(at)
    caption_text = " ".join(c.value for c in at.caption)
    assert "`state`" in caption_text
    assert "`total_cases`" in caption_text
    assert "values of" not in caption_text


def test_back_to_search_clears_state():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)
    _drive_to_detail_view(at)
    assert at.session_state["report_id"] == "abc-123"
    next(b for b in at.button if "Back to search" in b.label).click()
    at.run(timeout=15)
    assert not at.exception
    assert at.session_state["report_id"] is None
    assert at.session_state["profile_result"] is None
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
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)
    with patch("frontend.requests.get", return_value=_mock_response(200, _fake_search_results())):
        at.text_input[0].set_value("road accidents")
        at.run(timeout=15)
    assert not at.exception
    assert at.session_state["search_results"] == _fake_search_results()["results"]


# ── Streaming chat tests ───────────────────────────────────────────────────

def test_suggested_question_submits_and_streams_answer():
    """
    Clicking a suggested question streams a response. The assistant
    message that lands in session state should contain the streamed text.
    """
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    stream_events = [
        {"type": "token", "content": "Selangor "},
        {"type": "token", "content": "has 1,234."},
        {"type": "done", "content": "Selangor has 1,234.", "tool_calls_log": []},
    ]

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(stream_events=stream_events)):
        suggested = next(b for b in at.button if b.label == "Which state has the most accidents?")
        suggested.click()
        at.run(timeout=15)

    assert not at.exception
    # Last assistant message should carry the full streamed text.
    assistant_msgs = [m for m in at.session_state["messages"] if m["role"] == "assistant"]
    assert assistant_msgs
    assert "Selangor has 1,234." in assistant_msgs[-1]["content"]


def test_streaming_done_event_attaches_tool_calls():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    stream_events = [
        {"type": "token", "content": "Result."},
        {
            "type": "done",
            "content": "Result.",
            "tool_calls_log": [
                {
                    "name": "filter_rows",
                    "arguments": {"column": "state", "operator": "==", "value": "Selangor"},
                    "result_summary": "17 rows matched",
                    "chart_data": None,
                }
            ],
        },
    ]

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(stream_events=stream_events)):
        next(b for b in at.button if b.label == "Which state has the most accidents?").click()
        at.run(timeout=15)

    assert not at.exception
    assistant_msgs = [m for m in at.session_state["messages"] if m["role"] == "assistant"]
    last = assistant_msgs[-1]
    assert last["tool_calls"]
    assert last["tool_calls"][0]["name"] == "filter_rows"


def test_streaming_error_event_surfaces_error_type():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    stream_events = [
        {"type": "error", "message": "429 free-models-per-day", "error_type": "daily_quota_exhausted"},
    ]

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(stream_events=stream_events)):
        next(b for b in at.button if b.label == "Which state has the most accidents?").click()
        at.run(timeout=15)

    assert not at.exception
    assistant_msgs = [m for m in at.session_state["messages"] if m["role"] == "assistant"]
    assert assistant_msgs
    assert assistant_msgs[-1].get("error_type") == "daily_quota_exhausted"


def test_bar_chart_renders_for_value_counts_tool_call():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    stream_events = [
        {"type": "token", "content": "Here's the breakdown."},
        {
            "type": "done",
            "content": "Here's the breakdown.",
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
        },
    ]

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(stream_events=stream_events)):
        next(b for b in at.button if b.label == "Which state has the most accidents?").click()
        at.run(timeout=15)

    assert not at.exception
    assert _has_bar_chart(at)


def test_provenance_is_hidden_behind_button():
    """
    The provenance popover is not reachable via AppTest's element tree
    in Streamlit 1.58 — we assert on the data layer instead.
    """
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    stream_events = [
        {"type": "token", "content": "Selangor had 1,234."},
        {
            "type": "done",
            "content": "Selangor had 1,234.",
            "tool_calls_log": [
                {
                    "name": "filter_rows",
                    "arguments": {"column": "state", "operator": "==", "value": "Selangor"},
                    "result_summary": "17 rows matched",
                    "chart_data": None,
                }
            ],
        },
    ]

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(stream_events=stream_events)):
        next(b for b in at.button if b.label == "Which state has the most accidents?").click()
        at.run(timeout=15)

    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "How I got this answer" not in markdown_text

    assistant_msgs = [m for m in at.session_state["messages"] if m["role"] == "assistant"]
    assert assistant_msgs[-1].get("tool_calls")
    assert assistant_msgs[-1]["tool_calls"][0]["name"] == "filter_rows"


def test_reset_conversation_clears_messages():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    stream_events = _default_stream_events("Answer.")

    _drive_to_detail_view(at)

    with patch("frontend.requests.post", side_effect=_fake_post_dispatcher(stream_events=stream_events)):
        next(b for b in at.button if b.label == "Which state has the most accidents?").click()
        at.run(timeout=15)

    # Now messages should be non-empty.
    assert at.session_state["messages"]

    with patch("frontend.requests.post", return_value=_mock_response(200, {"reset": True})):
        next(b for b in at.button if b.label == "🗑️ Reset conversation").click()
        at.run(timeout=15)

    assert not at.exception
    assert at.session_state["messages"] == []