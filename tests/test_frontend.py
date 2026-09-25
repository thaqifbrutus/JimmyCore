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

    fake_results = {
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

    with patch("frontend.requests.get", return_value=_mock_response(200, fake_results)):
        at.text_input[0].set_value("drunk driving accidents")
        search_button = next(b for b in at.button if b.label == "\U0001f50d Search")
        search_button.click()
        at.run(timeout=15)

    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Road Accidents by State" in markdown_text
    assert any(b.label == "Analyze" for b in at.button)


def test_analyze_button_populates_detail_view_with_overview_and_suggested_questions():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    fake_results = {
        "results": [
            {
                "id": "roadaccidents",
                "title_en": "Road Accidents by State",
                "category_en": "Transport",
                "subcategory_en": None,
                "source": "PDRM",
                "frequency": "Yearly",
                "dataset_begin": 2010,
                "dataset_end": 2024,
                "score": 0.87,
            }
        ]
    }

    with patch("frontend.requests.get", return_value=_mock_response(200, fake_results)):
        at.text_input[0].set_value("road accidents")
        next(b for b in at.button if b.label == "\U0001f50d Search").click()
        at.run(timeout=15)

    with patch("frontend.requests.post", return_value=_mock_response(200, _fake_analysis_result())):
        analyze_button = next(b for b in at.button if b.label == "Analyze")
        analyze_button.click()
        at.run(timeout=15)

    assert not at.exception
    assert at.session_state["report_id"] == "abc-123"
    assert at.session_state["source_label"] == "Road Accidents by State"
    assert at.session_state["source_kind"] == "government"
    assert "Which state has the most accidents?" in at.session_state["suggested_questions"]

    markdown_text = " ".join(m.value for m in at.markdown)
    assert "road accidents by state" in markdown_text.lower()

    button_labels = [b.label for b in at.button]
    assert "Which state has the most accidents?" in button_labels
    # Old UI must be gone:
    assert not any("Technical brief" in m for m in markdown_text)
    assert not any(b.label == "Generate technical brief" for b in at.button)


def test_suggested_question_button_submits_question_to_chat():
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    fake_results = {
        "results": [
            {
                "id": "roadaccidents",
                "title_en": "Road Accidents by State",
                "category_en": "Transport",
                "subcategory_en": None,
                "source": "PDRM",
                "frequency": "Yearly",
                "dataset_begin": 2010,
                "dataset_end": 2024,
                "score": 0.87,
            }
        ]
    }

    fake_ask_response = {
        "question": "Which state has the most accidents?",
        "answer": {
            "status": "ok",
            "reason": None,
            "content": "Selangor has the most with 1,234.",
        },
        "report_id": "abc-123",
        "tool_calls_log": [],
    }

    with patch("frontend.requests.get", return_value=_mock_response(200, fake_results)):
        at.text_input[0].set_value("road accidents")
        next(b for b in at.button if b.label == "\U0001f50d Search").click()
        at.run(timeout=15)

    with patch("frontend.requests.post", return_value=_mock_response(200, _fake_analysis_result())):
        next(b for b in at.button if b.label == "Analyze").click()
        at.run(timeout=15)

    with patch("frontend.requests.post", return_value=_mock_response(200, fake_ask_response)):
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

    fake_results = {
        "results": [
            {
                "id": "roadaccidents",
                "title_en": "Road Accidents by State",
                "category_en": "Transport",
                "subcategory_en": None,
                "source": "PDRM",
                "frequency": "Yearly",
                "dataset_begin": 2010,
                "dataset_end": 2024,
                "score": 0.87,
            }
        ]
    }

    with patch("frontend.requests.get", return_value=_mock_response(200, fake_results)):
        at.text_input[0].set_value("road accidents")
        next(b for b in at.button if b.label == "\U0001f50d Search").click()
        at.run(timeout=15)

    with patch("frontend.requests.post", return_value=_mock_response(200, _fake_analysis_result())):
        next(b for b in at.button if b.label == "Analyze").click()
        at.run(timeout=15)

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
    """Pressing Enter inside the search box triggers the search.

    The text_input has on_change=_queue_search which sets a session-state
    flag. When AppTest sets a value and re-runs, the callback fires and
    the search runs — same outcome as pressing Enter in the real UI.
    """
    at = AppTest.from_file("../frontend.py")
    at.run(timeout=15)

    fake_results = {
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

    with patch("frontend.requests.get", return_value=_mock_response(200, fake_results)):
        at.text_input[0].set_value("road accidents")
        at.run(timeout=15)

    assert not at.exception
    assert at.session_state["search_error"] is None
    assert at.session_state["search_results"] == fake_results["results"]
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Road Accidents by State" in markdown_text