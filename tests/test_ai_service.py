"""
Tests for the daily-quota short-circuit in ai_service._call_ai_model.

Only the OpenRouter call is mocked. The point of these tests is the
control-flow contract: when OpenRouter reports the account-wide free-tier
daily budget is exhausted, the model chain must stop after one attempt
(no fallback can help — they all share the budget); for any other error,
the chain walks as before.
"""
import pytest

from app.services import ai_service


def test_daily_quota_error_aborts_model_chain(monkeypatch):
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        raise RuntimeError(
            "Error code: 429 - Too Many Requests. "
            "limit_source: openrouter_free_tier_daily; "
            "reason: free-models-per-day exceeded"
        )

    monkeypatch.setattr(ai_service.client.chat.completions, "create", fake_create)
    # Force a non-trivial chain so we can prove we don't walk it.
    monkeypatch.setattr(ai_service, "AI_FALLBACK_MODELS", ["model_b", "model_c"])

    result = ai_service._call_ai_model(
        messages=[{"role": "user", "content": "hi"}],
        context_label="test",
        max_tokens=100,
    )

    # Only the primary model was tried — the chain was aborted.
    assert call_count["n"] == 1
    assert result["status"] == "error"
    assert result["error_type"] == "daily_quota_exhausted"


def test_non_quota_error_walks_the_full_chain(monkeypatch):
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        raise RuntimeError("network error")

    monkeypatch.setattr(ai_service.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(ai_service, "AI_FALLBACK_MODELS", ["model_b", "model_c"])

    result = ai_service._call_ai_model(
        messages=[{"role": "user", "content": "hi"}],
        context_label="test",
        max_tokens=100,
    )

    # primary + 2 fallbacks = 3 attempts — the pre-existing behavior.
    assert call_count["n"] == 3
    assert result["status"] == "error"
    # Not a quota error, so no special error_type.
    assert result["error_type"] is None


def test_quota_error_detection_helper(monkeypatch):
    """
    Direct unit test of the pattern-match helper, since it's load-bearing
    for the short-circuit above.
    """
    assert ai_service._is_daily_quota_error(
        RuntimeError("429 free-models-per-day exceeded")
    )
    assert ai_service._is_daily_quota_error(
        RuntimeError("HTTP 429: openrouter_free_tier_daily")
    )
    # 429 without the free-tier markers — not a daily-quota error.
    assert not ai_service._is_daily_quota_error(RuntimeError("429 too many requests"))
    # Free-tier markers without a 429 — not a daily-quota error.
    assert not ai_service._is_daily_quota_error(RuntimeError("free-models-per-day"))
    assert not ai_service._is_daily_quota_error(RuntimeError("network error"))