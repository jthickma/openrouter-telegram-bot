"""Tests for exact OpenRouter cost tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from usage_tracker import UsageTracker


def test_openrouter_usage_records_exact_cost_and_model(tmp_path: Path) -> None:
    """Usage cost must come from OpenRouter, not static token-price math."""
    tracker = UsageTracker(7, "tester", logs_dir=str(tmp_path))

    tracker.add_openrouter_usage(321, 0.012345, "vendor/model")

    assert tracker.get_current_token_usage() == (321, 321)
    assert tracker.get_current_cost()["cost_today"] == pytest.approx(0.012345)
    assert tracker.get_cost_for_period("weekly") == pytest.approx(0.012345)
    assert tracker.get_model_usage()["vendor/model"] == {
        "requests": 1,
        "tokens": 321,
        "cost": pytest.approx(0.012345),
    }
    payload = json.loads((tmp_path / "7.json").read_text(encoding="utf-8"))
    assert payload["usage_history"]["openrouter_costs"]


def test_unknown_budget_period_is_rejected(tmp_path: Path) -> None:
    """Invalid periods should fail rather than silently bypass a budget."""
    tracker = UsageTracker(8, "tester", logs_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Unknown budget period"):
        tracker.get_cost_for_period("yearly")


def test_legacy_usage_file_is_migrated_and_preserved(tmp_path: Path) -> None:
    """Existing all-time costs and tokens should survive the schema migration."""
    legacy = {
        "user_name": "old-user",
        "current_cost": {
            "day": 1.0,
            "month": 2.0,
            "all_time": 3.0,
            "last_update": "2000-01-01",
        },
        "usage_history": {"chat_tokens": {"2000-01-01": 10}},
    }
    (tmp_path / "10.json").write_text(json.dumps(legacy), encoding="utf-8")

    tracker = UsageTracker(10, "new-name", logs_dir=str(tmp_path))

    assert tracker.get_current_cost() == {
        "cost_today": 0.0,
        "cost_month": 0.0,
        "cost_all_time": 3.0,
    }
    tracker.add_chat_tokens(1000, tokens_price=0.01)
    assert tracker.get_current_cost()["cost_all_time"] == pytest.approx(3.01)
    assert tracker.get_model_usage()["legacy/estimated"]["requests"] == 1


def test_invalid_usage_file_starts_clean(tmp_path: Path) -> None:
    """Corrupt usage data should be replaced by a usable empty structure."""
    (tmp_path / "11.json").write_text("[]", encoding="utf-8")
    tracker = UsageTracker(11, "tester", logs_dir=str(tmp_path))
    assert tracker.get_current_token_usage() == (0, 0)
    assert tracker.get_cost_for_period("daily") == 0
