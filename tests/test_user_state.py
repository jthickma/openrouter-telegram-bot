"""Tests for non-secret persistence and in-memory credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from user_state import UserStateStore


def test_api_key_is_never_persisted(tmp_path: Path) -> None:
    """Only model and budget preferences should reach the settings file."""
    settings_path = tmp_path / "settings.json"
    state = UserStateStore(settings_path, "openrouter/auto")

    state.set_api_key(42, "sk-or-v1-super-secret")
    state.set_model(42, "anthropic/claude-test")
    state.set_budget(42, 5.0, "monthly")

    payload = settings_path.read_text(encoding="utf-8")
    assert "sk-or-v1-super-secret" not in payload
    parsed = json.loads(payload)
    assert parsed["users"]["42"]["model"] == "anthropic/claude-test"
    assert parsed["users"]["42"]["budget_limit"] == 5.0

    restarted = UserStateStore(settings_path, "openrouter/auto")
    assert restarted.get_api_key(42) is None
    assert restarted.preferences_for(42).model == "anthropic/claude-test"


@pytest.mark.parametrize("limit", [0, -1, -0.01])
def test_budget_rejects_non_positive_limits(tmp_path: Path, limit: float) -> None:
    """A configured cap must be positive."""
    state = UserStateStore(tmp_path / "settings.json", "openrouter/auto")
    with pytest.raises(ValueError, match="greater than zero"):
        state.set_budget(1, limit, "monthly")


def test_budget_rejects_unknown_period(tmp_path: Path) -> None:
    """Only periods supported by the bot are accepted."""
    state = UserStateStore(tmp_path / "settings.json", "openrouter/auto")
    with pytest.raises(ValueError, match="Unsupported budget period"):
        state.set_budget(1, 1, "yearly")


def test_image_model_budget_and_key_can_be_cleared(tmp_path: Path) -> None:
    """Logout and budget-off operations should remove only the intended state."""
    settings_path = tmp_path / "settings.json"
    state = UserStateStore(settings_path, "openrouter/auto")
    state.set_api_key(3, "secret")
    state.set_image_model(3, "vendor/image")
    state.set_budget(3, 2, "daily")

    state.clear_api_key(3)
    state.clear_budget(3)

    assert state.get_api_key(3) is None
    preferences = state.preferences_for(3)
    assert preferences.image_model == "vendor/image"
    assert preferences.budget_limit is None


def test_corrupt_settings_fall_back_to_defaults(tmp_path: Path) -> None:
    """A broken settings file should not prevent startup."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("not-json", encoding="utf-8")
    state = UserStateStore(settings_path, "openrouter/auto")
    assert state.preferences_for(9).model == "openrouter/auto"


def test_system_prompts_are_persisted_per_user_chat_and_topic(tmp_path: Path) -> None:
    """Custom instructions should be isolated by conversation session."""
    settings_path = tmp_path / "settings.json"
    state = UserStateStore(settings_path, "openrouter/auto")
    state.set_system_prompt(4, "100:0", "Be concise.")
    state.set_system_prompt(4, "100:8", "Explain everything in detail.")

    restarted = UserStateStore(settings_path, "openrouter/auto")
    assert restarted.system_prompt_for(4, "100:0") == "Be concise."
    assert restarted.system_prompt_for(4, "100:8") == "Explain everything in detail."

    restarted.clear_system_prompt(4, "100:0")
    assert restarted.system_prompt_for(4, "100:0") is None
    assert restarted.system_prompt_for(4, "100:8") == "Explain everything in detail."


def test_system_prompt_rejects_empty_text(tmp_path: Path) -> None:
    state = UserStateStore(tmp_path / "settings.json", "openrouter/auto")
    with pytest.raises(ValueError, match="cannot be empty"):
        state.set_system_prompt(4, "100:0", "   ")
