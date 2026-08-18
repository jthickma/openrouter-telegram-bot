"""Per-user OpenRouter credentials and non-secret preferences."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VALID_BUDGET_PERIODS = frozenset({"daily", "weekly", "monthly", "all-time"})


@dataclass(slots=True)
class UserPreferences:
    """Preferences that are safe to persist on disk."""

    model: str
    image_model: str | None = None
    budget_limit: float | None = None
    budget_period: str = "monthly"

    @classmethod
    def from_dict(cls, value: dict[str, Any], default_model: str) -> UserPreferences:
        """Build preferences from a potentially older settings record."""
        period = str(value.get("budget_period", "monthly"))
        if period not in VALID_BUDGET_PERIODS:
            period = "monthly"
        raw_limit = value.get("budget_limit")
        limit = float(raw_limit) if raw_limit is not None else None
        return cls(
            model=str(value.get("model") or default_model),
            image_model=value.get("image_model") or None,
            budget_limit=limit,
            budget_period=period,
        )


class UserStateStore:
    """Keep API keys in memory and persist only non-secret user settings.

    Telegram users must provide their key again after a bot restart. This is a
    deliberate security boundary: plaintext OpenRouter keys are never written
    to the filesystem by the bot.
    """

    def __init__(self, settings_path: Path, default_model: str) -> None:
        self.settings_path = settings_path
        self.default_model = default_model
        self._api_keys: dict[int, str] = {}
        self._preferences: dict[int, UserPreferences] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.settings_path.is_file():
            return
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            users = payload.get("users", {}) if isinstance(payload, dict) else {}
            for raw_user_id, raw_preferences in users.items():
                if isinstance(raw_preferences, dict):
                    self._preferences[int(raw_user_id)] = UserPreferences.from_dict(
                        raw_preferences, self.default_model
                    )
        except (OSError, ValueError, TypeError) as exc:
            logging.warning("Could not load user preferences: %s", exc)

    def _save(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "users": {
                str(user_id): asdict(preferences)
                for user_id, preferences in self._preferences.items()
            },
        }
        temporary_path = self.settings_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary_path.replace(self.settings_path)

    def set_api_key(self, user_id: int, api_key: str) -> None:
        """Store a validated API key for the current process only."""
        with self._lock:
            self._api_keys[user_id] = api_key

    def get_api_key(self, user_id: int) -> str | None:
        """Return the in-memory API key for a user."""
        with self._lock:
            return self._api_keys.get(user_id)

    def clear_api_key(self, user_id: int) -> None:
        """Remove a user's key from process memory."""
        with self._lock:
            self._api_keys.pop(user_id, None)

    def preferences_for(self, user_id: int) -> UserPreferences:
        """Return a user's settings, creating defaults when needed."""
        with self._lock:
            if user_id not in self._preferences:
                self._preferences[user_id] = UserPreferences(model=self.default_model)
            return self._preferences[user_id]

    def set_model(self, user_id: int, model: str) -> None:
        """Persist the user's selected chat model."""
        with self._lock:
            self.preferences_for(user_id).model = model
            self._save()

    def set_image_model(self, user_id: int, model: str) -> None:
        """Persist the user's selected image-generation model."""
        with self._lock:
            self.preferences_for(user_id).image_model = model
            self._save()

    def set_budget(self, user_id: int, limit: float, period: str) -> None:
        """Persist a user-selected soft spending cap."""
        if limit <= 0:
            raise ValueError("Budget limit must be greater than zero")
        if period not in VALID_BUDGET_PERIODS:
            raise ValueError(f"Unsupported budget period: {period}")
        with self._lock:
            preferences = self.preferences_for(user_id)
            preferences.budget_limit = limit
            preferences.budget_period = period
            self._save()

    def clear_budget(self, user_id: int) -> None:
        """Disable the user's local soft spending cap."""
        with self._lock:
            self.preferences_for(user_id).budget_limit = None
            self._save()
