"""Persistent per-user token and exact OpenRouter cost accounting."""

from __future__ import annotations

import json
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any


class UsageTracker:
    """Track daily tokens, actual billed costs, and per-model usage.

    Older usage files from the original OpenAI bot are loaded without losing
    their accumulated ``current_cost`` totals. New OpenRouter request costs are
    also stored by date so weekly user budgets can be calculated.
    """

    def __init__(
        self,
        user_id: int | str,
        user_name: str,
        logs_dir: str | Path = "usage_logs",
    ) -> None:
        self.user_id = user_id
        self.logs_dir = Path(logs_dir)
        self.user_file = self.logs_dir / f"{user_id}.json"
        self._lock = threading.RLock()
        self.usage = self._load_or_create(user_name)

    @staticmethod
    def _empty_usage(user_name: str) -> dict[str, Any]:
        today = str(date.today())
        return {
            "version": 2,
            "user_name": user_name,
            "current_cost": {
                "day": 0.0,
                "month": 0.0,
                "all_time": 0.0,
                "last_update": today,
            },
            "usage_history": {
                "chat_tokens": {},
                "openrouter_costs": {},
                "models": {},
            },
        }

    def _load_or_create(self, user_name: str) -> dict[str, Any]:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.user_file.is_file():
            return self._empty_usage(user_name)
        try:
            raw = json.loads(self.user_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return self._empty_usage(user_name)
        if not isinstance(raw, dict):
            return self._empty_usage(user_name)

        usage = raw
        usage["version"] = 2
        usage.setdefault("user_name", user_name)
        usage.setdefault(
            "current_cost",
            {"day": 0.0, "month": 0.0, "all_time": 0.0, "last_update": str(date.today())},
        )
        current_cost = usage["current_cost"]
        if not isinstance(current_cost, dict):
            current_cost = {}
            usage["current_cost"] = current_cost
        current_cost.setdefault("day", 0.0)
        current_cost.setdefault("month", 0.0)
        current_cost.setdefault("all_time", 0.0)
        current_cost.setdefault("last_update", str(date.today()))

        history = usage.setdefault("usage_history", {})
        if not isinstance(history, dict):
            history = {}
            usage["usage_history"] = history
        history.setdefault("chat_tokens", {})
        history.setdefault("openrouter_costs", {})
        history.setdefault("models", {})
        return usage

    def _save(self) -> None:
        temporary_path = self.user_file.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(self.usage, separators=(",", ":")), encoding="utf-8")
        temporary_path.replace(self.user_file)

    def _roll_cost_window(self) -> None:
        today = date.today()
        current = self.usage["current_cost"]
        try:
            last_update = date.fromisoformat(str(current["last_update"]))
        except (ValueError, TypeError):
            last_update = today
        if today == last_update:
            return
        current["day"] = 0.0
        if (today.year, today.month) != (last_update.year, last_update.month):
            current["month"] = 0.0
        current["last_update"] = str(today)

    def add_openrouter_usage(self, tokens: int, cost: float, model: str) -> None:
        """Record native token counts and the exact cost returned by OpenRouter."""
        with self._lock:
            today = str(date.today())
            tokens = int(tokens or 0)
            cost = float(cost or 0.0)
            history = self.usage["usage_history"]
            token_history = history["chat_tokens"]
            token_history[today] = int(token_history.get(today, 0)) + tokens
            cost_history = history["openrouter_costs"]
            cost_history[today] = float(cost_history.get(today, 0.0)) + cost

            models_today = history["models"].setdefault(today, {})
            model_usage = models_today.setdefault(
                model or "unknown", {"requests": 0, "tokens": 0, "cost": 0.0}
            )
            model_usage["requests"] = int(model_usage.get("requests", 0)) + 1
            model_usage["tokens"] = int(model_usage.get("tokens", 0)) + tokens
            model_usage["cost"] = float(model_usage.get("cost", 0.0)) + cost

            self._roll_cost_window()
            current = self.usage["current_cost"]
            current["day"] = float(current["day"]) + cost
            current["month"] = float(current["month"]) + cost
            current["all_time"] = float(current["all_time"]) + cost
            self._save()

    def add_chat_tokens(self, tokens: int, tokens_price: float = 0.002) -> None:
        """Compatibility wrapper for callers that only have an estimated price."""
        estimated_cost = int(tokens) * float(tokens_price) / 1000
        self.add_openrouter_usage(int(tokens), estimated_cost, "legacy/estimated")

    def get_current_token_usage(self) -> tuple[int, int]:
        """Return token totals for today and the current calendar month."""
        token_history = self.usage["usage_history"]["chat_tokens"]
        today = str(date.today())
        month = today[:7]
        today_tokens = int(token_history.get(today, 0))
        month_tokens = sum(
            int(tokens)
            for raw_date, tokens in token_history.items()
            if str(raw_date).startswith(month)
        )
        return today_tokens, month_tokens

    def get_current_cost(self) -> dict[str, float]:
        """Return exact current-day, current-month, and all-time USD costs."""
        with self._lock:
            self._roll_cost_window()
            current = self.usage["current_cost"]
            return {
                "cost_today": float(current["day"]),
                "cost_month": float(current["month"]),
                "cost_all_time": float(current["all_time"]),
            }

    def get_cost_for_period(self, period: str) -> float:
        """Return tracked cost for daily, weekly, monthly, or all-time budgets."""
        current = self.get_current_cost()
        if period == "daily":
            return current["cost_today"]
        if period == "monthly":
            return current["cost_month"]
        if period == "all-time":
            return current["cost_all_time"]
        if period == "weekly":
            first_day = date.today() - timedelta(days=6)
            return sum(
                float(cost)
                for raw_date, cost in self.usage["usage_history"]["openrouter_costs"].items()
                if date.fromisoformat(str(raw_date)) >= first_day
            )
        raise ValueError(f"Unknown budget period: {period}")

    def get_model_usage(self) -> dict[str, dict[str, int | float]]:
        """Return today's per-model request, token, and cost totals."""
        models = self.usage["usage_history"]["models"].get(str(date.today()), {})
        return {
            str(model): {
                "requests": int(values.get("requests", 0)),
                "tokens": int(values.get("tokens", 0)),
                "cost": float(values.get("cost", 0.0)),
            }
            for model, values in models.items()
            if isinstance(values, dict)
        }
