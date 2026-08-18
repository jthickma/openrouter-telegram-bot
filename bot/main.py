"""Application entry point for the OpenRouter Telegram bot."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openrouter_helper import OpenRouterHelper
from telegram_bot import OpenRouterTelegramBot
from user_state import UserStateStore


def _as_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() == "true"


def build_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build OpenRouter and Telegram configuration from the environment."""
    openrouter_config: dict[str, Any] = {
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "default_model": os.environ.get("OPENROUTER_DEFAULT_MODEL", "openrouter/auto"),
        "http_referer": os.environ.get("OPENROUTER_HTTP_REFERER", ""),
        "app_title": os.environ.get("OPENROUTER_APP_TITLE", "OpenRouter Telegram Bot"),
        "proxy": os.environ.get("PROXY") or os.environ.get("OPENROUTER_PROXY"),
        "request_timeout": float(os.environ.get("OPENROUTER_REQUEST_TIMEOUT", "180")),
        "pdf_engine": os.environ.get("OPENROUTER_PDF_ENGINE", "cloudflare-ai"),
        "assistant_prompt": os.environ.get("ASSISTANT_PROMPT", "You are a helpful assistant."),
        "max_tokens": int(os.environ.get("MAX_TOKENS", "4096")),
        "max_history_size": int(os.environ.get("MAX_HISTORY_SIZE", "15")),
        "max_conversation_age_minutes": int(os.environ.get("MAX_CONVERSATION_AGE_MINUTES", "180")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.7")),
        "presence_penalty": float(os.environ.get("PRESENCE_PENALTY", "0")),
        "frequency_penalty": float(os.environ.get("FREQUENCY_PENALTY", "0")),
    }
    telegram_config: dict[str, Any] = {
        "token": os.environ["TELEGRAM_BOT_TOKEN"],
        "admin_user_ids": os.environ.get("ADMIN_USER_IDS", "-"),
        "allowed_user_ids": os.environ.get("ALLOWED_TELEGRAM_USER_IDS", "*"),
        "enable_quoting": _as_bool("ENABLE_QUOTING", True),
        "enable_image_generation": _as_bool("ENABLE_IMAGE_GENERATION", True),
        "stream": _as_bool("STREAM", True),
        "show_usage": _as_bool("SHOW_USAGE", True),
        "proxy": os.environ.get("PROXY") or os.environ.get("TELEGRAM_PROXY"),
        "budget_period": os.environ.get("BUDGET_PERIOD", "monthly").lower(),
        "user_budgets": os.environ.get("USER_BUDGETS", os.environ.get("MONTHLY_USER_BUDGETS", "*")),
        "guest_budget": float(
            os.environ.get("GUEST_BUDGET", os.environ.get("MONTHLY_GUEST_BUDGET", "100"))
        ),
        "group_trigger_keyword": os.environ.get("GROUP_TRIGGER_KEYWORD", ""),
        "ignore_group_attachments": _as_bool("IGNORE_GROUP_ATTACHMENTS", True),
        "max_file_size_mb": int(os.environ.get("MAX_FILE_SIZE_MB", "10")),
        "text_file_max_chars": int(os.environ.get("TEXT_FILE_MAX_CHARS", "200000")),
        "system_prompt_max_chars": int(os.environ.get("SYSTEM_PROMPT_MAX_CHARS", "12000")),
        "message_batch_window_seconds": float(
            os.environ.get("MESSAGE_BATCH_WINDOW_SECONDS", "1.25")
        ),
        "settings_path": Path(os.environ.get("USER_SETTINGS_PATH", "user_data/settings.json")),
        "usage_logs_dir": os.environ.get("USAGE_LOGS_DIR", "usage_logs"),
    }
    return openrouter_config, telegram_config


def main() -> None:
    """Configure and run Telegram polling."""
    load_dotenv()
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        raise SystemExit("TELEGRAM_BOT_TOKEN is required")

    openrouter_config, telegram_config = build_config()
    if telegram_config["budget_period"] not in {"daily", "monthly", "all-time"}:
        raise SystemExit("BUDGET_PERIOD must be daily, monthly, or all-time")
    if openrouter_config["pdf_engine"] not in {"cloudflare-ai", "mistral-ocr", "native"}:
        raise SystemExit("OPENROUTER_PDF_ENGINE must be cloudflare-ai, mistral-ocr, or native")
    if telegram_config["system_prompt_max_chars"] <= 0:
        raise SystemExit("SYSTEM_PROMPT_MAX_CHARS must be greater than zero")
    if telegram_config["message_batch_window_seconds"] < 0:
        raise SystemExit("MESSAGE_BATCH_WINDOW_SECONDS cannot be negative")

    state = UserStateStore(
        settings_path=telegram_config["settings_path"],
        default_model=openrouter_config["default_model"],
    )
    helper = OpenRouterHelper(openrouter_config)
    OpenRouterTelegramBot(telegram_config, helper, state).run()


if __name__ == "__main__":
    main()
