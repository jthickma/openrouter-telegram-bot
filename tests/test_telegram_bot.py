"""Smoke tests for Telegram application wiring."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from openrouter_helper import OpenRouterHelper
from telegram.ext import CommandHandler
from telegram_bot import OpenRouterTelegramBot
from user_state import UserStateStore


def telegram_config(tmp_path: Path) -> dict[str, object]:
    """Return the minimum complete Telegram configuration for app construction."""
    return {
        "token": "123456:test-token",
        "admin_user_ids": "-",
        "allowed_user_ids": "*",
        "enable_quoting": True,
        "enable_image_generation": True,
        "stream": True,
        "show_usage": True,
        "proxy": None,
        "budget_period": "monthly",
        "user_budgets": "*",
        "guest_budget": 100.0,
        "group_trigger_keyword": "",
        "ignore_group_attachments": True,
        "max_file_size_mb": 10,
        "text_file_max_chars": 200000,
        "settings_path": tmp_path / "settings.json",
        "usage_logs_dir": tmp_path / "usage",
    }


@pytest.mark.asyncio
async def test_application_registers_openrouter_commands(tmp_path: Path) -> None:
    """All documented user commands should be wired before polling begins."""
    helper = OpenRouterHelper(
        {"base_url": "https://openrouter.test/api/v1"},
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})),
    )
    state = UserStateStore(tmp_path / "settings.json", "openrouter/auto")
    bot = OpenRouterTelegramBot(telegram_config(tmp_path), helper, state)
    try:
        application = bot.build_application()
        command_names = {
            command
            for handlers in application.handlers.values()
            for handler in handlers
            if isinstance(handler, CommandHandler)
            for command in handler.commands
        }
        assert {
            "key",
            "keyinfo",
            "logout",
            "models",
            "model",
            "imagemodels",
            "imagemodel",
            "image",
            "budget",
            "stats",
            "reset",
            "resend",
            "chat",
        } <= command_names
    finally:
        await helper.close()


@pytest.mark.parametrize(
    ("filename", "mime_type", "expected"),
    [
        ("notes.txt", "application/octet-stream", True),
        ("data.bin", "application/json", True),
        ("archive.zip", "application/zip", False),
    ],
)
def test_text_document_detection(filename: str, mime_type: str, expected: bool) -> None:
    """Text/code files should be embedded while opaque binaries use file parts."""
    assert OpenRouterTelegramBot._is_text_document(filename, mime_type) is expected
