"""Smoke tests for Telegram application wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
        "system_prompt_max_chars": 12000,
        "message_batch_window_seconds": 0.01,
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
            "system",
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


def build_test_bot(tmp_path: Path) -> tuple[OpenRouterTelegramBot, OpenRouterHelper]:
    """Build a bot whose OpenRouter transport never reaches the network."""
    helper = OpenRouterHelper(
        {
            "base_url": "https://openrouter.test/api/v1",
            "assistant_prompt": "Default system prompt.",
        },
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})),
    )
    state = UserStateStore(tmp_path / "settings.json", "openrouter/auto")
    return OpenRouterTelegramBot(telegram_config(tmp_path), helper, state), helper


def fake_update(message_id: int) -> SimpleNamespace:
    """Return the effective Telegram fields needed by batching tests."""
    message = SimpleNamespace(message_id=message_id, is_topic_message=False)
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=7),
        effective_chat=SimpleNamespace(id=70, type="private"),
        effective_message=message,
    )


@pytest.mark.asyncio
async def test_split_text_messages_are_batched_into_one_inference(tmp_path: Path) -> None:
    """Rapid Telegram chunks should create one ordered model request."""
    bot, helper = build_test_bot(tmp_path)
    bot._infer = AsyncMock()  # type: ignore[method-assign]
    context = SimpleNamespace(
        application=SimpleNamespace(
            create_task=lambda coroutine, update: asyncio.create_task(coroutine)
        )
    )
    try:
        bot._queue_text_batch(fake_update(12), context, "second", "prompt")
        bot._queue_text_batch(fake_update(11), context, "first", "prompt")
        await asyncio.sleep(0.04)

        bot._infer.assert_awaited_once()
        assert bot._infer.await_args.args[2] == "first\nsecond"
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_text_document_uses_complete_response_path(tmp_path: Path) -> None:
    """Text attachments should not depend on provider streaming compatibility."""
    bot, helper = build_test_bot(tmp_path)
    bot._preflight = AsyncMock(return_value="sk-or-test")  # type: ignore[method-assign]
    bot._infer = AsyncMock()  # type: ignore[method-assign]
    media = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=bytearray(b"hello from a text file"))
    )
    context = SimpleNamespace(bot=SimpleNamespace(get_file=AsyncMock(return_value=media)))
    message = SimpleNamespace(
        document=SimpleNamespace(
            file_name="notes.txt", mime_type="text/plain", file_id="telegram-file"
        ),
        caption=None,
        is_topic_message=False,
    )
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=7),
        effective_chat=SimpleNamespace(id=70, type="private"),
        effective_message=message,
    )
    try:
        await bot.document(update, context)

        bot._infer.assert_awaited_once()
        content = bot._infer.await_args.args[2]
        assert "hello from a text file" in content
        assert bot._infer.await_args.kwargs["force_non_streaming"] is True
    finally:
        await helper.close()
