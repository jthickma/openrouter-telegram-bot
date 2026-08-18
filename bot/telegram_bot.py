"""Telegram handlers for per-user OpenRouter inference."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openrouter_helper import (
    ImageGenerationResult,
    ModelInfo,
    OpenRouterError,
    OpenRouterHelper,
    StreamUpdate,
    UsageInfo,
    file_content,
    image_content,
)
from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    constants,
)
from telegram.error import BadRequest, RetryAfter, TelegramError, TimedOut
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from usage_tracker import UsageTracker
from user_state import VALID_BUDGET_PERIODS, UserStateStore
from utils import (
    get_reply_to_message_id,
    get_thread_id,
    is_admin,
    is_allowed,
    is_group_chat,
    message_text,
    split_into_chunks,
)

TEXT_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/ld+json",
        "application/javascript",
        "application/sql",
        "application/toml",
        "application/xml",
        "application/x-httpd-php",
        "application/x-sh",
        "application/x-yaml",
    }
)
TEXT_EXTENSIONS = frozenset(
    {
        ".c",
        ".cfg",
        ".conf",
        ".cpp",
        ".cs",
        ".css",
        ".csv",
        ".env",
        ".go",
        ".h",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".log",
        ".lua",
        ".md",
        ".php",
        ".properties",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)


@dataclass(slots=True)
class CatalogView:
    """The last model catalog shown to a Telegram user."""

    kind: str
    models: list[ModelInfo]
    query: str
    input_modality: str | None = None


@dataclass(slots=True)
class PendingTextBatch:
    """Consecutive Telegram text chunks waiting to become one model request."""

    update: Update
    context: ContextTypes.DEFAULT_TYPE
    parts: list[tuple[int, str]]
    task: asyncio.Task[None] | None = None


class OpenRouterTelegramBot:
    """Telegram bot backed by each user's own OpenRouter API key."""

    MODEL_PAGE_SIZE = 8

    def __init__(
        self,
        config: dict[str, Any],
        openrouter: OpenRouterHelper,
        state: UserStateStore,
    ) -> None:
        self.config = config
        self.openrouter = openrouter
        self.state = state
        self.usage: dict[int | str, UsageTracker] = {}
        self.last_message: dict[tuple[int, str], str] = {}
        self.catalog_views: dict[tuple[int, str], CatalogView] = {}
        self.pending_text_batches: dict[tuple[int, str, str], PendingTextBatch] = {}
        self.awaiting_system_prompts: set[tuple[int, str]] = set()
        self.commands = [
            BotCommand("help", "Show commands and setup instructions"),
            BotCommand("key", "Authenticate with your OpenRouter API key"),
            BotCommand("keyinfo", "Show OpenRouter key usage and limits"),
            BotCommand("logout", "Forget your in-memory API key"),
            BotCommand("models", "Browse live OpenRouter chat models"),
            BotCommand("model", "Show or select a chat model"),
            BotCommand("system", "Set the system prompt for this chat"),
            BotCommand("imagemodels", "Browse live image-generation models"),
            BotCommand("imagemodel", "Show or select an image model"),
            BotCommand("image", "Generate an image with the selected image model"),
            BotCommand("budget", "Set or inspect a local spending cap"),
            BotCommand("stats", "Show exact OpenRouter usage and cost"),
            BotCommand("reset", "Reset this conversation"),
            BotCommand("resend", "Repeat your last text prompt"),
        ]
        self.group_commands = [BotCommand("chat", "Ask the bot in this group")] + self.commands

    @staticmethod
    def _user_id(update: Update) -> int:
        if update.effective_user is None:
            raise ValueError("Telegram update has no user")
        return update.effective_user.id

    @staticmethod
    def _session_id(update: Update) -> str:
        if update.effective_chat is None:
            raise ValueError("Telegram update has no chat")
        return f"{update.effective_chat.id}:{get_thread_id(update) or 0}"

    def _tracker(self, update: Update) -> UsageTracker:
        user_id = self._user_id(update)
        if user_id not in self.usage:
            name = update.effective_user.name if update.effective_user else str(user_id)
            self.usage[user_id] = UsageTracker(
                user_id, name, logs_dir=str(self.config["usage_logs_dir"])
            )
        return self.usage[user_id]

    def _guest_tracker(self) -> UsageTracker:
        if "guests" not in self.usage:
            self.usage["guests"] = UsageTracker(
                "guests",
                "all guest users in group chats",
                logs_dir=str(self.config["usage_logs_dir"]),
            )
        return self.usage["guests"]

    def _is_guest(self, user_id: int) -> bool:
        allowed = str(self.config["allowed_user_ids"])
        return allowed != "*" and str(user_id) not in allowed.split(",")

    def _server_budget(self, user_id: int) -> float:
        if is_admin(self.config, user_id) or self.config["user_budgets"] == "*":
            return math.inf
        budgets = str(self.config["user_budgets"]).split(",")
        allowed = str(self.config["allowed_user_ids"])
        if allowed == "*":
            return float(budgets[0])
        allowed_ids = allowed.split(",")
        if str(user_id) in allowed_ids:
            index = allowed_ids.index(str(user_id))
            return float(budgets[index]) if index < len(budgets) else 0.0
        return float(self.config["guest_budget"])

    def _remaining_budgets(self, update: Update) -> tuple[float, float]:
        user_id = self._user_id(update)
        tracker = self._tracker(update)
        server_limit = self._server_budget(user_id)
        if self._is_guest(user_id):
            server_spend = self._guest_tracker().get_cost_for_period(self.config["budget_period"])
        else:
            server_spend = tracker.get_cost_for_period(self.config["budget_period"])
        server_remaining = server_limit - server_spend

        preferences = self.state.preferences_for(user_id)
        if preferences.budget_limit is None:
            personal_remaining = math.inf
        else:
            personal_remaining = preferences.budget_limit - tracker.get_cost_for_period(
                preferences.budget_period
            )
        return server_remaining, personal_remaining

    async def _preflight(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        require_key: bool = True,
        enforce_budget: bool = True,
    ) -> str | None:
        if not await is_allowed(self.config, update, context):
            await update.effective_message.reply_text(
                "You are not allowed to use this bot.",
                message_thread_id=get_thread_id(update),
            )
            return None
        if enforce_budget:
            server_remaining, personal_remaining = self._remaining_budgets(update)
            if server_remaining <= 0 or personal_remaining <= 0:
                await update.effective_message.reply_text(
                    "Your configured spending budget has been reached. Use /budget to inspect it.",
                    message_thread_id=get_thread_id(update),
                )
                return None
        if not require_key:
            return ""
        api_key = self.state.get_api_key(self._user_id(update))
        if api_key is None:
            await update.effective_message.reply_text(
                "Authenticate first in a private chat with /key YOUR_OPENROUTER_API_KEY. "
                "The key is kept only in memory and must be re-entered after a restart.",
                message_thread_id=get_thread_id(update),
            )
            return None
        return api_key

    def _record_usage(self, update: Update, usage: UsageInfo, model: str) -> None:
        user_id = self._user_id(update)
        self._tracker(update).add_openrouter_usage(usage.total_tokens, usage.cost, model)
        if self._is_guest(user_id):
            self._guest_tracker().add_openrouter_usage(usage.total_tokens, usage.cost, model)

    def _usage_footer(self, usage: UsageInfo, model: str) -> str:
        if not self.config["show_usage"]:
            return ""
        return (
            f"\n\n—\n{model or 'OpenRouter'} • {usage.total_tokens:,} tokens • "
            f"${usage.cost:.6f}"
        )

    async def _reply_text(self, update: Update, text: str) -> None:
        chunks = split_into_chunks(text or "(empty response)")
        for index, chunk in enumerate(chunks):
            await update.effective_message.reply_text(
                chunk,
                message_thread_id=get_thread_id(update),
                reply_to_message_id=(
                    get_reply_to_message_id(self.config, update) if index == 0 else None
                ),
                disable_web_page_preview=True,
            )

    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Show user workflow and commands."""
        text = (
            "OpenRouter Telegram Bot\n\n"
            "1. In this private chat, send /key followed by your OpenRouter API key.\n"
            "2. Use /models and /model provider/model to choose any current chat model.\n"
            "3. Use /system to set a system prompt for this chat or topic.\n"
            "4. Send text, a supported image, a PDF, a text/code file, or another file "
            "supported natively by the selected model.\n"
            "5. Use /imagemodels, /imagemodel, and /image for OpenRouter image generation.\n\n"
            "Budget and privacy:\n"
            "• /budget 5 monthly sets a local $5 soft cap; /budget off disables it.\n"
            "• /stats combines exact response costs with your OpenRouter key limit.\n"
            "• Keys stay in process memory only and are lost on restart. /logout forgets yours.\n"
            "• Set keys only in private chat. The bot tries to delete the /key message immediately.\n\n"
            "Model filters: /models image finds vision models; /models file finds native "
            "file-capable models; /models image claude also searches by name. PDFs work "
            "with any text model through OpenRouter's file parser. Consecutive text chunks "
            "sent within the batching window are combined into one request."
        )
        await self._reply_text(update, text)

    async def set_key(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Validate and keep a user's OpenRouter API key only in memory."""
        if await self._preflight(update, context, require_key=False, enforce_budget=False) != "":
            return
        if (
            update.effective_chat is None
            or update.effective_chat.type != constants.ChatType.PRIVATE
        ):
            await update.effective_message.reply_text(
                "For safety, API keys can only be entered in a private chat with this bot."
            )
            return
        if not context.args:
            await update.effective_message.reply_text(
                "Usage: /key sk-or-v1-...\nThe message will be deleted when Telegram permits it, "
                "and the key will not be saved to disk."
            )
            return

        api_key = "".join(context.args).strip()
        deleted = False
        try:
            await update.effective_message.delete()
            deleted = True
        except TelegramError:
            logging.warning("Telegram would not delete an API-key message")

        try:
            info = await self.openrouter.validate_api_key(api_key)
        except OpenRouterError as exc:
            warning = "" if deleted else " Delete your key message manually."
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"OpenRouter rejected that key: {exc}.{warning}",
            )
            return

        user_id = self._user_id(update)
        self.state.set_api_key(user_id, api_key)
        label = info.get("label") or "validated key"
        remaining = info.get("limit_remaining")
        remaining_text = (
            f" Remaining key limit: ${float(remaining):.4f}." if remaining is not None else ""
        )
        warning = (
            "" if deleted else " Telegram could not delete the key message; delete it manually."
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=(
                f"Authenticated with {label}.{remaining_text} The key is held only in memory."
                f"{warning} Use /models next."
            ),
        )

    async def logout(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forget the user's in-memory key and chat histories."""
        if await self._preflight(update, context, require_key=False, enforce_budget=False) != "":
            return
        user_id = self._user_id(update)
        self.state.clear_api_key(user_id)
        self.openrouter.reset_user_history(user_id)
        await update.effective_message.reply_text("Your in-memory API key has been forgotten.")

    async def key_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show current OpenRouter API-key usage and limits."""
        api_key = await self._preflight(update, context, enforce_budget=False)
        if api_key is None:
            return
        try:
            info = await self.openrouter.get_key_info(api_key)
        except OpenRouterError as exc:
            await self._reply_text(update, str(exc))
            return
        lines = [
            f"Key: {info.get('label') or 'OpenRouter key'}",
            f"Usage: ${float(info.get('usage') or 0):.4f}",
            f"Today: ${float(info.get('usage_daily') or 0):.4f}",
            f"This week: ${float(info.get('usage_weekly') or 0):.4f}",
            f"This month: ${float(info.get('usage_monthly') or 0):.4f}",
        ]
        if info.get("limit") is not None:
            lines.append(
                f"Key limit: ${float(info['limit']):.4f} ({info.get('limit_reset') or 'reset unknown'})"
            )
            lines.append(f"Remaining: ${float(info.get('limit_remaining') or 0):.4f}")
        if info.get("expires_at"):
            lines.append(f"Expires: {info['expires_at']}")
        await self._reply_text(update, "\n".join(lines))

    @staticmethod
    def _parse_model_filter(arguments: list[str]) -> tuple[str | None, str]:
        if arguments and arguments[0].lower() in {"text", "image", "file", "audio", "video"}:
            return arguments[0].lower(), " ".join(arguments[1:]).strip()
        return None, " ".join(arguments).strip()

    async def models(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Browse the live OpenRouter text-output model catalog."""
        api_key = await self._preflight(update, context, enforce_budget=False)
        if api_key is None:
            return
        input_modality, query = self._parse_model_filter(context.args)
        try:
            models = await self.openrouter.list_models(
                api_key,
                output_modality="text",
                input_modality=input_modality,
                query=query,
            )
        except OpenRouterError as exc:
            await self._reply_text(update, str(exc))
            return
        self.catalog_views[(self._user_id(update), "text")] = CatalogView(
            kind="text", models=models, query=query, input_modality=input_modality
        )
        await self._show_catalog_page(update, context, "text", 0)

    async def image_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Browse the live OpenRouter image-output model catalog."""
        api_key = await self._preflight(update, context, enforce_budget=False)
        if api_key is None:
            return
        query = " ".join(context.args).strip()
        try:
            models = await self.openrouter.list_models(
                api_key, output_modality="image", query=query
            )
        except OpenRouterError as exc:
            await self._reply_text(update, str(exc))
            return
        self.catalog_views[(self._user_id(update), "image")] = CatalogView(
            kind="image", models=models, query=query
        )
        await self._show_catalog_page(update, context, "image", 0)

    async def _show_catalog_page(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        kind: str,
        page: int,
        *,
        edit: bool = False,
    ) -> None:
        user_id = self._user_id(update)
        view = self.catalog_views.get((user_id, kind))
        if view is None:
            await self._reply_text(
                update, "Run /models or /imagemodels again to refresh the catalog."
            )
            return
        total_pages = max(1, math.ceil(len(view.models) / self.MODEL_PAGE_SIZE))
        page = min(max(page, 0), total_pages - 1)
        first = page * self.MODEL_PAGE_SIZE
        visible = view.models[first : first + self.MODEL_PAGE_SIZE]
        current = (
            self.state.preferences_for(user_id).model
            if kind == "text"
            else self.state.preferences_for(user_id).image_model
        )
        filter_text = f" • input={view.input_modality}" if view.input_modality else ""
        query_text = f" • search={view.query}" if view.query else ""
        text = (
            f"OpenRouter {'chat' if kind == 'text' else 'image'} models "
            f"({len(view.models)} results{filter_text}{query_text})\n"
            f"Page {page + 1}/{total_pages} • current: {current or 'not selected'}"
        )
        rows: list[list[InlineKeyboardButton]] = []
        for index, model in enumerate(visible, start=first):
            label = f"✓ {model.name}" if model.id == current else model.name
            rows.append(
                [InlineKeyboardButton(label[:55], callback_data=f"modelpick:{kind[0]}:{index}")]
            )
        navigation: list[InlineKeyboardButton] = []
        if page > 0:
            navigation.append(
                InlineKeyboardButton("‹ Previous", callback_data=f"modelpage:{kind[0]}:{page - 1}")
            )
        if page + 1 < total_pages:
            navigation.append(
                InlineKeyboardButton("Next ›", callback_data=f"modelpage:{kind[0]}:{page + 1}")
            )
        if navigation:
            rows.append(navigation)
        markup = InlineKeyboardMarkup(rows) if rows else None
        if not visible:
            text += "\nNo matching models. Try a broader search."
        if edit and update.callback_query:
            await update.callback_query.edit_message_text(text=text, reply_markup=markup)
        else:
            await update.effective_message.reply_text(text=text, reply_markup=markup)

    async def model_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle catalog paging and model selection buttons."""
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()
        parts = query.data.split(":")
        if len(parts) != 3:
            return
        action, short_kind, raw_value = parts
        kind = "text" if short_kind == "t" else "image"
        try:
            value = int(raw_value)
        except ValueError:
            return
        view = self.catalog_views.get((self._user_id(update), kind))
        if view is None:
            await query.edit_message_text(
                "This catalog expired. Run /models or /imagemodels again."
            )
            return
        if action == "modelpage":
            await self._show_catalog_page(update, context, kind, value, edit=True)
            return
        if action != "modelpick" or value < 0 or value >= len(view.models):
            return
        model = view.models[value]
        if kind == "text":
            self.state.set_model(self._user_id(update), model.id)
            self.openrouter.reset_user_history(self._user_id(update))
        else:
            self.state.set_image_model(self._user_id(update), model.id)
        await query.edit_message_text(
            f"Selected {model.name}\n{model.id}\n{model.price_summary()}\n"
            f"Input: {', '.join(sorted(model.input_modalities))} • "
            f"Output: {', '.join(sorted(model.output_modalities))}"
        )

    async def model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show or select an exact OpenRouter chat-model slug."""
        api_key = await self._preflight(update, context, enforce_budget=False)
        if api_key is None:
            return
        preferences = self.state.preferences_for(self._user_id(update))
        if not context.args:
            await self._reply_text(
                update,
                f"Current chat model: {preferences.model}\n"
                "Select with /model provider/model or browse with /models.",
            )
            return
        model_id = context.args[0].strip()
        try:
            model = await self.openrouter.get_model(api_key, model_id)
            if "text" not in model.output_modalities:
                raise OpenRouterError("That model does not produce text; use /imagemodel instead.")
        except OpenRouterError as exc:
            await self._reply_text(update, str(exc))
            return
        self.state.set_model(self._user_id(update), model.id)
        self.openrouter.reset_user_history(self._user_id(update))
        await self._reply_text(
            update,
            f"Selected {model.name}\n{model.id}\n{model.price_summary()}\n"
            f"Inputs: {', '.join(sorted(model.input_modalities))}",
        )

    def _default_system_prompt(self) -> str:
        return str(self.openrouter.config.get("assistant_prompt") or "You are a helpful assistant.")

    def _effective_system_prompt(self, update: Update) -> str:
        custom_prompt = self.state.system_prompt_for(
            self._user_id(update), self._session_id(update)
        )
        return custom_prompt or self._default_system_prompt()

    async def _apply_system_prompt(self, update: Update, prompt: str) -> None:
        normalized = prompt.strip()
        maximum = int(self.config["system_prompt_max_chars"])
        if not normalized:
            await self._reply_text(update, "The system prompt cannot be empty.")
            return
        if len(normalized) > maximum:
            await self._reply_text(
                update,
                f"That system prompt has {len(normalized):,} characters; the configured limit "
                f"is {maximum:,}.",
            )
            return
        user_id = self._user_id(update)
        session_id = self._session_id(update)
        self.state.set_system_prompt(user_id, session_id, normalized)
        self.openrouter.reset_chat_history(user_id, session_id)
        await self._reply_text(
            update, "System prompt saved for this chat. Its conversation history was reset."
        )

    async def _restore_default_system_prompt(self, update: Update) -> None:
        user_id = self._user_id(update)
        session_id = self._session_id(update)
        self.awaiting_system_prompts.discard((user_id, session_id))
        self.state.clear_system_prompt(user_id, session_id)
        self.openrouter.reset_chat_history(user_id, session_id)
        await self._reply_text(
            update, "This chat now uses the deployment default system prompt. History was reset."
        )

    async def system_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show or change the custom system prompt for this user/chat/topic."""
        if await self._preflight(update, context, require_key=False, enforce_budget=False) != "":
            return
        supplied_prompt = message_text(update.effective_message)
        if supplied_prompt:
            command = supplied_prompt.casefold()
            if command in {"reset", "default", "off"}:
                await self._restore_default_system_prompt(update)
            elif command == "cancel":
                self.awaiting_system_prompts.discard(
                    (self._user_id(update), self._session_id(update))
                )
                await self._reply_text(update, "System-prompt editing cancelled.")
            else:
                await self._apply_system_prompt(update, supplied_prompt)
            return

        custom_prompt = self.state.system_prompt_for(
            self._user_id(update), self._session_id(update)
        )
        current = custom_prompt or self._default_system_prompt()
        source = "Custom prompt for this chat" if custom_prompt else "Deployment default"
        visible = current if len(current) <= 3000 else current[:3000] + "\n[…truncated]"
        markup = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Change prompt", callback_data="systemprompt:set")],
                [
                    InlineKeyboardButton(
                        "Use deployment default", callback_data="systemprompt:reset"
                    )
                ],
            ]
        )
        await update.effective_message.reply_text(
            f"{source}:\n\n{visible}",
            reply_markup=markup,
            message_thread_id=get_thread_id(update),
        )

    async def system_prompt_callback(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the per-chat system-prompt controls."""
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()
        session_key = (self._user_id(update), self._session_id(update))
        if query.data == "systemprompt:reset":
            self.awaiting_system_prompts.discard(session_key)
            self.state.clear_system_prompt(*session_key)
            self.openrouter.reset_chat_history(*session_key)
            await query.edit_message_text(
                "This chat now uses the deployment default system prompt. History was reset."
            )
            return
        if query.data != "systemprompt:set" or query.message is None:
            return
        self.awaiting_system_prompts.add(session_key)
        await query.message.reply_text(
            "Send the new system prompt now. Telegram-split chunks will be joined. "
            "Use /system cancel to stop.",
            reply_markup=ForceReply(
                selective=True, input_field_placeholder="Enter this chat's system prompt"
            ),
            message_thread_id=get_thread_id(update),
        )

    async def image_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show or select an exact OpenRouter image-model slug."""
        api_key = await self._preflight(update, context, enforce_budget=False)
        if api_key is None:
            return
        preferences = self.state.preferences_for(self._user_id(update))
        if not context.args:
            await self._reply_text(
                update,
                f"Current image model: {preferences.image_model or 'not selected'}\n"
                "Select with /imagemodel provider/model or browse with /imagemodels.",
            )
            return
        model_id = context.args[0].strip()
        try:
            model = await self.openrouter.get_model(api_key, model_id)
            if "image" not in model.output_modalities:
                raise OpenRouterError("That model does not generate images; use /model instead.")
        except OpenRouterError as exc:
            await self._reply_text(update, str(exc))
            return
        self.state.set_image_model(self._user_id(update), model.id)
        await self._reply_text(
            update,
            f"Selected image model {model.name}\n{model.id}\nInputs: "
            f"{', '.join(sorted(model.input_modalities))}",
        )

    async def budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show, set, or disable the user's local soft budget."""
        if await self._preflight(update, context, require_key=False, enforce_budget=False) != "":
            return
        user_id = self._user_id(update)
        preferences = self.state.preferences_for(user_id)
        tracker = self._tracker(update)
        if not context.args:
            if preferences.budget_limit is None:
                personal = "Local budget: off"
            else:
                spent = tracker.get_cost_for_period(preferences.budget_period)
                personal = (
                    f"Local budget: ${preferences.budget_limit:.2f} {preferences.budget_period}\n"
                    f"Tracked spend: ${spent:.6f}\nRemaining before next request: "
                    f"${max(0.0, preferences.budget_limit - spent):.6f}"
                )
            server_remaining, _ = self._remaining_budgets(update)
            server = (
                "Deployment budget: unlimited"
                if math.isinf(server_remaining)
                else f"Deployment budget remaining: ${max(0.0, server_remaining):.6f}"
            )
            await self._reply_text(
                update,
                f"{personal}\n{server}\n\nSet: /budget 5 monthly\nDisable: /budget off\n"
                "Periods: daily, weekly, monthly, all-time. This is a pre-request soft cap, so "
                "one request can exceed it; use an OpenRouter key limit for a hard cap.",
            )
            return
        if context.args[0].lower() == "off":
            self.state.clear_budget(user_id)
            await self._reply_text(update, "Your local soft budget is disabled.")
            return
        try:
            limit = float(context.args[0].removeprefix("$"))
            period = context.args[1].lower() if len(context.args) > 1 else "monthly"
            if period not in VALID_BUDGET_PERIODS:
                raise ValueError("invalid period")
            self.state.set_budget(user_id, limit, period)
        except (ValueError, IndexError):
            await self._reply_text(
                update,
                "Usage: /budget AMOUNT [daily|weekly|monthly|all-time], for example "
                "/budget 5 monthly, or /budget off.",
            )
            return
        await self._reply_text(update, f"Local soft budget set to ${limit:.2f} {period}.")

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show local exact costs, current model, and remote key limits."""
        api_key = await self._preflight(update, context, enforce_budget=False)
        if api_key is None:
            return
        tracker = self._tracker(update)
        tokens_today, tokens_month = tracker.get_current_token_usage()
        current_cost = tracker.get_current_cost()
        preferences = self.state.preferences_for(self._user_id(update))
        message_count, history_size = self.openrouter.get_conversation_stats(
            self._user_id(update), self._session_id(update)
        )
        lines = [
            f"Chat model: {preferences.model}",
            f"Image model: {preferences.image_model or 'not selected'}",
            f"Conversation: {message_count} messages, ~{history_size:,} serialized characters",
            "",
            f"Today: {tokens_today:,} tokens • ${current_cost['cost_today']:.6f}",
            f"This month: {tokens_month:,} tokens • ${current_cost['cost_month']:.6f}",
            f"All time: ${current_cost['cost_all_time']:.6f}",
        ]
        model_usage = tracker.get_model_usage()
        if model_usage:
            lines.append("\nToday's models:")
            for model, values in sorted(
                model_usage.items(), key=lambda item: item[1].get("cost", 0), reverse=True
            )[:5]:
                lines.append(
                    f"• {model}: {values.get('requests', 0)} requests, "
                    f"{values.get('tokens', 0):,} tokens, ${values.get('cost', 0):.6f}"
                )
        try:
            key_info = await self.openrouter.get_key_info(api_key)
            lines.append(f"\nOpenRouter key usage: ${float(key_info.get('usage') or 0):.4f}")
            if key_info.get("limit_remaining") is not None:
                lines.append(
                    f"OpenRouter key limit remaining: ${float(key_info['limit_remaining']):.4f}"
                )
        except OpenRouterError as exc:
            lines.append(f"\nCould not refresh key totals: {exc}")
        await self._reply_text(update, "\n".join(lines))

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reset the current user/chat/thread conversation."""
        if await self._preflight(update, context, enforce_budget=False) is None:
            return
        self.openrouter.reset_chat_history(self._user_id(update), self._session_id(update))
        await self._reply_text(update, "Conversation reset.")

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Repeat the user's last text prompt in this conversation."""
        session_key = (self._user_id(update), self._session_id(update))
        prompt = self.last_message.get(session_key)
        if not prompt:
            await self._reply_text(update, "There is no previous text prompt to resend.")
            return
        await self._infer(update, context, prompt)

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate an image through OpenRouter's dedicated Image API."""
        if not self.config["enable_image_generation"]:
            return
        api_key = await self._preflight(update, context)
        if api_key is None:
            return
        prompt = message_text(update.effective_message)
        if not prompt:
            await self._reply_text(update, "Usage: /image describe the image to create")
            return
        model_id = self.state.preferences_for(self._user_id(update)).image_model
        if not model_id:
            await self._reply_text(update, "Choose an image model first with /imagemodels.")
            return
        await update.effective_chat.send_action(
            constants.ChatAction.UPLOAD_PHOTO, message_thread_id=get_thread_id(update)
        )
        try:
            result = await self.openrouter.generate_image(api_key, model_id, prompt)
            await self._send_generated_images(update, result)
            self._record_usage(update, result.usage, result.model)
        except OpenRouterError as exc:
            await self._reply_text(update, str(exc))

    async def _send_generated_images(self, update: Update, result: ImageGenerationResult) -> None:
        for index, image in enumerate(result.images):
            suffix = {
                "image/jpeg": "jpg",
                "image/webp": "webp",
                "image/svg+xml": "svg",
            }.get(image.media_type, "png")
            file_object = io.BytesIO(image.data)
            file_object.name = f"openrouter-image-{index + 1}.{suffix}"
            if image.media_type == "image/svg+xml":
                await update.effective_message.reply_document(
                    document=file_object,
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=(
                        get_reply_to_message_id(self.config, update) if index == 0 else None
                    ),
                )
            else:
                await update.effective_message.reply_photo(
                    photo=file_object,
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=(
                        get_reply_to_message_id(self.config, update) if index == 0 else None
                    ),
                )
        if self.config["show_usage"]:
            await self._reply_text(update, self._usage_footer(result.usage, result.model).lstrip())

    def _queue_text_batch(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        purpose: str,
    ) -> None:
        """Debounce consecutive Telegram chunks into one logical message."""
        user_session = (self._user_id(update), self._session_id(update))
        batch_key = (*user_session, purpose)
        message_id = update.effective_message.message_id
        batch = self.pending_text_batches.get(batch_key)
        if batch is None:
            batch = PendingTextBatch(update=update, context=context, parts=[])
            self.pending_text_batches[batch_key] = batch
        elif batch.task is not None and not batch.task.done():
            batch.task.cancel()
        batch.update = update
        batch.context = context
        batch.parts.append((message_id, text))
        batch.task = context.application.create_task(
            self._flush_text_batch(batch_key, batch), update=update
        )

    async def _flush_text_batch(
        self,
        batch_key: tuple[int, str, str],
        batch: PendingTextBatch,
    ) -> None:
        await asyncio.sleep(max(0.0, float(self.config["message_batch_window_seconds"])))
        if self.pending_text_batches.get(batch_key) is not batch:
            return
        self.pending_text_batches.pop(batch_key, None)
        combined = "\n".join(text for _, text in sorted(batch.parts)).strip()
        if not combined:
            return
        user_id, session_id, purpose = batch_key
        if purpose == "system":
            self.awaiting_system_prompts.discard((user_id, session_id))
            await self._apply_system_prompt(batch.update, combined)
            return
        self.last_message[(user_id, session_id)] = combined
        await self._infer(batch.update, batch.context, combined)

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Queue a Telegram text message for one debounced OpenRouter request."""
        if update.edited_message or not update.message or update.message.via_bot:
            return
        prompt = message_text(update.message)
        session_key = (self._user_id(update), self._session_id(update))
        entering_system_prompt = session_key in self.awaiting_system_prompts
        if is_group_chat(update):
            trigger = str(self.config["group_trigger_keyword"])
            is_chat_command = bool(
                update.message.text and update.message.text.lower().startswith("/chat")
            )
            is_split_continuation = (*session_key, "prompt") in self.pending_text_batches
            if entering_system_prompt:
                pass
            elif trigger and prompt.lower().startswith(trigger.lower()):
                prompt = prompt[len(trigger) :].strip()
            elif (
                not is_chat_command
                and not is_split_continuation
                and not (
                    update.message.reply_to_message
                    and update.message.reply_to_message.from_user
                    and update.message.reply_to_message.from_user.id == context.bot.id
                )
            ):
                return
            if (
                not entering_system_prompt
                and not is_split_continuation
                and update.message.reply_to_message
                and update.message.reply_to_message.text
            ):
                prompt = f"Quoted message:\n{update.message.reply_to_message.text}\n\n{prompt}"
        if not prompt:
            await self._reply_text(update, "Send some text after /chat.")
            return
        purpose = "system" if entering_system_prompt else "prompt"
        self._queue_text_batch(update, context, prompt, purpose)

    async def photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a Telegram photo/image document to a vision-capable model."""
        if is_group_chat(update) and self.config["ignore_group_attachments"]:
            return
        api_key = await self._preflight(update, context)
        if api_key is None:
            return
        attachment = update.effective_message.effective_attachment
        telegram_file = attachment[-1] if isinstance(attachment, tuple | list) else attachment
        if telegram_file is None:
            return
        mime_type = "image/jpeg"
        if update.effective_message.document and update.effective_message.document.mime_type:
            mime_type = update.effective_message.document.mime_type
        try:
            model_id = self.state.preferences_for(self._user_id(update)).model
            model = await self.openrouter.get_model(api_key, model_id)
            if "image" not in model.input_modalities:
                raise OpenRouterError(
                    f"{model_id} does not accept images. Use /models image to choose a vision model."
                )
            media = await context.bot.get_file(telegram_file.file_id)
            data = bytes(await media.download_as_bytearray())
            self._check_file_size(data)
            prompt = update.effective_message.caption or "Describe and analyze this image."
            content = image_content(data, mime_type, prompt)
            await self._infer(update, context, content, api_key=api_key)
        except (OpenRouterError, ValueError) as exc:
            await self._reply_text(update, str(exc))

    def _check_file_size(self, data: bytes) -> None:
        maximum = int(self.config["max_file_size_mb"]) * 1024 * 1024
        if len(data) > maximum:
            raise ValueError(
                f"This file is {len(data) / 1024 / 1024:.1f} MB; the configured limit is "
                f"{self.config['max_file_size_mb']} MB."
            )

    @staticmethod
    def _is_text_document(filename: str, mime_type: str) -> bool:
        extension = Path(filename.lower()).suffix
        return (
            mime_type.startswith("text/")
            or mime_type in TEXT_MIME_TYPES
            or extension in TEXT_EXTENSIONS
        )

    async def document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send PDFs, text/code, and native model-supported files to OpenRouter."""
        if is_group_chat(update) and self.config["ignore_group_attachments"]:
            return
        api_key = await self._preflight(update, context)
        if api_key is None or update.effective_message.document is None:
            return
        document = update.effective_message.document
        filename = document.file_name or "telegram-upload"
        mime_type = (
            document.mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        try:
            media = await context.bot.get_file(document.file_id)
            data = bytes(await media.download_as_bytearray())
            self._check_file_size(data)
            prompt = update.effective_message.caption or (
                f"Read and analyze the attached file {filename}. Give a substantive text response."
            )
            if self._is_text_document(filename, mime_type):
                decoded = data.decode("utf-8-sig")
                if not decoded.strip():
                    raise ValueError("The uploaded text file is empty.")
                max_chars = int(self.config["text_file_max_chars"])
                if len(decoded) > max_chars:
                    raise ValueError(
                        f"Decoded text has {len(decoded):,} characters; the configured limit is "
                        f"{max_chars:,}."
                    )
                content: str | list[dict[str, Any]] = (
                    f"{prompt}\n\n--- BEGIN FILE: {filename} ---\n{decoded}\n"
                    f"--- END FILE: {filename} ---"
                )
            else:
                model_id = self.state.preferences_for(self._user_id(update)).model
                model = await self.openrouter.get_model(api_key, model_id)
                if mime_type != "application/pdf" and "file" not in model.input_modalities:
                    raise OpenRouterError(
                        f"{model_id} does not advertise native file input for {mime_type}. "
                        "Use /models file, or upload PDF/text/code instead."
                    )
                content = file_content(data, filename, mime_type, prompt)
            await self._infer(update, context, content, api_key=api_key, force_non_streaming=True)
        except UnicodeDecodeError:
            await self._reply_text(
                update,
                "That file looks textual but is not valid UTF-8. Convert it to UTF-8 or choose a "
                "model with native file support using /models file.",
            )
        except (OpenRouterError, ValueError) as exc:
            await self._reply_text(update, str(exc))

    async def _infer(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        content: str | list[dict[str, Any]],
        *,
        api_key: str | None = None,
        force_non_streaming: bool = False,
    ) -> None:
        if api_key is None:
            api_key = await self._preflight(update, context)
            if api_key is None:
                return
        model_id = self.state.preferences_for(self._user_id(update)).model
        system_prompt = self._effective_system_prompt(update)
        await update.effective_chat.send_action(
            constants.ChatAction.TYPING, message_thread_id=get_thread_id(update)
        )
        try:
            contains_file = isinstance(content, list) and any(
                part.get("type") == "file" for part in content
            )
            if self.config["stream"] and not contains_file and not force_non_streaming:
                final = await self._stream_response(
                    update, api_key, model_id, content, system_prompt
                )
                self._record_usage(update, final.usage, final.model or model_id)
            else:
                result = await self.openrouter.chat(
                    api_key,
                    self._user_id(update),
                    self._session_id(update),
                    model_id,
                    content,
                    system_prompt=system_prompt,
                )
                await self._reply_text(
                    update, result.text + self._usage_footer(result.usage, result.model or model_id)
                )
                self._record_usage(update, result.usage, result.model or model_id)
        except OpenRouterError as exc:
            logging.warning(
                "OpenRouter request failed for Telegram user %s: %s", self._user_id(update), exc
            )
            await self._reply_text(update, str(exc))

    async def _stream_response(
        self,
        update: Update,
        api_key: str,
        model_id: str,
        content: str | list[dict[str, Any]],
        system_prompt: str,
    ) -> StreamUpdate:
        sent_message = None
        last_edit = ""
        final_update = StreamUpdate(text="", done=True, model=model_id)
        async for stream_update in self.openrouter.chat_stream(
            api_key,
            self._user_id(update),
            self._session_id(update),
            model_id,
            content,
            system_prompt=system_prompt,
        ):
            if stream_update.done:
                final_update = stream_update
                break
            visible = stream_update.text[:4096]
            if not visible:
                continue
            if sent_message is None:
                sent_message = await update.effective_message.reply_text(
                    visible,
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                )
                last_edit = visible
            elif len(visible) - len(last_edit) >= 80:
                try:
                    await sent_message.edit_text(visible)
                    last_edit = visible
                except RetryAfter as exc:
                    await asyncio.sleep(exc.retry_after)
                except (TimedOut, BadRequest):
                    pass

        final_text = final_update.text + self._usage_footer(
            final_update.usage, final_update.model or model_id
        )
        chunks = split_into_chunks(final_text or "(empty response)")
        if sent_message is None:
            await self._reply_text(update, final_text)
        else:
            with contextlib.suppress(BadRequest):
                await sent_message.edit_text(chunks[0], disable_web_page_preview=True)
            for chunk in chunks[1:]:
                await update.effective_message.reply_text(
                    chunk,
                    message_thread_id=get_thread_id(update),
                    disable_web_page_preview=True,
                )
        return final_update

    async def post_init(self, application: Application) -> None:
        """Register Telegram command menus."""
        await application.bot.set_my_commands(
            self.group_commands, scope=BotCommandScopeAllGroupChats()
        )
        await application.bot.set_my_commands(self.commands)

    async def post_shutdown(self, _: Application) -> None:
        """Release OpenRouter HTTP connections during shutdown."""
        tasks = [
            batch.task
            for batch in self.pending_text_batches.values()
            if batch.task is not None and not batch.task.done()
        ]
        self.pending_text_batches.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.openrouter.close()

    async def error_handler(self, _: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log unexpected Telegram update failures without exposing secrets."""
        logging.error("Exception while handling Telegram update", exc_info=context.error)

    def build_application(self) -> Application:
        """Build the Telegram application and register all handlers."""
        builder = (
            ApplicationBuilder()
            .token(self.config["token"])
            .post_init(self.post_init)
            .post_shutdown(self.post_shutdown)
            .concurrent_updates(True)
        )
        if self.config.get("proxy"):
            builder = builder.proxy_url(self.config["proxy"]).get_updates_proxy_url(
                self.config["proxy"]
            )
        application = builder.build()

        application.add_handler(CommandHandler("start", self.help))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(CommandHandler("key", self.set_key))
        application.add_handler(CommandHandler("keyinfo", self.key_info))
        application.add_handler(CommandHandler("logout", self.logout))
        application.add_handler(CommandHandler("models", self.models))
        application.add_handler(CommandHandler("model", self.model))
        application.add_handler(CommandHandler("system", self.system_prompt))
        application.add_handler(CommandHandler("imagemodels", self.image_models))
        application.add_handler(CommandHandler("imagemodel", self.image_model))
        application.add_handler(CommandHandler("image", self.image))
        application.add_handler(CommandHandler("budget", self.budget))
        application.add_handler(CommandHandler("stats", self.stats))
        application.add_handler(CommandHandler("reset", self.reset))
        application.add_handler(CommandHandler("resend", self.resend))
        application.add_handler(
            CommandHandler("chat", self.prompt, filters=filters.ChatType.GROUPS)
        )
        application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.photo))
        application.add_handler(
            MessageHandler(filters.Document.ALL & ~filters.Document.IMAGE, self.document)
        )
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.prompt))
        application.add_handler(
            CallbackQueryHandler(self.model_callback, pattern=r"^model(?:pick|page):[ti]:\d+$")
        )
        application.add_handler(
            CallbackQueryHandler(
                self.system_prompt_callback, pattern=r"^systemprompt:(?:set|reset)$"
            )
        )
        application.add_error_handler(self.error_handler)
        return application

    def run(self) -> None:
        """Run Telegram long polling until stopped."""
        application = self.build_application()
        application.run_polling(allowed_updates=Update.ALL_TYPES)
