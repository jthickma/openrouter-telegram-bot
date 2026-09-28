"""Exercise native picker navigation and search without Telegram/network access."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from openrouter_helper import ModelInfo
from telegram import ForceReply
from telegram_bot import CatalogView
from test_telegram_bot import build_test_bot


def model(slug, name, inputs=("text",)):
    return ModelInfo.from_payload(
        {
            "id": slug,
            "name": name,
            "context_length": 128000,
            "architecture": {"input_modalities": inputs, "output_modalities": ["text"]},
        }
    )


def update(user=7, chat=70, data=None, text="hello", reply=None):
    message = SimpleNamespace(
        message_id=10,
        is_topic_message=False,
        text=text,
        reply_to_message=reply,
        reply_text=AsyncMock(return_value=SimpleNamespace(message_id=99)),
        via_bot=None,
        parse_entities=Mock(return_value={}),
    )
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user),
        effective_chat=SimpleNamespace(id=chat, type="private"),
        effective_message=message,
        message=message,
        edited_message=None,
        callback_query=(
            SimpleNamespace(
                data=data,
                answer=AsyncMock(),
                edit_message_text=AsyncMock(),
                message=message,
            )
            if data
            else None
        ),
    )


@pytest_asyncio.fixture
async def picker(tmp_path):
    bot, helper = build_test_bot(tmp_path)
    bot._preflight = AsyncMock(return_value="key")
    bot._reply_text = AsyncMock()
    view = CatalogView(
        "text",
        [
            model("google/flash", "Google: Flash", ("text", "image")),
            model("anthropic/claude", "Anthropic: Claude"),
        ],
        "",
    )
    bot.catalog_views[(7, "70:0", "text")] = view
    yield bot, view, SimpleNamespace(args=[], bot=SimpleNamespace(id=123))
    await helper.close()


def callback(view, action, value="0", **kwargs):
    return update(data=f"catalog:t:{view.token}:{action}:{value}", **kwargs)


def test_search_terms_and_capability_intersect():
    view = CatalogView(
        "text",
        [model("google/flash", "Google: Flash", ("image",)), model("google/pro", "Google: Pro")],
        "FLASH google",
        "image",
    )
    assert [m.id for m in view.matches()] == ["google/flash"]
    view.query = "google missing"
    assert view.matches() == []


@pytest.mark.asyncio
async def test_picker_search_first_and_escaped_empty_state(picker):
    bot, view, context = picker
    view.query = "<missing>"
    event = update()
    await bot._show_catalog_page(event, context, "text", 0)
    content = event.message.reply_text.await_args.kwargs
    assert "&lt;missing&gt;" in content["text"]
    assert "No models found" in content["text"]
    buttons = content["reply_markup"].inline_keyboard
    assert buttons[0][0].text == "Edit search"
    assert any(b.text == "Clear filters" for row in buttons for b in row)
    assert all(len(b.callback_data.encode()) <= 64 for row in buttons for b in row)


@pytest.mark.asyncio
async def test_search_reply_is_consumed_before_inference(picker):
    bot, view, context = picker
    event = callback(view, "search")
    await bot.model_callback(event, context)
    markup = event.message.reply_text.await_args.kwargs["reply_markup"]
    assert isinstance(markup, ForceReply)
    assert markup.input_field_placeholder == "Search models or providers"
    old_token = view.token
    bot._queue_text_batch = Mock()
    await bot.prompt(update(text="FLASH google", reply=SimpleNamespace(message_id=99)), context)
    assert view.query == "FLASH google"
    assert view.token != old_token
    assert not bot.model_searches
    bot._queue_text_batch.assert_not_called()


@pytest.mark.asyncio
async def test_ordinary_chat_during_search_is_not_swallowed(picker):
    bot, view, context = picker
    await bot.model_callback(callback(view, "search"), context)
    bot._queue_text_batch = Mock()
    await bot.prompt(update(text="a normal message"), context)
    bot._queue_text_batch.assert_called_once()
    assert view.query == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [{"user": 8}, {"chat": 80}, {}])
async def test_foreign_or_stale_buttons_cannot_select(picker, scope):
    bot, view, context = picker
    event = callback(view, "pick", **scope)
    if not scope:
        view.token = "new-token"
    await bot.model_callback(event, context)
    assert bot.state.preferences_for(7).model == "openrouter/auto"
    event.callback_query.edit_message_text.assert_not_awaited()
    assert event.callback_query.answer.await_args.kwargs["show_alert"]


@pytest.mark.asyncio
async def test_filter_invalidates_old_indices_and_clear_restores_results(picker):
    bot, view, context = picker
    old = callback(view, "pick")
    await bot.model_callback(callback(view, "filter", "image"), context)
    assert len(view.matches()) == 1
    await bot.model_callback(old, context)
    assert bot.state.preferences_for(7).model == "openrouter/auto"
    view.query = "missing"
    await bot.model_callback(callback(view, "clear"), context)
    assert len(view.matches()) == 2
    assert view.query == "" and view.input_modality is None


@pytest.mark.asyncio
async def test_cancel_and_late_reply_never_reach_inference(picker):
    bot, view, context = picker
    await bot.model_callback(callback(view, "search"), context)
    await bot.cancel_search(update(), context)
    bot._queue_text_batch = Mock()
    late = SimpleNamespace(
        message_id=99, from_user=SimpleNamespace(id=123), text="Search models\n\nReply with a model"
    )
    await bot.prompt(update(text="claude", reply=late), context)
    bot._queue_text_batch.assert_not_called()
    assert not bot.model_searches


@pytest.mark.asyncio
async def test_selection_keeps_navigation_and_active_model_does_not_reset_history(picker):
    bot, view, context = picker
    bot.openrouter.reset_user_history = Mock()
    event = callback(view, "pick", "1")
    await bot.model_callback(event, context)
    assert bot.state.preferences_for(7).model == "anthropic/claude"
    content = event.callback_query.edit_message_text.await_args.kwargs
    assert content["reply_markup"].inline_keyboard[0][0].text == "Change model"
    await bot.model_callback(callback(view, "pick", "1"), context)
    bot.openrouter.reset_user_history.assert_called_once()


@pytest.mark.asyncio
async def test_page_clamping_and_six_model_limit(picker):
    bot, view, context = picker
    view.models = [model(f"provider/{i}", f"Model {i}") for i in range(14)]
    for page, expected_count, expected_label in [(0, 6, "Page 1 of 3"), (999, 2, "Page 3 of 3")]:
        event = update()
        await bot._show_catalog_page(event, context, "text", page)
        content = event.message.reply_text.await_args.kwargs
        assert expected_label in content["text"]
        assert (
            sum(
                ":pick:" in b.callback_data
                for row in content["reply_markup"].inline_keyboard
                for b in row
            )
            == expected_count
        )


@pytest.mark.asyncio
async def test_image_search_selects_image_model_only(picker):
    bot, view, context = picker
    view.kind = "image"
    bot.catalog_views[(7, "70:0", "image")] = view
    event = update(data=f"catalog:i:{view.token}:pick:0")
    await bot.model_callback(event, context)
    assert bot.state.preferences_for(7).image_model == "google/flash"
    assert bot.state.preferences_for(7).model == "openrouter/auto"


@pytest.mark.asyncio
async def test_menu_system_button_does_not_save_menu_as_prompt(picker):
    bot, _, context = picker
    bot._preflight.return_value = ""
    event = update(data="menu:system", text="OpenRouter menu")
    await bot.menu_callback(event, context)
    assert bot.state.system_prompt_for(7, "70:0") is None
    assert "Deployment default" in event.message.reply_text.await_args.args[0]
