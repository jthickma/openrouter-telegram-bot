"""Contract tests for OpenRouter request and response handling."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from openrouter_helper import (
    ModelInfo,
    OpenRouterError,
    OpenRouterHelper,
    UsageInfo,
    data_url,
    file_content,
    image_content,
)


def model_payload(
    model_id: str = "vendor/text-model",
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> dict[str, Any]:
    """Build a representative Models API item."""
    return {
        "id": model_id,
        "name": "Test Model",
        "description": "A model used by tests",
        "context_length": 128000,
        "architecture": {
            "input_modalities": inputs or ["text", "image", "file"],
            "output_modalities": outputs or ["text"],
        },
        "supported_parameters": ["temperature", "max_tokens"],
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
    }


def helper_with_transport(handler: httpx.MockTransport) -> OpenRouterHelper:
    """Create a helper whose requests are handled entirely in memory."""
    return OpenRouterHelper(
        {
            "base_url": "https://openrouter.test/api/v1",
            "assistant_prompt": "Be helpful.",
            "max_history_size": 7,
            "max_conversation_age_minutes": 180,
            "temperature": 0.4,
            "max_tokens": 777,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "pdf_engine": "cloudflare-ai",
        },
        transport=handler,
    )


def test_multimodal_content_uses_documented_shapes() -> None:
    """Images and files should use OpenRouter's content-part schemas."""
    image = image_content(b"image", "image/png", "look")
    assert image[0] == {"type": "text", "text": "look"}
    assert image[1]["image_url"]["url"] == data_url(b"image", "image/png")

    file_parts = file_content(b"pdf", "report.pdf", "application/pdf", "summarize")
    assert file_parts[1]["file"]["filename"] == "report.pdf"
    assert file_parts[1]["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_model_metadata_normalizes_prices_and_modalities() -> None:
    """Model fields used for selection should be normalized predictably."""
    model = ModelInfo.from_payload(model_payload())
    assert model.input_modalities == frozenset({"text", "image", "file"})
    assert model.output_modalities == frozenset({"text"})
    assert model.price_summary() == "$1/$2 per 1M in/out"


def test_optional_metadata_and_validation_fallbacks() -> None:
    """Optional usage, parameter maps, and media validation should be defensive."""
    assert UsageInfo.from_payload(None) == UsageInfo()
    model = ModelInfo.from_payload(
        {
            "id": "vendor/free",
            "supported_parameters": {"temperature": {"type": "number"}},
        }
    )
    assert model.supported_parameters == frozenset({"temperature"})
    assert model.price_summary() == "pricing varies/free"
    with pytest.raises(ValueError, match="Unsupported image type"):
        image_content(b"bitmap", "image/bmp", "look")


@pytest.mark.asyncio
async def test_attribution_headers_are_optional() -> None:
    """Configured attribution should be present without affecting bearer auth."""
    helper = OpenRouterHelper(
        {
            "http_referer": "https://example.test",
            "app_title": "Test Bot",
        },
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})),
    )
    try:
        headers = helper._headers("secret")
        assert headers["HTTP-Referer"] == "https://example.test"
        assert headers["X-Title"] == "Test Bot"
        assert headers["Authorization"] == "Bearer secret"
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_key_validation_uses_bearer_auth() -> None:
    """The user's key should be sent only as the Authorization bearer token."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/key"
        assert request.headers["Authorization"] == "Bearer sk-or-test"
        return httpx.Response(200, json={"data": {"label": "test", "usage": 1.5}})

    helper = helper_with_transport(httpx.MockTransport(handler))
    try:
        info = await helper.validate_api_key("sk-or-test")
        assert info["label"] == "test"
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_pdf_chat_uses_parser_and_actual_usage_cost() -> None:
    """PDF requests should use the configured parser and returned cost."""
    captured_chat: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [model_payload()]})
        if request.url.path == "/api/v1/chat/completions":
            captured_chat.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "vendor/text-model",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "A concise summary.",
                                "annotations": [
                                    {
                                        "type": "file",
                                        "file": {"hash": "abc", "content": []},
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "cost": 0.0042,
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    helper = helper_with_transport(httpx.MockTransport(handler))
    content = file_content(b"pdf bytes", "report.pdf", "application/pdf", "summarize")
    try:
        result = await helper.chat("sk-or-test", 1, "100:0", "vendor/text-model", content)
        assert result.text == "A concise summary."
        assert result.usage.cost == pytest.approx(0.0042)
        assert captured_chat["plugins"] == [
            {"id": "file-parser", "pdf": {"engine": "cloudflare-ai"}}
        ]
        assert captured_chat["temperature"] == 0.4
        assert captured_chat["max_tokens"] == 777
        assert "presence_penalty" not in captured_chat
        history = helper._histories[(1, "100:0")]
        assert "file_data" not in json.dumps(history)
        assert history[-1]["annotations"][0]["file"]["hash"] == "abc"
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_model_catalog_filters_search_and_uses_cache() -> None:
    """Catalog filtering should happen locally after one live API request."""
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.url.params["output_modalities"] == "text"
        return httpx.Response(
            200,
            json={
                "data": [
                    model_payload("vendor/vision", inputs=["text", "image"]),
                    model_payload("vendor/text", inputs=["text"]),
                ]
            },
        )

    helper = helper_with_transport(httpx.MockTransport(handler))
    try:
        models = await helper.list_models("sk-or-test", input_modality="image", query="VISION")
        cached = await helper.list_models("sk-or-test", query="vendor")
        assert [model.id for model in models] == ["vendor/vision"]
        assert [model.id for model in cached] == ["vendor/vision", "vendor/text"]
        assert request_count == 1
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_stream_collects_text_and_final_usage() -> None:
    """The final SSE event should supply exact tokens and cost."""
    events = [
        {"model": "vendor/text-model", "choices": [{"delta": {"content": "Hel"}}]},
        {"model": "vendor/text-model", "choices": [{"delta": {"content": "lo"}}]},
        {
            "model": "vendor/text-model",
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3, "cost": 0.001},
        },
    ]
    stream_body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [model_payload()]})
        if request.url.path == "/api/v1/chat/completions":
            assert json.loads(request.content)["stream"] is True
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=stream_body.encode(),
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    helper = helper_with_transport(httpx.MockTransport(handler))
    try:
        updates = [
            update
            async for update in helper.chat_stream(
                "sk-or-test", 2, "200:0", "vendor/text-model", "hello"
            )
        ]
        assert [update.text for update in updates] == ["Hel", "Hello", "Hello"]
        assert updates[-1].done is True
        assert updates[-1].usage.cost == pytest.approx(0.001)
        assert helper.get_conversation_stats(2, "200:0")[0] == 3
        helper.reset_chat_history(2, "200:0")
        assert helper.get_conversation_stats(2, "200:0") == (0, 0)
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_image_generation_decodes_base64_response() -> None:
    """Image API output should be decoded into a Telegram-ready byte stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            output = request.url.params["output_modalities"]
            data = (
                [] if output == "text" else [model_payload("vendor/image-model", outputs=["image"])]
            )
            return httpx.Response(200, json={"data": data})
        if request.url.path == "/api/v1/images":
            request_body = json.loads(request.content)
            assert request_body == {
                "model": "vendor/image-model",
                "prompt": "draw a fox",
                "n": 1,
            }
            return httpx.Response(
                200,
                json={
                    "data": [{"b64_json": "aW1hZ2U=", "media_type": "image/webp"}],
                    "usage": {"total_tokens": 10, "cost": 0.04},
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    helper = helper_with_transport(httpx.MockTransport(handler))
    try:
        result = await helper.generate_image("sk-or-test", "vendor/image-model", "draw a fox")
        assert result.images[0].data == b"image"
        assert result.images[0].media_type == "image/webp"
        assert result.usage.cost == pytest.approx(0.04)
    finally:
        await helper.close()


def test_dynamic_router_price_is_not_rendered_as_negative_cost() -> None:
    model = ModelInfo(
        id="openrouter/auto",
        name="Auto",
        description="",
        context_length=1_000_000,
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        supported_parameters=frozenset(),
        prompt_price=-1,
        completion_price=-1,
    )

    assert model.price_summary() == "dynamic routing price"


@pytest.mark.asyncio
async def test_api_errors_are_converted_to_user_safe_errors() -> None:
    """Structured API failures should retain status and message without response dumps."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": 401, "message": "Invalid API key"}})

    helper = helper_with_transport(httpx.MockTransport(handler))
    try:
        with pytest.raises(OpenRouterError, match="401: Invalid API key"):
            await helper.validate_api_key("bad-key")
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_network_errors_do_not_echo_credentials() -> None:
    """Connection failures should have a stable message and never contain the key."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    helper = helper_with_transport(httpx.MockTransport(handler))
    try:
        with pytest.raises(OpenRouterError, match="Could not reach OpenRouter") as error:
            await helper.validate_api_key("sk-or-secret")
        assert "sk-or-secret" not in str(error.value)
    finally:
        await helper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, text="not-json"), "invalid JSON"),
        (httpx.Response(200, json=[]), "unexpected response"),
        (httpx.Response(200, json={}), "API key details"),
    ],
)
async def test_invalid_key_endpoint_shapes_are_rejected(
    response: httpx.Response, message: str
) -> None:
    """Malformed successful responses must fail closed."""
    helper = helper_with_transport(httpx.MockTransport(lambda _: response))
    try:
        with pytest.raises(OpenRouterError, match=message):
            await helper.validate_api_key("sk-or-test")
    finally:
        await helper.close()
