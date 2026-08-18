"""Async OpenRouter API client and conversation manager."""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})


class OpenRouterError(RuntimeError):
    """A user-safe OpenRouter request error."""


@dataclass(frozen=True, slots=True)
class UsageInfo:
    """Token and billed-cost details returned by OpenRouter."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    @classmethod
    def from_payload(cls, payload: Any) -> UsageInfo:
        """Parse an OpenRouter usage object without assuming optional fields."""
        if not isinstance(payload, dict):
            return cls()
        return cls(
            prompt_tokens=int(payload.get("prompt_tokens") or 0),
            completion_tokens=int(payload.get("completion_tokens") or 0),
            total_tokens=int(payload.get("total_tokens") or 0),
            cost=float(payload.get("cost") or 0.0),
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """OpenRouter model metadata used by Telegram model selection."""

    id: str
    name: str
    description: str
    context_length: int
    input_modalities: frozenset[str]
    output_modalities: frozenset[str]
    supported_parameters: frozenset[str]
    prompt_price: float
    completion_price: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ModelInfo:
        """Build normalized model metadata from the Models API."""
        architecture = payload.get("architecture") or {}
        pricing = payload.get("pricing") or {}
        supported = payload.get("supported_parameters") or []
        if isinstance(supported, dict):
            supported = supported.keys()
        return cls(
            id=str(payload.get("id") or ""),
            name=str(payload.get("name") or payload.get("id") or "Unknown model"),
            description=str(payload.get("description") or ""),
            context_length=int(payload.get("context_length") or 0),
            input_modalities=frozenset(architecture.get("input_modalities") or ["text"]),
            output_modalities=frozenset(architecture.get("output_modalities") or ["text"]),
            supported_parameters=frozenset(str(value) for value in supported),
            prompt_price=float(pricing.get("prompt") or 0.0),
            completion_price=float(pricing.get("completion") or 0.0),
        )

    def price_summary(self) -> str:
        """Return compact per-million-token pricing for Telegram."""
        if self.prompt_price < 0 or self.completion_price < 0:
            return "dynamic routing price"
        prompt = self.prompt_price * 1_000_000
        completion = self.completion_price * 1_000_000
        if prompt == 0 and completion == 0:
            return "pricing varies/free"
        return f"${prompt:g}/${completion:g} per 1M in/out"


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """A completed text inference result."""

    text: str
    model: str
    usage: UsageInfo
    annotations: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class StreamUpdate:
    """One accumulated streaming update."""

    text: str
    done: bool
    model: str = ""
    usage: UsageInfo = field(default_factory=UsageInfo)


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """One generated image returned by OpenRouter's Image API."""

    data: bytes
    media_type: str


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    """Image outputs and billed usage for one request."""

    images: tuple[GeneratedImage, ...]
    model: str
    usage: UsageInfo


class OpenRouterHelper:
    """Call OpenRouter with per-user credentials and bounded chat history."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        proxy = config.get("proxy") or None
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=str(config.get("base_url") or OPENROUTER_API_BASE).rstrip("/"),
            proxy=proxy,
            transport=transport,
            timeout=httpx.Timeout(float(config.get("request_timeout", 180.0))),
            follow_redirects=True,
        )
        self._histories: dict[tuple[int, str], list[dict[str, Any]]] = {}
        self._last_updated: dict[tuple[int, str], dt.datetime] = {}
        self._session_locks: dict[tuple[int, str], asyncio.Lock] = {}
        self._model_cache: dict[str, tuple[float, list[ModelInfo]]] = {}

    async def close(self) -> None:
        """Close the shared HTTP connection pool."""
        await self.client.aclose()

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = str(self.config.get("http_referer") or "").strip()
        title = str(self.config.get("app_title") or "OpenRouter Telegram Bot").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error", payload) if isinstance(payload, dict) else payload
            if isinstance(error, dict):
                message = error.get("message") or error.get("code")
                if message:
                    return str(message)
            if isinstance(error, str):
                return error
        except (ValueError, TypeError):
            pass
        return response.reason_phrase or "OpenRouter request failed"

    async def _request_json(
        self,
        method: str,
        path: str,
        api_key: str,
        *,
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self.client.request(
                method,
                path,
                headers=self._headers(api_key),
                params=params,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise OpenRouterError("OpenRouter timed out. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError("Could not reach OpenRouter. Please try again.") from exc
        if response.is_error:
            raise OpenRouterError(
                f"OpenRouter returned {response.status_code}: {self._error_message(response)}"
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise OpenRouterError("OpenRouter returned an invalid JSON response.") from exc
        if not isinstance(result, dict):
            raise OpenRouterError("OpenRouter returned an unexpected response.")
        return result

    async def validate_api_key(self, api_key: str) -> dict[str, Any]:
        """Validate an API key and return its current limits and usage."""
        payload = await self._request_json("GET", "/key", api_key)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise OpenRouterError("OpenRouter did not return API key details.")
        return data

    async def get_key_info(self, api_key: str) -> dict[str, Any]:
        """Fetch current usage and limit information for an API key."""
        return await self.validate_api_key(api_key)

    async def list_models(
        self,
        api_key: str,
        *,
        output_modality: str = "text",
        input_modality: str | None = None,
        query: str = "",
        force_refresh: bool = False,
    ) -> list[ModelInfo]:
        """Fetch current models directly from OpenRouter and filter them."""
        cache_identity = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        cache_key = f"{cache_identity}:{output_modality}"
        cached = self._model_cache.get(cache_key)
        if force_refresh or cached is None or time.monotonic() - cached[0] > 300:
            payload = await self._request_json(
                "GET",
                "/models",
                api_key,
                params={"output_modalities": output_modality, "sort": "most-popular"},
            )
            raw_models = payload.get("data")
            if not isinstance(raw_models, list):
                raise OpenRouterError("OpenRouter did not return a model catalog.")
            models = [
                ModelInfo.from_payload(item)
                for item in raw_models
                if isinstance(item, dict) and item.get("id")
            ]
            self._model_cache[cache_key] = (time.monotonic(), models)
        else:
            models = cached[1]

        normalized_query = query.casefold().strip()
        return [
            model
            for model in models
            if output_modality in model.output_modalities
            and (input_modality is None or input_modality in model.input_modalities)
            and (
                not normalized_query
                or normalized_query in model.id.casefold()
                or normalized_query in model.name.casefold()
                or normalized_query in model.description.casefold()
            )
        ]

    async def get_model(self, api_key: str, model_id: str) -> ModelInfo:
        """Look up a model in the live text/image catalogs."""
        for output_modality in ("text", "image"):
            models = await self.list_models(api_key, output_modality=output_modality)
            match = next((model for model in models if model.id == model_id), None)
            if match is not None:
                return match
        raise OpenRouterError(f"Model '{model_id}' is not in OpenRouter's current catalog.")

    def reset_chat_history(self, user_id: int, session_id: str) -> None:
        """Clear one user's conversation in one Telegram chat/thread."""
        session_key = (user_id, session_id)
        self._histories.pop(session_key, None)
        self._last_updated.pop(session_key, None)

    def reset_user_history(self, user_id: int) -> None:
        """Clear every in-memory conversation for a user."""
        for session_key in [key for key in self._histories if key[0] == user_id]:
            self._histories.pop(session_key, None)
            self._last_updated.pop(session_key, None)

    def get_conversation_stats(self, user_id: int, session_id: str) -> tuple[int, int]:
        """Return message count and approximate serialized history size."""
        history = self._histories.get((user_id, session_id), [])
        serialized_size = sum(len(json.dumps(message)) for message in history)
        return len(history), serialized_size

    def _history_for(self, user_id: int, session_id: str) -> list[dict[str, Any]]:
        session_key = (user_id, session_id)
        last_updated = self._last_updated.get(session_key)
        max_age = dt.timedelta(minutes=int(self.config.get("max_conversation_age_minutes", 180)))
        if last_updated is not None and dt.datetime.now(dt.UTC) - last_updated > max_age:
            self.reset_chat_history(user_id, session_id)
        if session_key not in self._histories:
            self._histories[session_key] = [
                {
                    "role": "system",
                    "content": str(
                        self.config.get("assistant_prompt") or "You are a helpful assistant."
                    ),
                }
            ]
        self._last_updated[session_key] = dt.datetime.now(dt.UTC)
        return self._histories[session_key]

    def _trim_history(self, history: list[dict[str, Any]]) -> None:
        max_messages = max(3, int(self.config.get("max_history_size", 15)))
        while len(history) > max_messages:
            history.pop(1)

    @staticmethod
    def _stable_user_id(user_id: int) -> str:
        digest = hashlib.sha256(f"telegram:{user_id}".encode()).hexdigest()
        return f"tg-{digest[:24]}"

    @staticmethod
    def _contains_pdf(content: str | list[dict[str, Any]]) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if part.get("type") != "file":
                continue
            file_data = (part.get("file") or {}).get("file_data", "")
            if isinstance(file_data, str) and file_data.startswith("data:application/pdf"):
                return True
        return False

    async def _chat_payload(
        self,
        api_key: str,
        user_id: int,
        model_id: str,
        messages: list[dict[str, Any]],
        content: str | list[dict[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        model = await self.get_model(api_key, model_id)
        if "text" not in model.output_modalities:
            raise OpenRouterError(
                f"{model_id} does not produce text. Select it with /imagemodel and use /image."
            )
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages + [{"role": "user", "content": content}],
            "stream": stream,
            "user": self._stable_user_id(user_id),
        }
        optional_parameters = {
            "temperature": self.config.get("temperature"),
            "max_tokens": self.config.get("max_tokens"),
            "presence_penalty": self.config.get("presence_penalty"),
            "frequency_penalty": self.config.get("frequency_penalty"),
        }
        for name, value in optional_parameters.items():
            if value is not None and name in model.supported_parameters:
                payload[name] = value
        if self._contains_pdf(content):
            payload["plugins"] = [
                {
                    "id": "file-parser",
                    "pdf": {"engine": str(self.config.get("pdf_engine") or "cloudflare-ai")},
                }
            ]
        return payload

    @staticmethod
    def _history_content(content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        """Drop large file data after parsing while preserving image follow-ups."""
        if not isinstance(content, list):
            return content
        sanitized: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "file":
                file_value = part.get("file") or {}
                filename = str(file_value.get("filename") or "uploaded file")
                sanitized.append({"type": "text", "text": f"[Uploaded file: {filename}]"})
            else:
                sanitized.append(part)
        return sanitized

    @staticmethod
    def _parse_completion(payload: dict[str, Any]) -> InferenceResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenRouterError("OpenRouter returned no completion choices.")
        message = choices[0].get("message") or {}
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            text = raw_content.strip()
        elif isinstance(raw_content, list):
            text = "\n".join(
                str(part.get("text") or "")
                for part in raw_content
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
            ).strip()
        else:
            text = ""
        annotations = message.get("annotations") or []
        return InferenceResult(
            text=text or "OpenRouter returned an empty text response.",
            model=str(payload.get("model") or ""),
            usage=UsageInfo.from_payload(payload.get("usage")),
            annotations=tuple(item for item in annotations if isinstance(item, dict)),
        )

    def _save_exchange(
        self,
        history: list[dict[str, Any]],
        content: str | list[dict[str, Any]],
        result: InferenceResult,
    ) -> None:
        history.append({"role": "user", "content": self._history_content(content)})
        assistant_message: dict[str, Any] = {"role": "assistant", "content": result.text}
        if result.annotations:
            assistant_message["annotations"] = list(result.annotations)
        history.append(assistant_message)
        self._trim_history(history)

    async def chat(
        self,
        api_key: str,
        user_id: int,
        session_id: str,
        model_id: str,
        content: str | list[dict[str, Any]],
    ) -> InferenceResult:
        """Send one non-streaming chat completion."""
        session_key = (user_id, session_id)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            history = self._history_for(user_id, session_id)
            payload = await self._chat_payload(
                api_key, user_id, model_id, history, content, stream=False
            )
            response = await self._request_json(
                "POST", "/chat/completions", api_key, payload=payload
            )
            result = self._parse_completion(response)
            self._save_exchange(history, content, result)
            return result

    async def chat_stream(
        self,
        api_key: str,
        user_id: int,
        session_id: str,
        model_id: str,
        content: str | list[dict[str, Any]],
    ) -> AsyncIterator[StreamUpdate]:
        """Stream a text completion, including final OpenRouter cost data."""
        session_key = (user_id, session_id)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            history = self._history_for(user_id, session_id)
            payload = await self._chat_payload(
                api_key, user_id, model_id, history, content, stream=True
            )
            answer = ""
            usage = UsageInfo()
            response_model = model_id
            annotations: tuple[dict[str, Any], ...] = ()
            try:
                async with self.client.stream(
                    "POST",
                    "/chat/completions",
                    headers=self._headers(api_key),
                    json=payload,
                ) as response:
                    if response.is_error:
                        await response.aread()
                        raise OpenRouterError(
                            f"OpenRouter returned {response.status_code}: "
                            f"{self._error_message(response)}"
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except ValueError:
                            logging.debug("Ignoring malformed OpenRouter SSE event")
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("error"):
                            error = event["error"]
                            message = error.get("message") if isinstance(error, dict) else error
                            raise OpenRouterError(f"OpenRouter stream failed: {message}")
                        response_model = str(event.get("model") or response_model)
                        if event.get("usage"):
                            usage = UsageInfo.from_payload(event["usage"])
                        choices = event.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            delta_content = delta.get("content")
                            if isinstance(delta_content, str):
                                answer += delta_content
                            raw_annotations = delta.get("annotations") or []
                            if raw_annotations:
                                annotations = tuple(
                                    item for item in raw_annotations if isinstance(item, dict)
                                )
                            if delta_content:
                                yield StreamUpdate(text=answer, done=False)
            except httpx.TimeoutException as exc:
                raise OpenRouterError("OpenRouter timed out. Please try again.") from exc
            except httpx.HTTPError as exc:
                raise OpenRouterError("Could not reach OpenRouter. Please try again.") from exc

            result = InferenceResult(
                text=answer.strip() or "OpenRouter returned an empty text response.",
                model=response_model,
                usage=usage,
                annotations=annotations,
            )
            self._save_exchange(history, content, result)
            yield StreamUpdate(
                text=result.text,
                done=True,
                model=result.model,
                usage=result.usage,
            )

    async def generate_image(
        self,
        api_key: str,
        model_id: str,
        prompt: str,
    ) -> ImageGenerationResult:
        """Generate images with OpenRouter's dedicated Image API."""
        model = await self.get_model(api_key, model_id)
        if "image" not in model.output_modalities:
            raise OpenRouterError(f"{model_id} is not an image-generation model.")
        payload: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "n": 1,
        }
        response = await self._request_json("POST", "/images", api_key, payload=payload)
        raw_images = response.get("data")
        if not isinstance(raw_images, list) or not raw_images:
            raise OpenRouterError("OpenRouter returned no generated images.")
        images: list[GeneratedImage] = []
        for raw_image in raw_images:
            if not isinstance(raw_image, dict) or not raw_image.get("b64_json"):
                continue
            try:
                image_bytes = base64.b64decode(raw_image["b64_json"], validate=True)
            except (ValueError, TypeError) as exc:
                raise OpenRouterError("OpenRouter returned invalid image data.") from exc
            images.append(
                GeneratedImage(
                    data=image_bytes,
                    media_type=str(raw_image.get("media_type") or "image/png"),
                )
            )
        if not images:
            raise OpenRouterError("OpenRouter returned no usable generated images.")
        return ImageGenerationResult(
            images=tuple(images),
            model=str(response.get("model") or model_id),
            usage=UsageInfo.from_payload(response.get("usage")),
        )


def data_url(data: bytes, mime_type: str) -> str:
    """Encode private Telegram media as an OpenRouter data URL."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_content(data: bytes, mime_type: str, prompt: str) -> list[dict[str, Any]]:
    """Build an OpenRouter image-understanding content array."""
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type}")
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url(data, mime_type)}},
    ]


def file_content(data: bytes, filename: str, mime_type: str, prompt: str) -> list[dict[str, Any]]:
    """Build an OpenRouter file content array for native/PDF processing."""
    return [
        {"type": "text", "text": prompt},
        {
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": data_url(data, mime_type),
            },
        },
    ]
