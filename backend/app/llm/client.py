"""Async client for any OpenAI-compatible provider, built on httpx.

Supports chat completions (buffered and streaming), embeddings and JSON mode.
A single AsyncClient per application so the connection pool is reused.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The provider returned an error or is unreachable."""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=10.0),
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- public API ----------------------------------------------------

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        payload = self._chat_payload(messages, temperature, max_tokens, json_mode, stream=False)
        data = await self._post_json("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:  # pragma: no cover - guards malformed responses
            raise LLMError(f"Unexpected provider response: {data}") from exc

    async def chat_json(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Chat in JSON mode. Returns {} when the provider sends invalid JSON."""
        raw = await self.chat(messages, temperature=temperature, json_mode=True)
        return _loads_lenient(raw)

    async def chat_stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yields tokens as they arrive (the provider's SSE stream)."""
        payload = self._chat_payload(messages, temperature, max_tokens, False, stream=True)
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")
                raise LLMError(f"provider returned {response.status_code}: {body[:500]}")
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                token = delta.get("content")
                if token:
                    yield token

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self._settings.embed_model, "input": list(texts)}
        data = await self._post_json("/embeddings", payload)
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    # --- internals -----------------------------------------------------

    def _chat_payload(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": list(messages),
            "temperature": self._settings.llm_temperature if temperature is None else temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                response = await self._client.post(path, json=payload)
                if response.status_code in RETRYABLE_STATUS:
                    raise LLMError(f"{response.status_code}: {response.text[:300]}")
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, LLMError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt >= self._settings.llm_max_retries:
                    break
                delay = 0.5 * 2**attempt
                logger.warning("Retrying %s in %.1fs: %s", path, delay, exc)
                await asyncio.sleep(delay)
        raise LLMError(f"Request to {path} failed: {last_error}") from last_error


def _loads_lenient(raw: str) -> dict[str, Any]:
    """Parse JSON out of a model reply: strip ``` fences, take the first object."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}
