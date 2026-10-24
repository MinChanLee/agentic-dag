"""OpenAI LLM client using the openai SDK."""

from __future__ import annotations

import asyncio
import base64
import logging
import random
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from .base import BaseLLMClient, TokenUsage

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_DEFAULT_RATE_LIMIT_WAIT = 30  # fallback wait when no Retry-After header (seconds)
_SERVER_BACKOFF_BASE = 2        # 5xx exponential backoff base (seconds)
_SERVER_BACKOFF_MAX = 32        # cap for individual backoff wait (seconds)

_TRANSIENT_EXC: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
)


class OpenAIClient(BaseLLMClient):
    """Async OpenAI client using the Chat Completions API.

    Uses ``AsyncOpenAI`` for non-blocking calls.
    Includes automatic retry with backoff on rate limits (429) and transient errors.

    Args:
        api_key: OpenAI API key.
        model: Model name to use (default: ``gpt-4o``).
    """

    BACKEND: str = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def _call_with_retry(self, coro_factory):
        """Wrap an OpenAI API call with retry logic.

        Retries:
          - 429 / RateLimitError: Retry-After header or default wait
          - 5xx APIStatusError: exponential backoff + jitter
          - Transient network errors: exponential backoff + jitter
          - Other 4xx APIStatusError: raise immediately (not retryable)
        """
        last_err: BaseException | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                return await coro_factory()

            except RateLimitError as e:
                last_err = e
                retry_after = (e.response.headers or {}).get("retry-after")
                wait = float(retry_after) + 2 if retry_after else _DEFAULT_RATE_LIMIT_WAIT
                logger.warning(
                    "OpenAI 429 rate limit (attempt %d/%d), waiting %.1fs",
                    attempt + 1, _MAX_RETRIES, wait,
                )

            except APIStatusError as e:
                last_err = e
                if e.status_code < 500:
                    raise  # 4xx (except 429 handled above) are not retryable
                wait = min(_SERVER_BACKOFF_BASE * (2 ** attempt), _SERVER_BACKOFF_MAX)
                wait += random.uniform(0, 1.0)
                logger.warning(
                    "OpenAI %d error (attempt %d/%d), waiting %.1fs: %s",
                    e.status_code, attempt + 1, _MAX_RETRIES, wait, str(e),
                )

            except _TRANSIENT_EXC as e:
                last_err = e
                wait = min(_SERVER_BACKOFF_BASE * (2 ** attempt), _SERVER_BACKOFF_MAX)
                wait += random.uniform(0, 1.0)
                logger.warning(
                    "OpenAI network error %s (attempt %d/%d), waiting %.1fs",
                    type(e).__name__, attempt + 1, _MAX_RETRIES, wait,
                )

            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"OpenAI API failed after {_MAX_RETRIES} retries"
        ) from last_err

    def _extract_usage(self, response) -> TokenUsage:
        """Extract token usage from a ChatCompletion response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return TokenUsage()
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        return TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=cached,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    async def _do_generate(
        self,
        prompt: str,
        response_schema: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._call_with_retry(
            lambda: self._client.chat.completions.create(**kwargs)
        )
        text = response.choices[0].message.content or ""
        return text, self._extract_usage(response)

    async def _do_generate_with_images(
        self,
        prompt: str,
        images: list[bytes | tuple[bytes, str]],
        response_schema: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        """OpenAI multimodal: text + image(s) via base64 data URLs.

        Each image can be raw bytes (defaults to image/jpeg) or a
        (bytes, mime_type) tuple.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        for img in images:
            if isinstance(img, tuple):
                img_bytes, mime = img
            else:
                img_bytes, mime = img, "image/jpeg"

            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                    "detail": "auto",
                },
            })

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
        }
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._call_with_retry(
            lambda: self._client.chat.completions.create(**kwargs)
        )
        text = response.choices[0].message.content or ""
        return text, self._extract_usage(response)
