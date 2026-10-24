"""Google Gemini LLM client using the google-genai SDK."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx
from google import genai
from google.genai.errors import ClientError, ServerError

from .base import BaseLLMClient, TokenUsage

logger = logging.getLogger(__name__)

# 외부 라이브러리 INFO 로그 숨기기
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

_MAX_RETRIES = 5
_DEFAULT_RATE_LIMIT_WAIT = 40  # 429 fallback wait (seconds)
_SERVER_BACKOFF_BASE = 2       # 5xx/network exponential backoff base (seconds)
_SERVER_BACKOFF_MAX = 32       # cap for individual backoff wait (seconds)

_TRANSIENT_NETWORK_EXC: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
)


def _parse_retry_delay_from_details(details: Any) -> float | None:
    """APIError.details 에서 retryDelay 추출 (JSON 구조 접근).

    Gemini 429 응답은 다음과 같은 구조로 옴:
        {"error": {"details": [{"@type": "...RetryInfo", "retryDelay": "30s"}]}}
    """
    if not isinstance(details, dict):
        return None
    err = details.get("error", details)
    if not isinstance(err, dict):
        return None
    for d in err.get("details", []) or []:
        if not isinstance(d, dict):
            continue
        delay = d.get("retryDelay") or d.get("retry_delay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                pass
        if isinstance(delay, dict) and "seconds" in delay:
            try:
                return float(delay["seconds"])
            except (TypeError, ValueError):
                pass
    return None


class GeminiClient(BaseLLMClient):

    BACKEND: str = "gemini"
    """Async Gemini client via ``google-genai`` SDK.

    Uses ``client.aio.models.generate_content`` for non-blocking calls.
    Includes automatic retry with backoff on 429 RESOURCE_EXHAUSTED.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3-flash-preview",
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def _call_with_retry(self, coro_factory):
        """Wrap a Gemini API call with retry logic.

        Retries:
          - 429 / RESOURCE_EXHAUSTED (ClientError): retryDelay 기반 대기 후 재시도
          - 5xx (ServerError): exponential backoff + jitter
          - Transient network errors (httpx): exponential backoff + jitter
          - 그 외 4xx ClientError: 즉시 raise (재시도 불가)
        """
        last_err: BaseException | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                return await coro_factory()

            except ClientError as e:
                last_err = e
                if e.code == 429 or e.status == "RESOURCE_EXHAUSTED":
                    delay = _parse_retry_delay_from_details(e.details)
                    wait = (delay if delay is not None else _DEFAULT_RATE_LIMIT_WAIT) + 2
                    logger.warning(
                        "Gemini 429 rate limit (attempt %d/%d), waiting %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, wait, e.message or str(e),
                    )
                else:
                    # 다른 4xx (400/403 등) 은 재시도해도 의미 없음
                    raise

            except ServerError as e:
                last_err = e
                wait = min(_SERVER_BACKOFF_BASE * (2 ** attempt), _SERVER_BACKOFF_MAX)
                wait += random.uniform(0, 1.0)  # jitter
                logger.warning(
                    "Gemini %d %s (attempt %d/%d), waiting %.1fs: %s",
                    e.code, e.status, attempt + 1, _MAX_RETRIES, wait,
                    e.message or str(e),
                )

            except _TRANSIENT_NETWORK_EXC as e:
                last_err = e
                wait = min(_SERVER_BACKOFF_BASE * (2 ** attempt), _SERVER_BACKOFF_MAX)
                wait += random.uniform(0, 1.0)  # jitter
                logger.warning(
                    "Gemini network error %s (attempt %d/%d), waiting %.1fs",
                    type(e).__name__, attempt + 1, _MAX_RETRIES, wait,
                )

            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"Gemini API failed after {_MAX_RETRIES} retries"
        ) from last_err

    def _extract_usage(self, response) -> TokenUsage:
        """Extract token usage from Gemini response."""
        meta = getattr(response, "usage_metadata", None)
        if not meta:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
            thinking_tokens=getattr(meta, "thoughts_token_count", 0) or 0,
            tool_tokens=getattr(meta, "tool_use_prompt_token_count", 0) or 0,
            total_tokens=getattr(meta, "total_token_count", 0) or 0,
        )

    async def _do_generate(
        self,
        prompt: str,
        response_schema: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        config: dict | None = None
        if response_schema is not None:
            config = {
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }

        response = await self._call_with_retry(
            lambda: self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        )
        return response.text or "", self._extract_usage(response)

    async def _do_generate_with_images(
        self,
        prompt: str,
        images: list[bytes | tuple[bytes, str]],
        response_schema: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        """Gemini multimodal: text + image(s).

        Each image can be raw bytes (defaults to image/jpeg) or a
        (bytes, mime_type) tuple.
        """
        from google.genai import types

        parts = [types.Part.from_text(text=prompt)]
        for img in images:
            if isinstance(img, tuple):
                img_bytes, mime = img
            else:
                img_bytes, mime = img, "image/jpeg"
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

        config: dict | None = None
        if response_schema is not None:
            config = {
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }

        response = await self._call_with_retry(
            lambda: self._client.aio.models.generate_content(
                model=self._model,
                contents=parts,
                config=config,
            )
        )
        return response.text or "", self._extract_usage(response)
