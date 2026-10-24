"""Tests for OpenAIClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import RateLimitError

from agentic.llm.openai import OpenAIClient
from agentic.llm.base import TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(
    content: str,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a minimal mock ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response.usage.prompt_tokens_details = None
    return response


def _make_http_response(status_code: int, headers: dict | None = None, content: bytes = b"{}") -> httpx.Response:
    """Build an httpx.Response with a request attached (required by openai SDK)."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(
        status_code=status_code,
        headers=headers or {},
        content=content,
        request=request,
    )
    return response


def _make_rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    """Build a RateLimitError with optional Retry-After header."""
    headers = {"retry-after": retry_after} if retry_after else {}
    http_response = _make_http_response(
        429,
        headers=headers,
        content=b'{"error": {"message": "Rate limit exceeded"}}',
    )
    return RateLimitError(
        message="Rate limit exceeded",
        response=http_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


@pytest.fixture()
def client() -> OpenAIClient:
    return OpenAIClient(api_key="test-key", model="gpt-4o")


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

class TestGenerate:
    @pytest.mark.asyncio
    async def test_returns_text(self, client: OpenAIClient) -> None:
        mock_response = _make_response("Hello, world!")
        with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            result = await client.generate("Say hello")
        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_sends_user_message(self, client: OpenAIClient) -> None:
        create_mock = AsyncMock(return_value=_make_response("ok"))
        with patch.object(client._client.chat.completions, "create", new=create_mock):
            await client.generate("My prompt")
        kwargs = create_mock.call_args.kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "My prompt"}]
        assert kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_no_response_format_without_schema(self, client: OpenAIClient) -> None:
        create_mock = AsyncMock(return_value=_make_response("ok"))
        with patch.object(client._client.chat.completions, "create", new=create_mock):
            await client.generate("prompt")
        assert "response_format" not in create_mock.call_args.kwargs

    @pytest.mark.asyncio
    async def test_json_mode_with_schema(self, client: OpenAIClient) -> None:
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        create_mock = AsyncMock(return_value=_make_response('{"key": "val"}'))
        with patch.object(client._client.chat.completions, "create", new=create_mock):
            result = await client.generate("Return JSON", response_schema=schema)
        assert create_mock.call_args.kwargs["response_format"] == {"type": "json_object"}
        assert result == '{"key": "val"}'

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_string(self, client: OpenAIClient) -> None:
        mock_response = _make_response("")
        mock_response.choices[0].message.content = None
        with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            result = await client.generate("prompt")
        assert result == ""


# ---------------------------------------------------------------------------
# Multimodal
# ---------------------------------------------------------------------------

class TestGenerateWithImages:
    @pytest.mark.asyncio
    async def test_sends_text_and_image_parts(self, client: OpenAIClient) -> None:
        img_bytes = b"\xff\xd8\xff"  # JPEG magic bytes
        create_mock = AsyncMock(return_value=_make_response("I see an image"))
        with patch.object(client._client.chat.completions, "create", new=create_mock):
            result = await client.generate_with_images("What is this?", [img_bytes])

        content = create_mock.call_args.kwargs["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "What is this?"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert result == "I see an image"

    @pytest.mark.asyncio
    async def test_custom_mime_type(self, client: OpenAIClient) -> None:
        img_bytes = b"\x89PNG"
        create_mock = AsyncMock(return_value=_make_response("PNG"))
        with patch.object(client._client.chat.completions, "create", new=create_mock):
            await client.generate_with_images("Describe", [(img_bytes, "image/png")])

        content = create_mock.call_args.kwargs["messages"][0]["content"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_multiple_images(self, client: OpenAIClient) -> None:
        images = [b"img1", b"img2", b"img3"]
        create_mock = AsyncMock(return_value=_make_response("three images"))
        with patch.object(client._client.chat.completions, "create", new=create_mock):
            await client.generate_with_images("How many?", images)

        content = create_mock.call_args.kwargs["messages"][0]["content"]
        assert len(content) == 4  # 1 text + 3 images

    @pytest.mark.asyncio
    async def test_json_mode_with_images_and_schema(self, client: OpenAIClient) -> None:
        schema = {"type": "object"}
        create_mock = AsyncMock(return_value=_make_response('{"ok": true}'))
        with patch.object(client._client.chat.completions, "create", new=create_mock):
            await client.generate_with_images("Analyze", [b"img"], response_schema=schema)

        assert create_mock.call_args.kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

class TestTokenUsage:
    @pytest.mark.asyncio
    async def test_usage_extracted(self, client: OpenAIClient) -> None:
        mock_response = _make_response("ok", prompt_tokens=100, completion_tokens=50)
        with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            _, usage = await client._do_generate("prompt")

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert isinstance(usage, TokenUsage)

    @pytest.mark.asyncio
    async def test_cached_tokens_extracted(self, client: OpenAIClient) -> None:
        mock_response = _make_response("ok")
        details = MagicMock()
        details.cached_tokens = 40
        mock_response.usage.prompt_tokens_details = details

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            _, usage = await client._do_generate("prompt")

        assert usage.cached_tokens == 40


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self, client: OpenAIClient) -> None:
        mock_response = _make_response("success after retry")
        create_mock = AsyncMock(side_effect=[_make_rate_limit_error(), mock_response])

        with patch.object(client._client.chat.completions, "create", new=create_mock):
            with patch("asyncio.sleep", new=AsyncMock()):
                result = await client.generate("Test")

        assert result == "success after retry"
        assert create_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_retry_after_header(self, client: OpenAIClient) -> None:
        mock_response = _make_response("ok")
        create_mock = AsyncMock(side_effect=[_make_rate_limit_error(retry_after="10"), mock_response])
        sleep_mock = AsyncMock()

        with patch.object(client._client.chat.completions, "create", new=create_mock):
            with patch("asyncio.sleep", new=sleep_mock):
                await client.generate("Test")

        # Should wait retry_after (10) + 2 = 12s
        sleep_mock.assert_called_once_with(12.0)

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, client: OpenAIClient) -> None:
        create_mock = AsyncMock(side_effect=_make_rate_limit_error())

        with patch.object(client._client.chat.completions, "create", new=create_mock):
            with patch("asyncio.sleep", new=AsyncMock()):
                with pytest.raises(RuntimeError, match="OpenAI API failed after"):
                    await client.generate("Test")

        assert create_mock.call_count == 5  # _MAX_RETRIES

    @pytest.mark.asyncio
    async def test_does_not_retry_on_4xx(self, client: OpenAIClient) -> None:
        from openai import BadRequestError

        http_response = _make_http_response(
            400,
            content=b'{"error": {"message": "Bad request"}}',
        )
        error = BadRequestError(
            message="Bad request",
            response=http_response,
            body={"error": {"message": "Bad request"}},
        )
        create_mock = AsyncMock(side_effect=error)

        with patch.object(client._client.chat.completions, "create", new=create_mock):
            with pytest.raises(BadRequestError):
                await client.generate("Test")

        assert create_mock.call_count == 1  # no retry


# ---------------------------------------------------------------------------
# Backend identifier
# ---------------------------------------------------------------------------

class TestBackend:
    def test_backend_name(self, client: OpenAIClient) -> None:
        assert client.BACKEND == "openai"

    def test_default_model(self, client: OpenAIClient) -> None:
        assert client._model == "gpt-4o"

    def test_custom_model(self) -> None:
        c = OpenAIClient(api_key="key", model="gpt-4o-mini")
        assert c._model == "gpt-4o-mini"
