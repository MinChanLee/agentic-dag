"""Mock LLM client – deterministic responses without an API key."""

from __future__ import annotations

import hashlib
import json

from .base import BaseLLMClient, TokenUsage


class MockLLMClient(BaseLLMClient):
    """Returns deterministic JSON based on prompt hash. No API key needed."""

    BACKEND: str = "mock"

    async def _do_generate(
        self,
        prompt: str,
        response_schema: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        tag = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return json.dumps({"result": f"mock_{tag}"}), TokenUsage()
