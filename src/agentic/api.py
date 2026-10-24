"""Public programmatic API for running pipelines.

Usage::

    from agentic import run_pipeline

    result = run_pipeline(
        "pipelines/example.yaml",
        {"input1": "시장 데이터", "input2": "경쟁사 데이터", "input3": "타겟 정보"},
    )
    print(result.outputs_only())
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .llm.base import BaseLLMClient
from .models import PipelineResult
from .runtime.executor import Executor

load_dotenv()


def _resolve_llm_client(
    llm: str | None,
    api_key: str | None,
) -> BaseLLMClient:
    """Build an LLM client from name + optional key."""
    if llm is None:
        llm = "gemini" if os.environ.get("GEMINI_API_KEY") else "mock"

    if llm == "mock":
        from .llm.mock import MockLLMClient

        return MockLLMClient()

    if llm == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in environment or pass api_key=."
            )
        from .llm.gemini import GeminiClient

        return GeminiClient(api_key=key)

    if llm == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in environment or pass api_key=."
            )
        from .llm.openai import OpenAIClient

        return OpenAIClient(api_key=key)

    raise ValueError(f"Unknown LLM backend: {llm!r}. Available: mock, gemini, openai")


def run_pipeline(
    pipeline_path: str | Path,
    inputs: dict[str, Any],
    *,
    llm: str | None = None,
    api_key: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> PipelineResult:
    """Execute a pipeline synchronously and return the result.

    Args:
        pipeline_path: Path to the pipeline YAML file.
        inputs: Dict of input key -> value.  Values can be plain data
                (str, dict, etc.) or ``FileInput`` objects for file-based inputs.
        llm: LLM backend name (``"mock"``, ``"gemini"`` or ``"openai"``).  Auto-detected if *None*.
        api_key: API key for the LLM service.  Falls back to env vars.
        llm_client: Pre-built LLM client (overrides *llm*/*api_key* if provided).

    Returns:
        ``PipelineResult`` with ``.context`` (full dict) and ``.input_nodes`` (metadata).
    """
    if llm_client is None:
        llm_client = _resolve_llm_client(llm, api_key)

    executor = Executor(llm_client=llm_client)
    return asyncio.run(executor.run(pipeline_path, inputs))


async def arun_pipeline(
    pipeline_path: str | Path,
    inputs: dict[str, Any],
    *,
    llm: str | None = None,
    api_key: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> PipelineResult:
    """Async version of :func:`run_pipeline` for use within an existing event loop."""
    if llm_client is None:
        llm_client = _resolve_llm_client(llm, api_key)

    executor = Executor(llm_client=llm_client)
    return await executor.run(pipeline_path, inputs)
