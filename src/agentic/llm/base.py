"""LLM client interface with automatic logging via RunLogger."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Token counts from an LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    thinking_tokens: int = 0
    tool_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Stdout helper
# ---------------------------------------------------------------------------

def _print_llm_log(prompt: str, response: str, sub_label: str = "") -> None:
    """Print LLM call log to stdout (fallback when no RunLogger is active)."""
    logs = ""
    if sub_label:
        logs += f"[{sub_label}]\n"
    logs += "\n" + "-" * 30 + " LLM PROMPT " + "-" * 30 + "\n"
    logs += prompt + "\n"
    logs += "-" * 72 + "\n"
    logs += "\n" + "-" * 30 + " LLM RESPONSE " + "-" * 30 + "\n"
    logs += (response or "") + "\n"
    logs += "-" * 74 + "\n"
    line = str({"logs": logs}) + "\n"
    sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


# ---------------------------------------------------------------------------
# Base class (Template Method pattern)
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Abstract base for LLM clients.

    Subclasses implement ``_do_generate`` (and optionally
    ``_do_generate_with_images``).

    The public ``generate*`` methods call the ``_do_*`` counterpart and
    then auto-log the prompt/response via the current ``RunLogger``.
    """

    BACKEND: str = "unknown"

    # ------------------------------------------------------------------
    # Public API (auto-logging wrappers)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        response_schema: dict | None = None,
        *,
        label: str = "",
    ) -> str:
        text, usage = await self._do_generate(prompt, response_schema)
        self._auto_log(label, prompt, text, usage)
        return text

    async def generate_with_images(
        self,
        prompt: str,
        images: list[bytes | tuple[bytes, str]],
        response_schema: dict | None = None,
        *,
        label: str = "",
    ) -> str:
        text, usage = await self._do_generate_with_images(prompt, images, response_schema)
        self._auto_log(label, prompt, text, usage)
        return text

    # ------------------------------------------------------------------
    # Abstract / overridable (subclasses implement these)
    # ------------------------------------------------------------------

    @abstractmethod
    async def _do_generate(
        self,
        prompt: str,
        response_schema: dict | None = None,
    ) -> tuple[str, TokenUsage]: ...

    async def _do_generate_with_images(
        self,
        prompt: str,
        images: list[bytes | tuple[bytes, str]],
        response_schema: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        """Default: ignore images and call ``_do_generate``."""
        return await self._do_generate(prompt, response_schema)

    # ------------------------------------------------------------------
    # Auto-log helper
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_log(label: str, prompt: str, response: str, usage: TokenUsage | None = None) -> None:
        """Log via RunLogger if available, else fall back to stdout."""
        from ..db.run_logger import get_run_logger

        rl = get_run_logger()
        if rl:
            rl.log_llm(label, prompt, response, usage=usage)
        elif label:
            # No RunLogger (e.g. local test_run.py) — stdout only
            _print_llm_log(prompt, response, label)
