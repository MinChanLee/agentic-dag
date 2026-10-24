"""Generic prompt operator – the default workhorse."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from ..llm.base import BaseLLMClient
from .base import BaseOperator

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

# Language-specific section-1 / section-2 keyword pairs.
# Each tuple: (section_1_keyword, section_2_keyword, key_1_label, key_2_label)
_SECTION_MARKERS: list[tuple[str, str, str, str]] = [
    # English
    (
        "1. Detailed Explanation of Reasoning Process:",
        "2. Final Results:",
        "1. Detailed Explanation of Reasoning Process",
        "2. Final Results",
    ),
    # Korean
    (
        "1. 추론 과정 상세 설명:",
        "2. 최종 결과:",
        "1. 추론 과정 상세 설명",
        "2. 최종 결과",
    ),
]


def _extract_json(raw: str) -> str:
    """Strip markdown code fences (```json ... ```) if present."""
    m = _CODE_BLOCK_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


_MARKDOWN_STRIP_RE = re.compile(r"[`*]")


def _normalize_line(line: str) -> str:
    """Remove backticks and asterisks from a line."""
    return _MARKDOWN_STRIP_RE.sub("", line)


def _parse_custom_format(raw: str) -> str | None:
    """Parse the '1. …  2. …' format and return only the Final Results section.

    Handles markdown-formatted headings like **2. 최종 결과:** by stripping
    backticks and asterisks before matching.
    """
    for *_, key2 in _SECTION_MARKERS:
        # Find the line containing the section-2 keyword (ignoring markdown formatting)
        lines = raw.split("\n")
        target_line_idx = None
        for i, line in enumerate(lines):
            if key2 in _normalize_line(line):
                target_line_idx = i
                break
        if target_line_idx is None:
            continue
        normalized_first_line = _normalize_line(lines[target_line_idx])
        # Split on the keyword within the normalized first line to get trailing content
        _, _, after_key = normalized_first_line.partition(key2)
        # Strip the colon and whitespace that may follow the keyword
        after_key = after_key.lstrip(":").strip()
        # Combine any trailing content on the same line with subsequent lines
        subsequent = "\n".join(lines[target_line_idx + 1 :]).strip()
        if after_key and subsequent:
            section2 = after_key + "\n" + subsequent
        else:
            section2 = after_key or subsequent
        return section2.strip() if section2 else None
    return None


_MAX_RETRIES = 4


class GenericPromptOperator(BaseOperator):
    """Render → LLM call → JSON parse (or custom format) → pydantic validate → dict."""

    VERSION: str = "1.0"

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    async def execute(
        self,
        rendered_prompt: str,
        output_schema: type[BaseModel] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_raw: str = ""

        node_id = (context or {}).get("_node_id", "")
        for attempt in range(_MAX_RETRIES):
            sub_label = f"{node_id}_attempt_{attempt + 1}" if node_id else (f"attempt_{attempt + 1}" if _MAX_RETRIES > 1 else "")
            raw = await self.llm_client.generate(rendered_prompt, label=sub_label)
            last_raw = raw or ""

            if not raw:
                # LLM returned empty/None (e.g. safety filter) — retry
                continue

            cleaned = _extract_json(raw)

            parsed = None
            # 1. Try JSON parsing first
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                # 2. If JSON fails, try the custom text format
                parsed = _parse_custom_format(raw)

            if parsed is not None:
                break  # success — exit retry loop
            # else: retry

        if parsed is None:
            raise ValueError(
                f"LLM returned invalid format after {_MAX_RETRIES} attempts. "
                f"Expected JSON or '1. …  2. …' structure.\n"
                f"Last raw response:\n{last_raw}"
            )

        if output_schema is not None:
            validated = output_schema(**parsed)
            return validated.model_dump()

        return parsed
