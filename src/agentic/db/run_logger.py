"""Per-run DB logger with buffer-based writes and graceful degradation.

Table names are read from ``config/db.json`` (keys ``runs_table`` and
``logs_table``).  Defaults to ``pipeline_runs`` / ``pipeline_logs``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import asyncpg

from .pool import get_pool

logger = logging.getLogger(__name__)

_current_run_logger: ContextVar[RunLogger | None] = ContextVar(
    "current_run_logger", default=None
)


def get_run_logger() -> RunLogger | None:
    """Return the RunLogger for the current async context, or ``None``."""
    return _current_run_logger.get()


def _generate_run_id() -> str:
    """Generate a UUID v4 string."""
    return str(uuid.uuid4())


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------

_db_config: dict[str, str] | None = None


def _load_db_config() -> dict[str, str]:
    """Load ``config/db.json`` once and cache."""
    global _db_config
    if _db_config is not None:
        return _db_config

    config_path = Path(__file__).resolve().parents[3] / "config" / "db.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            _db_config = json.load(f)
    except FileNotFoundError:
        _db_config = {}
    return _db_config


def _runs_table() -> str:
    return _load_db_config().get("runs_table", "pipeline_runs")


def _logs_table() -> str:
    return _load_db_config().get("logs_table", "pipeline_logs")


# ------------------------------------------------------------------
# RunLogger
# ------------------------------------------------------------------


class RunLogger:
    """Manages a single pipeline run's DB row and buffered log entries."""

    _MAX_ERRORS = 20

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id: str = run_id or _generate_run_id()
        self._pool: asyncpg.Pool | None = None
        self._token: Any = None  # contextvars Token
        self._buffer: list[dict[str, Any]] = []
        self._error_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        *,
        input_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create the run row and register this logger in the ContextVar.

        ``input_metadata`` is stored as JSONB — put any project-specific
        input fields here (e.g. brand_name, product_name, …).
        """
        self._pool = await get_pool()
        self._token = _current_run_logger.set(self)

        if not self._pool:
            return
        try:
            label = os.getenv("RUN_LABEL", "production")
            runs = _runs_table()
            await self._pool.execute(
                f"""INSERT INTO {runs}
                    (id, status, label, started_at, input_metadata)
                    VALUES ($1, 'running', $2, now(), $3)""",
                self.run_id,
                label,
                json.dumps(input_metadata, ensure_ascii=False)
                if input_metadata
                else None,
            )
        except Exception:
            logger.exception("Failed to insert run row %s", self.run_id)

    async def complete(
        self,
        *,
        output_data: dict[str, Any] | None = None,
    ) -> None:
        """Mark run as completed, store outputs, flush logs.

        ``output_data`` is stored as JSONB — put any project-specific
        output here (e.g. figma_slides_content, scripter_code, …).
        """
        await self._flush_buffer()

        if self._pool:
            try:
                runs = _runs_table()
                await self._pool.execute(
                    f"""UPDATE {runs}
                        SET status = 'completed', completed_at = now(),
                            output_data = $2
                        WHERE id = $1""",
                    self.run_id,
                    json.dumps(output_data, ensure_ascii=False)
                    if output_data
                    else None,
                )
            except Exception:
                logger.exception(
                    "Failed to update run %s to completed", self.run_id
                )

        self._reset_context()

    async def fail(self, error_message: str) -> None:
        """Mark run as failed, flush logs."""
        await self._flush_buffer()

        if self._pool:
            try:
                runs = _runs_table()
                await self._pool.execute(
                    f"""UPDATE {runs}
                        SET status = 'failed', completed_at = now(),
                            error_message = $2
                        WHERE id = $1""",
                    self.run_id,
                    error_message[:4000],
                )
            except Exception:
                logger.exception(
                    "Failed to update run %s to failed", self.run_id
                )

        self._reset_context()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log(
        self,
        level: str,
        message: str,
        *,
        prompt: str | None = None,
        response: str | None = None,
    ) -> None:
        """Buffer a log entry. Flushed to DB on complete()/fail()."""
        self._buffer.append(
            {
                "level": level,
                "message": message,
                "prompt": prompt,
                "response": response,
            }
        )
        if level == "error":
            self._error_count += 1
            if self._error_count >= self._MAX_ERRORS:
                raise RuntimeError(
                    f"Too many errors ({self._error_count}), auto-stopping run {self.run_id}"
                )

    def log_llm(self, label: str, prompt: str, response: str, *, usage=None) -> None:
        """Buffer an LLM log entry. Flushed to DB on complete()/fail()."""
        from ..llm.base import _print_llm_log

        _print_llm_log(prompt, response, label)
        self._buffer.append({
            "level": "llm",
            "message": label,
            "prompt": prompt,
            "response": response,
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "cached_tokens": usage.cached_tokens if usage else None,
            "thinking_tokens": usage.thinking_tokens if usage else None,
            "tool_tokens": usage.tool_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _flush_buffer(self) -> None:
        """Bulk-write buffered logs to DB. Called only from complete()/fail()."""
        if not self._buffer or not self._pool:
            return

        entries = self._buffer[:]
        self._buffer.clear()

        try:
            logs = _logs_table()
            await self._pool.executemany(
                f"""INSERT INTO {logs}
                    (run_id, level, message, prompt, response,
                     input_tokens, output_tokens, cached_tokens,
                     thinking_tokens, tool_tokens, total_tokens)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                [
                    (
                        self.run_id,
                        e["level"],
                        e["message"],
                        e.get("prompt"),
                        e.get("response"),
                        e.get("input_tokens"),
                        e.get("output_tokens"),
                        e.get("cached_tokens"),
                        e.get("thinking_tokens"),
                        e.get("tool_tokens"),
                        e.get("total_tokens"),
                    )
                    for e in entries
                ],
            )
        except Exception as e:
            # print만 사용 (logger.exception → DBLogHandler → _flush_buffer 재귀 방지)
            print(f"[RunLogger] Failed to flush {len(entries)} log entries for run {self.run_id}: {e}")

    def _reset_context(self) -> None:
        """Remove this logger from the ContextVar."""
        if self._token:
            try:
                _current_run_logger.reset(self._token)
            except ValueError:
                _current_run_logger.set(None)
            self._token = None
