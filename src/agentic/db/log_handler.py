"""Custom logging.Handler that routes Python logger calls to the per-run DB."""

from __future__ import annotations

import logging

from .run_logger import get_run_logger


class DBLogHandler(logging.Handler):
    """Captures ``logger.info()`` / ``.warning()`` / ``.error()`` calls and
    buffers them in the current RunLogger for later DB flush.

    This handler is sync-safe: it only appends to an in-memory list.
    The actual DB write happens when ``RunLogger.complete()`` or
    ``RunLogger.fail()`` calls ``_flush_buffer()``.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rl = get_run_logger()
            if rl is None:
                return  # no active run — skip DB buffering
            level = record.levelname.lower()
            message = self.format(record)
            rl.log(level=level, message=message)
        except Exception:
            self.handleError(record)
