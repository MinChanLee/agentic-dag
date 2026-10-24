"""Async PostgreSQL connection pool (singleton)."""

from __future__ import annotations

import logging
import os

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool | None:
    """Return the shared pool, creating it lazily on first call.

    Returns ``None`` when ``DATABASE_URL`` is not set so the rest of the
    application can degrade gracefully (no DB logging).
    """
    global _pool
    if _pool is not None:
        return _pool

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.info("DATABASE_URL not set -- DB logging disabled")
        return None

    try:
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        logger.info("asyncpg pool created")
        return _pool
    except Exception:
        logger.exception("Failed to create asyncpg pool -- DB logging disabled")
        return None


async def close_pool() -> None:
    """Shut down the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
