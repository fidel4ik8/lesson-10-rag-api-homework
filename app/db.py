from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

_pool: AsyncConnectionPool | None = None


async def _configure_connection(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)


async def init_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    settings = get_settings()
    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        configure=_configure_connection,
        open=False,
    )
    await _pool.open()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_conn() -> AsyncIterator[psycopg.AsyncConnection]:
    pool = await init_pool()
    async with pool.connection() as conn:
        yield conn


# Sync helpers for the indexing script (no async needed there)
def sync_connect() -> psycopg.Connection:
    from pgvector.psycopg import register_vector

    settings = get_settings()
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn
