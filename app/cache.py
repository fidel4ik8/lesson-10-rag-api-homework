"""Semantic cache backed by pgvector.

ТЗ §5 originally targeted Qdrant; we use pgvector instead — same concept
(separate `cache_entries` table next to `chunks`), simpler stack since we
already run Postgres. Redis is intentionally NOT used here because Upstash
free tier has no vector search.

Pipeline contract: the caller passes the query embedding produced by
app.rag.retrieve() — there must be exactly ONE embedding call per request
(ТЗ §1, §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.db import get_conn


@dataclass(frozen=True)
class CacheHit:
    response: str
    model: str
    similarity: float
    input_tokens: int
    output_tokens: int


async def lookup(query_embedding: list[float]) -> CacheHit | None:
    """Return the closest non-expired cache entry above the threshold, else None.

    Uses the HNSW index on `cache_entries.embedding`. The expiry filter is
    applied as a WHERE clause; we rely on a periodic cleanup elsewhere to
    keep the table small (or accept the bloat for a homework workload).
    """
    settings = get_settings()
    threshold = settings.cache_similarity_threshold

    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT response, model, input_tokens, output_tokens,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM cache_entries
                WHERE expires_at > NOW()
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (query_embedding, query_embedding),
            )
            row = await cur.fetchone()

    if row is None:
        return None
    similarity = float(row[4])
    if similarity < threshold:
        return None
    return CacheHit(
        response=row[0],
        model=row[1],
        input_tokens=int(row[2] or 0),
        output_tokens=int(row[3] or 0),
        similarity=similarity,
    )


async def store(
    *,
    query: str,
    query_embedding: list[float],
    response: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.cache_ttl_seconds)

    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO cache_entries
                    (query, response, embedding, model, input_tokens, output_tokens, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (query, response, query_embedding, model, input_tokens, output_tokens, expires_at),
            )
        await conn.commit()
