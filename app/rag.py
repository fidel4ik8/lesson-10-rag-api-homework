"""Vector retrieval over the chunks table."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.db import get_conn
from app.embeddings import embed_text


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    content: str
    similarity: float


async def retrieve(query: str, top_k: int | None = None) -> tuple[list[float], list[RetrievedChunk]]:
    """Embed `query` once and return (embedding, top-k chunks).

    Returning the embedding lets the caller reuse it for the semantic cache
    lookup (phase 8) — never embed twice for the same request.
    """
    settings = get_settings()
    k = top_k or settings.rag_top_k

    qv = embed_text(query)

    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (qv, qv, k),
            )
            rows = await cur.fetchall()

    chunks = [RetrievedChunk(id=r[0], content=r[1], similarity=float(r[2])) for r in rows]
    return qv, chunks
