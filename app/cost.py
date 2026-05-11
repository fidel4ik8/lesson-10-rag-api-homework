"""Cost record persistence and usage aggregations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app.db import get_conn


@dataclass(frozen=True)
class CostRecord:
    request_id: uuid.UUID
    api_key: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    ttft_ms: int | None
    cache_hit: bool
    fallback_used: bool
    output_filtered: bool = False


async def insert_cost_record(rec: CostRecord) -> None:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO cost_records (
                    request_id, api_key, model,
                    input_tokens, output_tokens, cost_usd,
                    latency_ms, ttft_ms, cache_hit, fallback_used, output_filtered
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(rec.request_id),
                    rec.api_key,
                    rec.model,
                    rec.input_tokens,
                    rec.output_tokens,
                    rec.cost_usd,
                    rec.latency_ms,
                    rec.ttft_ms,
                    rec.cache_hit,
                    rec.fallback_used,
                    rec.output_filtered,
                ),
            )
        await conn.commit()


def _today_start_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def usage_today(api_key: str) -> dict:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    COUNT(*) AS requests,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens,
                    COALESCE(SUM(cost_usd), 0) AS cost_usd
                FROM cost_records
                WHERE api_key = %s AND created_at >= %s
                """,
                (api_key, _today_start_utc()),
            )
            row = await cur.fetchone()
    return {
        "requests": int(row[0]),
        "input_tokens": int(row[1]),
        "output_tokens": int(row[2]),
        "tokens": int(row[3]),
        "cost_usd": float(row[4]),
    }


async def usage_breakdown(api_key: str, hours: int = 24) -> dict:
    """Per-model breakdown plus aggregate cache/fallback rates and latency
    percentiles. Window: last `hours` hours (default 24)."""
    since_clause = f"created_at >= NOW() - INTERVAL '{int(hours)} hours'"

    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT
                    model,
                    COUNT(*) AS requests,
                    SUM(input_tokens) AS input_tokens,
                    SUM(output_tokens) AS output_tokens,
                    SUM(cost_usd) AS cost_usd,
                    AVG(latency_ms)::int AS avg_latency_ms,
                    PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
                FROM cost_records
                WHERE api_key = %s AND {since_clause}
                GROUP BY model
                ORDER BY requests DESC
                """,
                (api_key,),
            )
            per_model_rows = await cur.fetchall()

            await cur.execute(
                f"""
                SELECT
                    COUNT(*)                                                AS total,
                    SUM(CASE WHEN cache_hit       THEN 1 ELSE 0 END)        AS cache_hits,
                    SUM(CASE WHEN fallback_used   THEN 1 ELSE 0 END)        AS fallbacks,
                    SUM(CASE WHEN output_filtered THEN 1 ELSE 0 END)        AS filtered,
                    AVG(latency_ms)::int                                    AS avg_latency_ms,
                    PERCENTILE_DISC(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
                    PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
                    AVG(ttft_ms)::int                                       AS avg_ttft_ms
                FROM cost_records
                WHERE api_key = %s AND {since_clause}
                """,
                (api_key,),
            )
            agg = await cur.fetchone()

    total = int(agg[0] or 0)
    return {
        "window_hours": hours,
        "total_requests": total,
        "cache_hit_rate": (int(agg[1] or 0) / total) if total else 0.0,
        "fallback_rate": (int(agg[2] or 0) / total) if total else 0.0,
        "output_filtered_rate": (int(agg[3] or 0) / total) if total else 0.0,
        "avg_latency_ms": int(agg[4] or 0),
        "p50_latency_ms": int(agg[5] or 0),
        "p95_latency_ms": int(agg[6] or 0),
        "avg_ttft_ms": int(agg[7] or 0),
        "by_model": [
            {
                "model": r[0],
                "requests": int(r[1]),
                "input_tokens": int(r[2] or 0),
                "output_tokens": int(r[3] or 0),
                "cost_usd": float(r[4] or 0),
                "avg_latency_ms": int(r[5] or 0),
                "p95_latency_ms": int(r[6] or 0),
            }
            for r in per_model_rows
        ],
    }
