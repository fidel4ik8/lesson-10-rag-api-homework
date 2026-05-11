"""Token bucket rate limit, backed by Redis.

Implementation is a fixed-window counter — chosen over a true sliding bucket
because ТЗ §4 mandates plain INCR/EXPIRE (Upstash REST does not support Lua).
The window key is the minute boundary (`floor(now/60)`), so the bucket
refills implicitly when the wall clock ticks into the next minute.

Trade-off: a request at 12:00:59 followed by another at 12:01:00 sees the
counter reset, which can let two minutes' worth of traffic through in a
single second window. For a homework Q&A bot this is acceptable. Production
systems usually layer a sliding-window or true-bucket implementation on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import redis.asyncio as aioredis

from app.config import get_settings

WINDOW_SECONDS = 60
KEY_TTL_SECONDS = 120  # outlives the window so concurrent requests still see it


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    used: int
    limit: int
    retry_after: int   # seconds until the next minute boundary


def _window_key(api_key: str, now: float) -> tuple[str, int]:
    window_start = int(now) // WINDOW_SECONDS * WINDOW_SECONDS
    key = f"rl:{api_key}:{window_start}"
    retry_after = WINDOW_SECONDS - (int(now) - window_start)
    return key, retry_after


async def check(api_key: str, limit: int) -> RateLimitDecision:
    """Pre-flight: read current usage; allow if strictly under the limit.

    A simple GET — no mutation. We charge the actual tokens later via
    `consume()` to avoid pre-paying for output we have not generated yet.
    """
    r = get_redis()
    now = time.time()
    key, retry_after = _window_key(api_key, now)
    raw = await r.get(key)
    used = int(raw) if raw else 0
    return RateLimitDecision(
        allowed=used < limit,
        used=used,
        limit=limit,
        retry_after=retry_after,
    )


async def consume(api_key: str, tokens: int) -> int:
    """Add `tokens` to the current minute's bucket and return new total."""
    r = get_redis()
    now = time.time()
    key, _ = _window_key(api_key, now)
    pipe = r.pipeline()
    pipe.incrby(key, tokens)
    pipe.expire(key, KEY_TTL_SECONDS)
    incr_result, _ = await pipe.execute()
    return int(incr_result)
