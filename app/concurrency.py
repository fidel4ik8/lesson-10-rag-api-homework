"""Process-wide LLM concurrency limit.

asyncio.Semaphore caps how many LLM calls can be in flight simultaneously
across all requests. Above the cap, additional callers wait inside `async with`
until a slot frees up. Bounds memory and prevents us from being throttled by
upstream providers under load spikes.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from app.config import get_settings


@lru_cache(maxsize=1)
def get_llm_semaphore() -> asyncio.Semaphore:
    settings = get_settings()
    return asyncio.Semaphore(settings.max_concurrent_llm_calls)
