"""Langfuse client + lifecycle helpers.

If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are missing the SDK self-disables
and every span/observation call becomes a no-op, so the rest of the app does
not need to special-case the absence of credentials.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langfuse import Langfuse

from app.config import get_settings

log = logging.getLogger("rag.langfuse")


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    settings = get_settings()
    return Langfuse(
        public_key=settings.langfuse_public_key or None,
        secret_key=settings.langfuse_secret_key or None,
        host=settings.langfuse_host,
    )


async def shutdown_langfuse() -> None:
    """Flush pending spans on application shutdown so we don't lose data."""
    try:
        get_langfuse().flush()
    except Exception as exc:
        log.warning("langfuse flush failed: %s", exc)
