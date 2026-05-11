"""Per-model circuit breaker.

Sliding 60s window of failure timestamps. Once a model crosses the threshold,
the breaker opens for `cooldown` seconds — calls to that model are short-
circuited (we go straight to the next model in the fallback chain).

Process-local on purpose: with multiple instances each tracks its own view,
which is fine — if one instance sees primary as flaky and another doesn't,
they will independently make the right local decision.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: int = 60,
        cooldown_seconds: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self._failures: dict[str, deque[float]] = {}
        self._open_until: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def _prune(self, model: str, now: float) -> None:
        dq = self._failures.get(model)
        if not dq:
            return
        cutoff = now - self.window_seconds
        while dq and dq[0] < cutoff:
            dq.popleft()

    async def is_open(self, model: str) -> bool:
        async with self._lock:
            now = time.monotonic()
            until = self._open_until.get(model, 0.0)
            if until > now:
                return True
            if until and until <= now:
                # Cooldown expired — reset state for this model
                self._open_until.pop(model, None)
                self._failures.pop(model, None)
            return False

    async def record_failure(self, model: str) -> None:
        async with self._lock:
            now = time.monotonic()
            dq = self._failures.setdefault(model, deque())
            dq.append(now)
            self._prune(model, now)
            if len(dq) >= self.failure_threshold:
                self._open_until[model] = now + self.cooldown_seconds

    async def record_success(self, model: str) -> None:
        async with self._lock:
            self._failures.pop(model, None)
            self._open_until.pop(model, None)


breaker = CircuitBreaker()
