"""Process-local runtime counters surfaced via /health.

These are intentionally in-memory (not Redis): they are cheap signals for the
local instance, not metrics that need to be aggregated across replicas. For a
multi-instance deploy you would expose them via Prometheus and aggregate there.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class RuntimeCounters:
    active_streams: int = 0
    aborted_streams: int = 0
    completed_streams: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def stream_started(self) -> None:
        async with self._lock:
            self.active_streams += 1

    async def stream_completed(self) -> None:
        async with self._lock:
            self.active_streams = max(0, self.active_streams - 1)
            self.completed_streams += 1

    async def stream_aborted(self) -> None:
        async with self._lock:
            self.active_streams = max(0, self.active_streams - 1)
            self.aborted_streams += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "active_streams": self.active_streams,
            "aborted_streams": self.aborted_streams,
            "completed_streams": self.completed_streams,
        }


counters = RuntimeCounters()
