"""Client-side rate limiters for external market data APIs.

``YfinanceGate`` starts aggressive (pace/chunk/slop ceilings) and backs those knobs
down on rate-limit signals; successes recover toward the configured floors/ceilings.
``FredGate`` enforces FRED's per-minute cap. Ray workers use matching async actors in
``ray.gate_actors`` (``asyncio.sleep`` during waits; ``max_concurrency=1``).
"""
from __future__ import annotations

import abc
import asyncio
import time

from lexis_markets.logging_setup import get_logger

logger = get_logger("domain.gates")

YF_RATE_LIMIT_MARKERS = ("YFRateLimitError", "Too Many Requests", "429")


def is_yf_rate_limited(err: str) -> bool:
    return any(m in err for m in YF_RATE_LIMIT_MARKERS)


class RateGate(abc.ABC):
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._next = 0.0

    def acquire(self) -> None:
        now = time.monotonic()
        wait = self._next - now
        if wait > 0:
            time.sleep(wait)
        self._next = time.monotonic() + self.min_interval

    async def acquire_async(self) -> None:
        now = time.monotonic()
        wait = self._next - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._next = time.monotonic() + self.min_interval


class YfinanceGate:
    """Cluster-wide yfinance pacing + batch knobs (chunk size, start slop)."""

    def __init__(
        self,
        min_interval: float = 2.5,
        *,
        max_interval: float = 30.0,
        backoff_factor: float = 2.0,
        recovery_step: float = 0.1,
        chunk_size: int = 250,
        min_chunk_size: int = 30,
        start_slop_days: int = 365,
        min_start_slop_days: int = 30,
        grow_after: int = 3,
    ):
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.backoff_factor = float(backoff_factor)
        self.recovery_step = float(recovery_step)
        self.chunk_max = max(1, int(chunk_size))
        self.chunk_min = max(1, min(int(min_chunk_size), self.chunk_max))
        self.slop_max = max(0, int(start_slop_days))
        self.slop_min = max(0, min(int(min_start_slop_days), self.slop_max))
        self.grow_after = max(1, int(grow_after))
        self._interval = self.min_interval
        self._chunk = self.chunk_max
        self._slop = self.slop_max
        self._success_streak = 0
        self._next = 0.0

    @property
    def interval(self) -> float:
        return self._interval

    @property
    def chunk_size(self) -> int:
        return self._chunk

    @property
    def start_slop_days(self) -> int:
        return self._slop

    def batch_knobs(self) -> dict:
        return {
            "chunk_size": self._chunk,
            "start_slop_days": self._slop,
            "interval": self._interval,
        }

    def acquire(self) -> None:
        now = time.monotonic()
        wait = self._next - now
        if wait > 0:
            time.sleep(wait)
        self._next = time.monotonic() + self._interval

    async def acquire_async(self) -> None:
        now = time.monotonic()
        wait = self._next - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._next = time.monotonic() + self._interval

    def record_success(self) -> None:
        prev_i = self._interval
        self._interval = max(self.min_interval, self._interval - self.recovery_step)
        self._success_streak += 1
        grew = False
        if self._success_streak >= self.grow_after:
            if self._chunk < self.chunk_max:
                self._chunk = min(
                    self.chunk_max,
                    max(self._chunk + 10, int(self._chunk * 1.25)),
                )
                grew = True
            if self._slop < self.slop_max:
                self._slop = min(
                    self.slop_max,
                    max(self._slop + 10, int(self._slop * 1.25)),
                )
                grew = True
            self._success_streak = 0
        if self._interval < prev_i or grew:
            logger.info(
                "yf_gate recover interval=%.2fs chunk=%s slop=%sd",
                self._interval,
                self._chunk,
                self._slop,
            )

    def record_rate_limited(self) -> None:
        prev_i, prev_c, prev_s = self._interval, self._chunk, self._slop
        self._interval = min(self.max_interval, self._interval * self.backoff_factor)
        self._chunk = max(self.chunk_min, self._chunk // 2)
        self._slop = max(self.slop_min, self._slop // 2)
        self._success_streak = 0
        # Hold the line after a 429 before the next caller proceeds.
        self._next = max(self._next, time.monotonic() + self._interval)
        if self._interval > prev_i or self._chunk < prev_c or self._slop < prev_s:
            logger.warning(
                "yf_gate backoff interval=%.2fs chunk=%s slop=%sd",
                self._interval,
                self._chunk,
                self._slop,
            )


class FredGate(RateGate):
    # FRED documents 120 requests/minute.
    FRED_PER_MINUTE = 120.0

    def __init__(self, per_minute: float = FRED_PER_MINUTE):
        super().__init__(60.0 / per_minute)
