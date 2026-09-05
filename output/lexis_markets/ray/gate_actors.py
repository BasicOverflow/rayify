"""Ray actor wrappers around ``domain.gates`` for cluster-wide rate limiting."""
from __future__ import annotations

import ray

from lexis_markets.fred.alfred import AdaptiveWindowDays
from lexis_markets.domain.gates import FredGate, YfinanceGate
from lexis_markets.logging_setup import configure_logging

configure_logging()


@ray.remote(max_concurrency=1)
class YfinanceGateActor:
    def __init__(
        self,
        min_interval: float = 2.5,
        max_interval: float = 30.0,
        backoff_factor: float = 2.0,
        recovery_step: float = 0.1,
        chunk_size: int = 150,
        min_chunk_size: int = 30,
        start_slop_days: int = 150,
        min_start_slop_days: int = 30,
    ):
        self._gate = YfinanceGate(
            min_interval,
            max_interval=max_interval,
            backoff_factor=backoff_factor,
            recovery_step=recovery_step,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            start_slop_days=start_slop_days,
            min_start_slop_days=min_start_slop_days,
        )

    async def acquire(self) -> None:
        await self._gate.acquire_async()

    async def record_success(self) -> None:
        self._gate.record_success()

    async def record_rate_limited(self) -> None:
        self._gate.record_rate_limited()

    def interval(self) -> float:
        return self._gate.interval

    def batch_knobs(self) -> dict:
        return self._gate.batch_knobs()


@ray.remote(max_concurrency=1)
class FredGateActor:
    def __init__(
        self,
        per_minute: float = FredGate.FRED_PER_MINUTE,
        initial_window_days: int = 1825,
        min_window_days: int = 31,
    ):
        self._gate = FredGate(per_minute)
        self._windows = AdaptiveWindowDays(
            current=initial_window_days,
            min_days=min_window_days,
            max_days=initial_window_days,
        )

    async def acquire(self) -> None:
        await self._gate.acquire_async()

    def pace_per_minute(self) -> float:
        return 60.0 / self._gate.min_interval

    def preferred_window_days(self) -> int:
        return self._windows.preferred()

    def report_window_fetch(self, span_days: int, bisects: int) -> int:
        return self._windows.observe(span_days=span_days, bisects=bisects)
