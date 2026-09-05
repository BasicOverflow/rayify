"""Coalesce concurrent identical Ray submissions on one Serve replica.

Cache builds use ``(series_id, start, end)``. Series bar queries include
revision mode / as_of / granularity / features so concurrent HTTP callers share
one in-flight remote instead of duplicate work.
"""
from __future__ import annotations

import threading
from typing import Callable, Hashable

import ray

from lexis_markets.logging_setup import get_logger

logger = get_logger("serve.dedupe")


class InFlightDedupe:
    """In-flight map from a hashable request key to a Ray ObjectRef."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[Hashable, ray.ObjectRef] = {}

    def get_or_submit(
        self, key: Hashable, submit_fn: Callable[[], ray.ObjectRef]
    ) -> ray.ObjectRef:
        with self._lock:
            ref = self._pending.get(key)
            if ref is None:
                ref = submit_fn()
                self._pending[key] = ref
                logger.debug("dedupe submit key=%s", key)
            else:
                logger.debug("dedupe reuse key=%s", key)
            return ref

    def release(self, key: Hashable, ref: ray.ObjectRef) -> None:
        with self._lock:
            if self._pending.get(key) is ref:
                del self._pending[key]

    def wait(self, key: Hashable, submit_fn: Callable[[], ray.ObjectRef]):
        ref = self.get_or_submit(key, submit_fn)
        try:
            return ray.get(ref)
        finally:
            self.release(key, ref)
