"""Batch submission with ``ray.wait`` backpressure.

``run_batches`` keeps at most ``max_in_flight`` tasks pending and drains results
one at a time so large ingest jobs do not flood the object store.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import ray

from lexis_markets.logging_setup import get_logger

logger = get_logger("jobs.scheduler")


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    logger.info("%s %.1fs", label, time.perf_counter() - t0)


@dataclass
class TaskShape:
    batch_size: int
    max_in_flight: int = 0


def plan_task_resources(*, batch_size: int = 50, max_in_flight: int = 0, **_) -> TaskShape:
    logger.debug("plan_task_resources batch_size=%s max_in_flight=%s", batch_size, max_in_flight)
    return TaskShape(batch_size=batch_size, max_in_flight=max_in_flight)


def chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_batches(label: str, items: list, submit_fn: Callable, shape: TaskShape) -> list:
    batches = chunks(items, shape.batch_size)
    total_batches = len(batches)
    total_items = len(items)
    results: list = []
    pending: list = []
    next_i = 0
    done_batches = 0
    done_items = 0
    t0 = time.perf_counter()
    cap = shape.max_in_flight if shape.max_in_flight > 0 else len(batches)
    while next_i < len(batches) or pending:
        while next_i < len(batches) and len(pending) < cap:
            pending.append(submit_fn(batches[next_i]))
            next_i += 1
        ready, pending = ray.wait(pending, num_returns=1)
        batch = ray.get(ready[0])
        results.append(batch)
        done_batches += 1
        done_items += len(batch) if isinstance(batch, list) else 1
        elapsed = time.perf_counter() - t0
        rate = done_items / elapsed if elapsed else 0
        logger.info(
            "%s batch %s/%s items %s/%s rate=%.1f/s elapsed=%.0fs",
            label, done_batches, total_batches, done_items, total_items, rate, elapsed,
        )
    elapsed = time.perf_counter() - t0
    logger.info("%s done %.1fs batches=%s", label, elapsed, total_batches)
    return results
