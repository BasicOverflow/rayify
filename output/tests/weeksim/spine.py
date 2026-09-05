"""Drive supervisor cron enqueue + ``jobs.dispatch`` — same spine as production."""
from __future__ import annotations

from lexis_markets.config import MarketsConfig
from lexis_markets.jobs.dispatch import submit_loop
from lexis_markets.jobs.queue import SupervisorState
from lexis_markets.ray.gate_client import GateHandles


def drain_queue(
    cfg: MarketsConfig,
    state: SupervisorState,
    *,
    gates: GateHandles,
    max_rounds: int = 64,
) -> int:
    """Run ``submit_loop`` until the pending queue is empty. Returns jobs started."""
    started = 0
    for _ in range(max_rounds):
        n = submit_loop(cfg, state, gates=gates, max_in_flight_jobs=4)
        if n == 0:
            return started
        started += n
    raise RuntimeError(f"pending queue not drained after {max_rounds} submit_loop rounds")
