"""Ray client helpers for gate actor handles."""

from __future__ import annotations

from dataclasses import dataclass

import ray

from lexis_markets.config import MarketsConfig
from lexis_markets.domain.gates import is_yf_rate_limited
from lexis_markets.ray.markets_actors import get_fred_gate_actor, get_yf_gate_actor

__all__ = [
    "GateHandles",
    "is_yf_rate_limited",
    "resolve_gates",
    "refresh_gates",
    "yf_acquire",
    "yf_batch_knobs",
    "yf_record_rate_limited",
    "yf_record_success",
]


@dataclass(frozen=True)
class GateHandles:
    yf_gate: object
    fred_gate: object


def resolve_gates(cfg: MarketsConfig, gates: GateHandles | None = None) -> GateHandles:
    if gates is not None:
        return gates
    return GateHandles(
        yf_gate=get_yf_gate_actor(cfg),
        fred_gate=get_fred_gate_actor(cfg),
    )


def refresh_gates(cfg: MarketsConfig) -> GateHandles:
    from lexis_markets.ray.markets_actors import _clear_gate_cache

    _clear_gate_cache()
    return resolve_gates(cfg)


def yf_acquire(yf_gate) -> None:
    ray.get(yf_gate.acquire.remote())


def yf_batch_knobs(yf_gate) -> dict:
    return dict(ray.get(yf_gate.batch_knobs.remote()))


def yf_record_success(yf_gate) -> None:
    ray.get(yf_gate.record_success.remote())


def yf_record_rate_limited(yf_gate) -> None:
    ray.get(yf_gate.record_rate_limited.remote())
