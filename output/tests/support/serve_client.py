"""Async-friendly Serve HTTP client (Ray head proxies requests)."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from lexis_markets.config import MarketsConfig

_DEFAULT_TIMEOUT = 300.0
_READY_ATTEMPTS = 24
_READY_PAUSE = 5.0


def _base_url(cfg: MarketsConfig) -> str:
    return cfg.ray_serve_url.rstrip("/")


def _head_node_id() -> str:
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        if node.get("Resources", {}).get("node:__internal_head__"):
            return node["NodeID"]
    raise RuntimeError("no alive Ray head node")


@ray.remote(num_cpus=0)
def _serve_http(
    method: str,
    path: str,
    *,
    base_url: str,
    params: dict[str, Any] | None = None,
    json_body: dict | None = None,
    timeout: float,
) -> dict:
    import requests

    url = f"{base_url.rstrip('/')}{path}"
    r = requests.request(method, url, params=params, json=json_body, timeout=timeout)
    if not r.ok:
        detail = r.text[:500] if r.text else r.reason
        raise RuntimeError(f"{method} {url} -> {r.status_code}: {detail}")
    return r.json()


async def call_serve(
    cfg: MarketsConfig,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    ref = _serve_http.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=_head_node_id(), soft=False),
    ).remote(
        method,
        path,
        base_url=_base_url(cfg),
        params=params,
        json_body=json_body,
        timeout=timeout,
    )
    return await asyncio.to_thread(ray.get, ref)


async def health(cfg: MarketsConfig, *, timeout: float = 30.0) -> dict:
    return await call_serve(cfg, "GET", "/health", timeout=timeout)


async def wait_ready(cfg: MarketsConfig) -> None:
    last: Exception | None = None
    for _ in range(_READY_ATTEMPTS):
        try:
            body = await health(cfg, timeout=10.0)
            if body.get("ok"):
                return
            last = RuntimeError(f"health not ok: {body}")
        except Exception as exc:
            last = exc
        await asyncio.sleep(_READY_PAUSE)
    raise RuntimeError(f"serve API not ready at {_base_url(cfg)}: {last}")


async def ensure_serve_api(cfg: MarketsConfig) -> None:
    from lexis_markets.serve.app import deploy_serve

    await asyncio.to_thread(deploy_serve, cfg)
    await wait_ready(cfg)


async def get_series(
    cfg: MarketsConfig,
    series_id: str,
    start: str,
    end: str,
    *,
    granularity: str = "daily",
    min_volume: float | None = None,
    revision_mode: str | None = None,
    as_of: str | None = None,
    features: list[str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    params: dict[str, Any] = {"start": start, "end": end, "granularity": granularity}
    if min_volume is not None:
        params["min_volume"] = min_volume
    if revision_mode is not None:
        params["revision_mode"] = revision_mode
    if as_of is not None:
        params["as_of"] = as_of
    if features:
        params["features"] = features
    return await call_serve(cfg, "GET", f"/v1/series/{series_id}", params=params, timeout=timeout)


async def create_dataset(
    cfg: MarketsConfig,
    spec: dict,
    *,
    sync: bool = False,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    return await call_serve(
        cfg,
        "POST",
        "/v1/datasets",
        params={"sync": sync},
        json_body=spec,
        timeout=timeout,
    )


async def get_dataset(cfg: MarketsConfig, job_id: str, *, timeout: float = 60.0) -> dict:
    return await call_serve(cfg, "GET", f"/v1/datasets/{job_id}", timeout=timeout)


class ServeClient:
    """Direct httpx client when Serve is reachable from the test runner."""

    def __init__(self, cfg: MarketsConfig):
        self.cfg = cfg
        self.base = _base_url(cfg)

    async def get_series(self, series_id: str, start: str, end: str, **kwargs) -> dict:
        return await get_series(self.cfg, series_id, start, end, **kwargs)
