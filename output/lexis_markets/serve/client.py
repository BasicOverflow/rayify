"""HTTP client for the Markets Serve API (ingest warm, cache_fill, QA)."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Iterable
from urllib.parse import quote

import requests

from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import get_logger

logger = get_logger("serve.client")

DEFAULT_WORKERS = 32
DEFAULT_TIMEOUT = 300.0


def hash_sample(rows: list[dict], n: int, *, seed: int, key: str = "series_id") -> list[dict]:
    """Stable subsample: rank by sha256(seed:id), take first n."""
    if n <= 0 or not rows:
        return []
    if len(rows) <= n:
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda r: int(
            hashlib.sha256(f"{seed}:{r[key]}".encode()).hexdigest(),
            16,
        ),
    )
    return ranked[:n]


def serve_base(cfg: MarketsConfig) -> str:
    return cfg.ray_serve_url.rstrip("/")


def serve_health(base: str, *, timeout: float = 5.0) -> bool:
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def fetch_series(
    base: str,
    series_id: str,
    start: date,
    end: date,
    *,
    revision_mode: str = "latest",
    include_rows: bool = True,
    as_of: date | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    path = quote(series_id, safe="")
    params: dict[str, str] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "revision_mode": revision_mode,
        "include_rows": "true" if include_rows else "false",
    }
    if as_of is not None:
        params["as_of"] = as_of.isoformat()
    url = f"{base}/v1/series/{path}"
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def warm_series(
    cfg: MarketsConfig,
    jobs: Iterable[tuple[str, date, date]],
    *,
    include_rows: bool = False,
    revision_mode: str = "latest",
    workers: int = DEFAULT_WORKERS,
    timeout: float = DEFAULT_TIMEOUT,
    require_health: bool = True,
) -> dict:
    """GET each series window. L3 persist happens on the Serve replica when spans write."""
    work = [(sid, start, end) for sid, start, end in jobs if start <= end]
    if not work:
        return {"ok": 0, "skip": 0, "errors": []}
    base = serve_base(cfg)
    if require_health and not serve_health(base):
        raise RuntimeError(
            f"Serve health failed at {base}/health — start the supervisor or deploy Serve first"
        )
    workers = len(work) if int(workers) <= 0 else max(1, int(workers))
    ok = 0
    skip = 0
    errors: list[str] = []

    def _one(item: tuple[str, date, date]) -> tuple[str, str | None]:
        sid, start, end = item
        try:
            fetch_series(
                base,
                sid,
                start,
                end,
                revision_mode=revision_mode,
                include_rows=include_rows,
                timeout=timeout,
            )
            return "ok", None
        except Exception as exc:
            return "skip", f"{sid}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, item) for item in work]
        for fut in as_completed(futs):
            status, err = fut.result()
            if status == "ok":
                ok += 1
            else:
                skip += 1
                if err:
                    errors.append(err)
                    logger.warning("serve warm skip %s", err)
    logger.info("serve warm ok=%s skip=%s workers=%s", ok, skip, workers)
    return {"ok": ok, "skip": skip, "errors": errors[:20], "total": len(work)}
