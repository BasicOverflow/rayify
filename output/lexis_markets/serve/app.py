"""Ray Serve deployment for the Lexis Markets HTTP API (route ``/markets``)."""
from __future__ import annotations

import asyncio
import time
from typing import Literal
from urllib.parse import urlparse

import ray
import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, model_validator
from ray import serve
from ray.serve.config import HTTPOptions, ProxyLocation
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from lexis_markets.config import SERVE_APP_NAME, SERVE_ROUTE_PREFIX, MarketsConfig
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.runtime import OPS_REMOTE_OPTS
from lexis_markets.serve.dedupe import InFlightDedupe
from lexis_markets.serve.handlers import DEFAULT_EOD_LOOKBACK_DAYS, MarketsService

logger = get_logger("serve.app")

app = FastAPI(
    title="Lexis Markets",
    description=(
        "Stitched US equity/ETF + FRED macro OHLCV API. "
        "L3 cache-first by default (`revision_mode=latest`); pass `as_of` for live vintage "
        "collapse. Recipes: `range_panel`, `eod_snapshot`, `wide_matrix`. "
        "Universe listings and multi-series Parquet datasets "
        "(always one `dataset.parquet` per job)."
    ),
    version="1.2.0",
)


class DatasetSpec(BaseModel):
    """Body for ``POST /v1/datasets``. Output is always one Parquet part under MinIO."""

    mode: Literal["range", "range_panel", "eod_snapshot", "wide_matrix"] = Field(
        "range_panel",
        description=(
            "`range_panel` (alias `range`) — long history: one row per session per series. "
            "`eod_snapshot` — one row per matching series: last daily bar on/before "
            "`eod_date` (default last EOD target / yesterday). "
            "`wide_matrix` — dates × series_id matrix of `value_col` (default close); "
            "see `nan_policy`."
        ),
    )
    series_ids: list[str] | None = Field(
        None,
        description="Exact series ids (e.g. `equity:AAPL`, `macro:DGS10`). Omit for filter-only selection.",
    )
    asset_classes: list[str] | None = Field(
        None,
        description="Restrict to these classes: `equity`, `etf`, `macro`.",
    )
    symbols: list[str] | None = Field(
        None,
        description="Match canonical / alias tickers (case-insensitive). Ignored when `series_ids` is set.",
    )
    start: str | None = Field(
        None,
        description="Inclusive YYYY-MM-DD. **Required** for `range_panel` / `wide_matrix`.",
        examples=["2020-01-01"],
    )
    end: str | None = Field(
        None,
        description="Inclusive YYYY-MM-DD. **Required** for `range_panel` / `wide_matrix`.",
        examples=["2020-12-31"],
    )
    eod_date: str | None = Field(
        None,
        description="Snapshot day for `mode=eod_snapshot` (default: `eod_target_date()` = yesterday).",
        examples=["2020-12-30"],
    )
    eod_lookback_days: int = Field(
        DEFAULT_EOD_LOOKBACK_DAYS,
        description="Days before `eod_date` to search for the last available bar (weekends/holidays).",
        ge=1,
        le=365,
    )
    granularity: str = Field(
        "daily",
        description="Bar aggregation for panel modes: `daily`, `weekly`, or `monthly`. Forced `daily` for snapshots.",
    )
    statuses: list[str] | None = Field(
        None,
        description="Allow only these `series_meta.status` values: `ACTIVE`, `DELISTED`, `UNSUPPORTED`.",
    )
    max_gap_count: int | None = Field(
        None,
        description="Exclude series whose registry `gap_count` exceeds this value.",
    )
    max_suspicious_count: int | None = Field(
        None,
        description="Exclude series whose registry `suspicious_count` exceeds this value.",
    )
    include_partial_coverage: bool = Field(
        True,
        description="When false, drop series whose effective last day is before `end` / snapshot day.",
    )
    min_volume: float | None = Field(
        None,
        description="Drop bars with `volume` below this threshold (after stitch / before export).",
    )
    revision_mode: str | None = Field(
        None,
        description="FRED vintage collapse: `latest` (default, L3 cache) or `as_of` (point-in-time).",
    )
    as_of: str | None = Field(
        None,
        description="Point-in-time date for `revision_mode=as_of`; defaults to `end`.",
    )
    features: list[str] | None = Field(
        None,
        description=(
            "Derived columns: `sma_<n>` / `ema_<n>`, `returns`, `log_price`, "
            "`gap_mask`, `is_suspicious`."
        ),
        examples=[["sma_20", "returns", "is_suspicious"]],
    )
    nan_policy: Literal["keep", "drop_rows", "ffill"] | None = Field(
        None,
        description=(
            "wide_matrix only. `keep` (default): calendar union with NaNs; "
            "`drop_rows`: drop dates with any missing series; `ffill`: forward-fill columns."
        ),
    )
    value_col: str | None = Field(
        None,
        description="wide_matrix value column (default `close`).",
    )
    column_key: str | None = Field(
        None,
        description="wide_matrix column key (default `series_id`).",
    )

    @model_validator(mode="after")
    def _range_requires_bounds(self):
        if self.mode in ("range", "range_panel", "wide_matrix") and (not self.start or not self.end):
            raise ValueError("start and end are required when mode=range_panel or wide_matrix")
        return self


def _universe_filter_params(
    series_ids: list[str] | None,
    symbols: list[str] | None,
    asset_classes: list[str] | None,
    statuses: list[str] | None,
    max_gap_count: int | None,
    max_suspicious_count: int | None,
    include_partial_coverage: bool,
    end: str | None,
):
    return dict(
        series_ids=series_ids,
        symbols=symbols,
        asset_classes=asset_classes,
        statuses=statuses,
        max_gap_count=max_gap_count,
        max_suspicious_count=max_suspicious_count,
        include_partial_coverage=include_partial_coverage,
        end=end,
    )


@serve.deployment(name=SERVE_APP_NAME, num_replicas=1, max_ongoing_requests=10_000)
@serve.ingress(app)
class MarketsApi:
    def __init__(self, cfg_d: dict):
        cfg = MarketsConfig.from_dict(cfg_d)
        self.svc = MarketsService(cfg, dedupe=InFlightDedupe())
        logger.info("MarketsApi ready namespace=%s", cfg.ray_namespace)

    @app.get(
        "/health",
        tags=["ops"],
        summary="Liveness",
        description="Returns `{ok: true}` when the Serve replica is up.",
    )
    async def health(self):
        return {"ok": True, "service": "lexis-markets"}

    @app.get(
        "/v1/universe",
        tags=["universe"],
        summary="List series names / registry rows",
        description=(
            "Returns series identity and quality rollups **without OHLCV**. "
            "Supports status / asset_class / gap / suspicious / partial-coverage filters, "
            "plus optional `series_ids` / `symbols`. "
            "Bar-level filters (`min_volume`, `features`) belong on history / EOD / dataset routes, "
            "not on this listing."
        ),
    )
    async def list_universe(
        self,
        series_ids: list[str] | None = Query(None, description="Exact series ids to include"),
        symbols: list[str] | None = Query(None, description="Ticker / alias match (case-insensitive)"),
        asset_classes: list[str] | None = Query(
            None, description="`equity`, `etf`, and/or `macro`"
        ),
        statuses: list[str] | None = Query(
            None, description="`ACTIVE`, `DELISTED`, and/or `UNSUPPORTED`"
        ),
        max_gap_count: int | None = Query(None, description="Max registry gap_count"),
        max_suspicious_count: int | None = Query(None, description="Max registry suspicious_count"),
        include_partial_coverage: bool = Query(
            True,
            description="When false, require effective last day >= `end`",
        ),
        end: str | None = Query(
            None,
            description="Coverage end date for partial-coverage gate (default: last EOD target)",
        ),
        limit: int | None = Query(None, ge=1, description="Page size"),
        offset: int = Query(0, ge=0, description="Page offset"),
    ):
        return await asyncio.to_thread(
            self.svc.list_universe,
            **_universe_filter_params(
                series_ids,
                symbols,
                asset_classes,
                statuses,
                max_gap_count,
                max_suspicious_count,
                include_partial_coverage,
                end,
            ),
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/v1/eod/{series_id:path}",
        tags=["eod"],
        summary="Last EOD bar for one series",
        description=(
            "Returns the last daily stitched bar on or before `eod_date` "
            "(default: yesterday / EOD target). Uses a lookback window so weekends "
            "and holidays still resolve to the prior session. "
            "Response includes `bar` (object) and `rows` (one-element list)."
        ),
    )
    async def get_series_eod(
        self,
        series_id: str,
        eod_date: str | None = Query(
            None, description="Snapshot day YYYY-MM-DD (default: eod_target_date)"
        ),
        eod_lookback_days: int = Query(
            DEFAULT_EOD_LOOKBACK_DAYS,
            ge=1,
            le=365,
            description="Days before eod_date to search for a bar",
        ),
        min_volume: float | None = Query(None, description="Drop bar if volume below this"),
        revision_mode: str | None = Query(
            "latest",
            description="FRED vintage collapse: `as_of` or `latest` (default latest for snapshots)",
        ),
        features: list[str] | None = Query(
            None, description="Optional `sma_<n>` / `ema_<n>` on the snapshot bar"
        ),
        statuses: list[str] | None = Query(None, description="Universe status allow-list"),
        max_gap_count: int | None = Query(None),
        max_suspicious_count: int | None = Query(None),
        include_partial_coverage: bool = Query(
            True,
            description="When false, series must cover through the snapshot day",
        ),
    ):
        return await asyncio.to_thread(
            self.svc.query_series_eod,
            series_id,
            eod_date=eod_date,
            eod_lookback_days=eod_lookback_days,
            min_volume=min_volume,
            revision_mode=revision_mode,
            features=features,
            statuses=statuses,
            max_gap_count=max_gap_count,
            max_suspicious_count=max_suspicious_count,
            include_partial_coverage=include_partial_coverage,
        )

    @app.post(
        "/v1/datasets",
        tags=["datasets"],
        summary="Create multi-series Parquet dataset",
        description=(
            "Queues (or sync-builds) a dataset job. Result is **one** "
            "`layer_3/outputs/{job_id}/dataset.parquet` plus `manifest.json`. "
            "Recipes: `range_panel` (alias `range`), `eod_snapshot`, `wide_matrix` "
            "(dates × series; `nan_policy=keep|drop_rows|ffill`). "
            "Default `revision_mode=latest` (L3)."
        ),
    )
    async def create_dataset(
        self,
        spec: DatasetSpec,
        sync: bool = Query(
            False,
            description="When true, build in-request and return the completed manifest",
        ),
    ):
        body = spec.model_dump()
        return await asyncio.to_thread(self.svc.create_dataset, body, sync=sync)

    @app.get(
        "/v1/datasets/{job_id}",
        tags=["datasets"],
        summary="Dataset job status",
        description="Poll job status, row/series counts, S3 prefix, and `files` when complete.",
    )
    async def get_dataset(self, job_id: str):
        return await asyncio.to_thread(self.svc.get_dataset_job, job_id)

    @app.get(
        "/v1/series/{series_id:path}",
        tags=["series"],
        summary="Stitched history for one series",
        description=(
            "Returns stitched OHLCV rows for `[start, end]` plus a window quality block. "
            "For a single last-EOD bar use `GET /v1/eod/{series_id}` instead."
        ),
    )
    async def get_series(
        self,
        series_id: str,
        start: str = Query(..., description="Inclusive start YYYY-MM-DD", examples=["2020-01-01"]),
        end: str = Query(..., description="Inclusive end YYYY-MM-DD", examples=["2020-12-31"]),
        granularity: str = Query(
            "daily",
            description="`daily`, `weekly`, or `monthly`",
        ),
        min_volume: float | None = Query(None, description="Drop bars below this volume"),
        revision_mode: str | None = Query(
            None,
            description="FRED vintage collapse: `as_of` (default) or `latest`",
        ),
        as_of: str | None = Query(
            None,
            description="Point-in-time date for revision_mode=as_of; defaults to end",
        ),
        features: list[str] | None = Query(
            None,
            description="Derived columns: sma_<n> or ema_<n> on close (e.g. sma_20, ema_50)",
        ),
        statuses: list[str] | None = Query(
            None,
            description="Allow only these series_meta statuses: ACTIVE, DELISTED, UNSUPPORTED",
        ),
        max_gap_count: int | None = Query(
            None,
            description="Exclude series whose registry gap_count exceeds this value",
        ),
        max_suspicious_count: int | None = Query(
            None,
            description="Exclude series whose registry suspicious_count exceeds this value",
        ),
        include_partial_coverage: bool = Query(
            True,
            description="When false, exclude series that end before the request end date",
        ),
        include_rows: bool = Query(
            True,
            description="When false, omit OHLCV rows (quality + meta only). Ingest/cache_fill use this.",
        ),
    ):
        return await asyncio.to_thread(
            self.svc.query_series,
            series_id,
            start,
            end,
            granularity=granularity,
            min_volume=min_volume,
            revision_mode=revision_mode,
            as_of=as_of,
            features=features,
            statuses=statuses,
            max_gap_count=max_gap_count,
            max_suspicious_count=max_suspicious_count,
            include_partial_coverage=include_partial_coverage,
            include_rows=include_rows,
        )


def _serve_http_options(cfg: MarketsConfig) -> HTTPOptions:
    parsed = urlparse(cfg.ray_serve_url)
    return HTTPOptions(
        host="0.0.0.0",
        port=parsed.port or 8000,
        location=ProxyLocation.EveryNode,
    )


def _head_node_id() -> str:
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        if node.get("Resources", {}).get("node:__internal_head__"):
            return node["NodeID"]
    raise RuntimeError("no alive Ray head node")


@ray.remote(num_cpus=0, **OPS_REMOTE_OPTS)
def _restart_serve_http_on_head(port: int) -> None:
    import time as _time

    from ray import serve as _serve
    from ray.serve.config import HTTPOptions as _HTTPOptions, ProxyLocation as _ProxyLocation

    _serve.shutdown()
    _time.sleep(8)
    _serve.start(
        http_options=_HTTPOptions(
            host="0.0.0.0",
            port=port,
            location=_ProxyLocation.EveryNode,
        )
    )
    _time.sleep(3)


def _serve_proxy_reachable(cfg: MarketsConfig, *, timeout: float = 3.0) -> bool:
    """True when the HTTP proxy accepts connections (app may not be deployed yet)."""
    url = f"{cfg.ray_serve_url.rstrip('/')}/health"
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        return False
    except Exception:
        return True


def _serve_http_reachable(cfg: MarketsConfig, *, timeout: float = 3.0) -> bool:
    url = f"{cfg.ray_serve_url.rstrip('/')}/health"
    try:
        r = requests.get(url, timeout=timeout)
        return r.ok and r.json().get("ok") is True
    except Exception:
        return False


def _ensure_serve_http(cfg: MarketsConfig) -> None:
    if _serve_proxy_reachable(cfg):
        return
    opts = _serve_http_options(cfg)
    logger.info(
        "serve HTTP not reachable at %s; restarting on head with host=0.0.0.0 port=%s",
        cfg.ray_serve_url,
        opts.port,
    )
    ref = _restart_serve_http_on_head.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=_head_node_id(),
            soft=False,
        ),
    ).remote(opts.port)
    ray.get(ref)
    deadline = time.time() + 90
    while time.time() < deadline:
        if _serve_proxy_reachable(cfg, timeout=5.0):
            return
        time.sleep(2)
    raise RuntimeError(f"serve HTTP proxy not reachable at {cfg.ray_serve_url} after restart")


def deploy_serve(cfg: MarketsConfig, *, retries: int = 3):
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            _ensure_serve_http(cfg)
            try:
                serve.delete(SERVE_APP_NAME, _blocking=True)
                time.sleep(3)
            except Exception:
                pass
            handle = serve.run(
                MarketsApi.bind(cfg.to_dict()),
                name=SERVE_APP_NAME,
                route_prefix=SERVE_ROUTE_PREFIX,
            )
            logger.info("serve deployed route=%s url=%s", SERVE_ROUTE_PREFIX, cfg.ray_serve_url)
            return handle
        except TimeoutError as e:
            # Ray Serve may report the app ready then still time out waiting on proxies.
            if _serve_http_reachable(cfg, timeout=5.0):
                logger.warning(
                    "serve.run proxy wait timed out attempt %s/%s but /health ok; continuing",
                    attempt + 1,
                    retries,
                )
                return None
            last_err = e
            logger.warning("serve deploy attempt %s/%s failed: %s", attempt + 1, retries, e)
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            last_err = e
            logger.warning("serve deploy attempt %s/%s failed: %s", attempt + 1, retries, e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"deploy_serve failed after {retries} attempts") from last_err
