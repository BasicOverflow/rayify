"""FRED / ALFRED HTTP client (cluster-wide rate gate required)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from uuid import uuid4

import pandas as pd
import ray
import requests

from lexis_markets.lake import LakeStore, L1Writer
from lexis_markets.config import MarketsConfig
from lexis_markets.fred.alfred import (
    ALFRED_OUTPUT_TYPE_VINTAGE,
    FRED_HISTORY_FLOOR,
    FRED_OBS_URL,
    FRED_PAGE_LIMIT,
    FRED_VINTAGE_DATES_URL,
    FetchStats,
    filter_vintage_dates,
    observations_to_df,
    plan_ingest_windows,
)
from lexis_markets.logging_setup import get_logger

logger = get_logger("fred.client")

FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"
_GATEWAY = frozenset({502, 503, 504})
_RETRYABLE_RATE = frozenset({429})
_SPLITTABLE = frozenset({400, 502, 503, 504})
_RATE_ATTEMPTS = 3
_MIN_WINDOW_DAYS_DEFAULT = 31
_MAX_WINDOW_DAYS_DEFAULT = 1825  # ~5y opening split


@dataclass(frozen=True)
class FredVintagePlan:
    vintage_start: date
    vintage_end: date
    vintage_dates: list[date]


def _require_fred_gate(rate_gate) -> None:
    if rate_gate is None:
        raise ValueError(
            "fred rate_gate required; pass the FredGateActor handle from supervisor bootstrap or CLI"
        )


def gate_preferred_window_days(rate_gate, default: int) -> int:
    _require_fred_gate(rate_gate)
    return int(ray.get(rate_gate.preferred_window_days.remote()))


def gate_report_window_fetch(rate_gate, *, span_days: int, bisects: int) -> int:
    _require_fred_gate(rate_gate)
    return int(ray.get(rate_gate.report_window_fetch.remote(span_days, bisects)))


def _fred_api_get(
    url: str,
    params: dict,
    rate_gate,
    *,
    fail_fast_gateway: bool = False,
) -> requests.Response:
    """Rate-gated GET. Gateway 502/503/504: fail immediately when ``fail_fast_gateway``
    (observation windows → bisect). Metadata calls still retry briefly.
    """
    _require_fred_gate(rate_gate)
    resp: requests.Response | None = None
    attempts = 1 if fail_fast_gateway else _RATE_ATTEMPTS
    for attempt in range(attempts):
        ray.get(rate_gate.acquire.remote())
        resp = requests.get(url, params=params, timeout=120)
        if fail_fast_gateway and resp.status_code in _GATEWAY:
            resp.raise_for_status()
        if resp.status_code not in (_RETRYABLE_RATE | _GATEWAY):
            return resp
        if fail_fast_gateway:
            return resp
        logger.warning("fred %s retry attempt=%s url=%s", resp.status_code, attempt + 1, url)
        time.sleep(min(2**attempt, 5))
    assert resp is not None
    resp.raise_for_status()
    return resp


def _fred_observations_paged(params: dict, rate_gate) -> list:
    observations: list = []
    offset = 0
    limit = int(params.get("limit") or FRED_PAGE_LIMIT)
    while True:
        page_params = {**params, "offset": offset, "limit": limit}
        resp = _fred_api_get(
            FRED_OBS_URL, page_params, rate_gate, fail_fast_gateway=True
        )
        resp.raise_for_status()
        batch = resp.json().get("observations", [])
        observations.extend(batch)
        if len(batch) < limit:
            break
        offset += len(batch)
    return observations


def fetch_fred_series_observation_start(series_id: str, api_key: str, rate_gate) -> date | None:
    resp = _fred_api_get(
        FRED_SERIES_URL,
        {"series_id": series_id, "api_key": api_key, "file_type": "json"},
        rate_gate,
    )
    resp.raise_for_status()
    rows = resp.json().get("seriess") or []
    if not rows:
        return None
    raw = rows[0].get("observation_start")
    if not raw:
        return None
    return date.fromisoformat(str(raw)[:10])


def fetch_fred_vintage_dates(series_id: str, api_key: str, rate_gate) -> list[date]:
    """ALFRED revision dates for a series (oldest → newest)."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    resp = _fred_api_get(FRED_VINTAGE_DATES_URL, params, rate_gate)
    if resp.status_code == 400:
        return []
    resp.raise_for_status()
    out: list[date] = []
    for raw in resp.json().get("vintage_dates") or []:
        try:
            out.append(date.fromisoformat(str(raw)[:10]))
        except ValueError:
            continue
    return sorted(set(out))


def plan_fred_vintage_range(
    series_id: str,
    api_key: str,
    vintage_start: date,
    vintage_end: date,
    rate_gate,
) -> FredVintagePlan | None:
    """Clamp range with vintagedates + observation_start; return revision dates in range."""
    vintage_start = max(vintage_start, FRED_HISTORY_FLOOR)
    if vintage_start > vintage_end:
        return None

    vintage_dates = fetch_fred_vintage_dates(series_id, api_key, rate_gate)
    if vintage_dates and vintage_dates[0] > vintage_start:
        logger.info(
            "fred vintagedates clamp series=%s %s -> %s (n_vintages=%s)",
            series_id,
            vintage_start,
            vintage_dates[0],
            len(vintage_dates),
        )
        vintage_start = vintage_dates[0]

    obs_start = fetch_fred_series_observation_start(series_id, api_key, rate_gate)
    if obs_start is not None and obs_start > vintage_start:
        logger.info(
            "fred obs_start clamp series=%s %s -> %s",
            series_id,
            vintage_start,
            obs_start,
        )
        vintage_start = obs_start

    if vintage_start > vintage_end:
        return None

    in_range = filter_vintage_dates(vintage_dates, vintage_start, vintage_end)
    return FredVintagePlan(
        vintage_start=vintage_start,
        vintage_end=vintage_end,
        vintage_dates=in_range,
    )


def merge_fred_window_details(details: list[dict]) -> list[dict]:
    """Collapse per-window ingest details back to one row per series."""
    by_id: dict[str, dict] = {}
    for d in details:
        sid = d.get("series_id")
        if not sid:
            continue
        cur = by_id.get(sid)
        if cur is None:
            by_id[sid] = dict(d)
            continue
        cur["rows"] = int(cur.get("rows") or 0) + int(d.get("rows") or 0)
        cur["months_written"] = int(cur.get("months_written") or 0) + int(
            d.get("months_written") or 0
        )
        for key in ("first", "last"):
            a, b = cur.get(key), d.get(key)
            if a and b:
                cur[key] = min(a, b) if key == "first" else max(a, b)
            elif b:
                cur[key] = b
        vt = d.get("vintage_through")
        if vt:
            prev = cur.get("vintage_through")
            cur["vintage_through"] = max(prev, vt) if prev else vt
        cur["bisects"] = int(cur.get("bisects") or 0) + int(d.get("bisects") or 0)
        cur["windows_fetched"] = int(cur.get("windows_fetched") or 0) + int(
            d.get("windows_fetched") or 0
        )
    return list(by_id.values())


def _fetch_vintage_range(
    series_id: str,
    api_key: str,
    vintage_start: date,
    vintage_end: date,
    rate_gate,
) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": vintage_start.isoformat(),
        "realtime_end": vintage_end.isoformat(),
        "output_type": ALFRED_OUTPUT_TYPE_VINTAGE,
        "limit": FRED_PAGE_LIMIT,
    }
    observations = _fred_observations_paged(params, rate_gate)
    return observations_to_df(series_id, observations, mode="alfred_vintage")


def _bisect_mid(
    vintage_start: date,
    vintage_end: date,
    vintage_dates: list[date] | None,
) -> date:
    """Split point for left half end; prefer a real revision date near the middle."""
    span_days = (vintage_end - vintage_start).days + 1
    if vintage_dates:
        in_range = [d for d in vintage_dates if vintage_start <= d <= vintage_end]
        if len(in_range) >= 2:
            mid_v = in_range[(len(in_range) - 1) // 2]
            if vintage_start <= mid_v < vintage_end:
                return mid_v
    mid = vintage_start + timedelta(days=span_days // 2 - 1)
    if mid < vintage_start:
        mid = vintage_start
    if mid >= vintage_end:
        mid = vintage_end - timedelta(days=1)
    return mid


def _fetch_adaptive(
    series_id: str,
    api_key: str,
    vintage_start: date,
    vintage_end: date,
    rate_gate,
    *,
    min_window_days: int,
    vintage_dates: list[date] | None = None,
    stats: FetchStats | None = None,
) -> pd.DataFrame:
    """Try window once; on 400/502/503/504 bisect immediately (no gateway retries)."""
    if stats is None:
        stats = FetchStats()
    if vintage_start > vintage_end:
        return observations_to_df(series_id, [], mode="alfred_vintage")

    span_days = (vintage_end - vintage_start).days + 1
    if stats.span_days == 0:
        stats.span_days = span_days
    stats.attempts += 1
    try:
        return _fetch_vintage_range(series_id, api_key, vintage_start, vintage_end, rate_gate)
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else None
        if code not in _SPLITTABLE:
            raise
        if span_days <= min_window_days:
            logger.warning(
                "fred %s skip series=%s vintage=%s..%s span_days=%s",
                code,
                series_id,
                vintage_start,
                vintage_end,
                span_days,
            )
            return observations_to_df(series_id, [], mode="alfred_vintage")
        mid = _bisect_mid(vintage_start, vintage_end, vintage_dates)
        stats.bisects += 1
        logger.warning(
            "fred %s bisect series=%s %s..%s -> %s|%s",
            code,
            series_id,
            vintage_start,
            vintage_end,
            mid,
            mid + timedelta(days=1),
        )
        left = _fetch_adaptive(
            series_id,
            api_key,
            vintage_start,
            mid,
            rate_gate,
            min_window_days=min_window_days,
            vintage_dates=vintage_dates,
            stats=stats,
        )
        right = _fetch_adaptive(
            series_id,
            api_key,
            mid + timedelta(days=1),
            vintage_end,
            rate_gate,
            min_window_days=min_window_days,
            vintage_dates=vintage_dates,
            stats=stats,
        )
        frames = [f for f in (left, right) if not f.empty]
        if not frames:
            return observations_to_df(series_id, [], mode="alfred_vintage")
        return pd.concat(frames, ignore_index=True)


def fetch_fred_revisions(
    series_id: str,
    api_key: str,
    vintage_start: date,
    vintage_end: date,
    rate_gate,
    *,
    min_window_days: int = _MIN_WINDOW_DAYS_DEFAULT,
    clamp: bool = True,
    vintage_dates: list[date] | None = None,
    stats: FetchStats | None = None,
) -> pd.DataFrame:
    """Leakage-proof ALFRED vintage pull for a closed realtime range (one window)."""
    if vintage_start > vintage_end:
        return observations_to_df(series_id, [], mode="alfred_vintage")

    dates = list(vintage_dates or [])
    if clamp:
        plan = plan_fred_vintage_range(
            series_id, api_key, vintage_start, vintage_end, rate_gate
        )
        if plan is None:
            return observations_to_df(series_id, [], mode="alfred_vintage")
        vintage_start, vintage_end = plan.vintage_start, plan.vintage_end
        if not dates:
            dates = list(plan.vintage_dates)

    logger.info(
        "fred vintage pull series=%s %s..%s min_window=%sd vintages=%s",
        series_id,
        vintage_start,
        vintage_end,
        min_window_days,
        len(dates),
    )
    return _fetch_adaptive(
        series_id,
        api_key,
        vintage_start,
        vintage_end,
        rate_gate,
        min_window_days=max(1, min_window_days),
        vintage_dates=dates or None,
        stats=stats,
    )


def fetch_fred_revisions_adaptive(
    series_id: str,
    api_key: str,
    vintage_start: date,
    vintage_end: date,
    rate_gate,
    *,
    vintage_dates: list[date],
    min_window_days: int,
    max_window_days: int,
) -> tuple[pd.DataFrame, FetchStats]:
    """Fetch a series range as vintage-packed windows; retune max days after each window."""
    total = FetchStats()
    if vintage_start > vintage_end:
        return observations_to_df(series_id, [], mode="alfred_vintage"), total

    preferred = gate_preferred_window_days(rate_gate, max_window_days)
    preferred = min(max_window_days, max(min_window_days, preferred))
    remaining_dates = filter_vintage_dates(vintage_dates, vintage_start, vintage_end)
    frames: list[pd.DataFrame] = []

    if remaining_dates:
        while remaining_dates:
            windows = plan_ingest_windows(
                vintage_start=remaining_dates[0],
                vintage_end=remaining_dates[-1],
                vintage_dates=remaining_dates,
                max_days=preferred,
            )
            if not windows:
                break
            ws, we = windows[0]
            window_stats = FetchStats()
            df = fetch_fred_revisions(
                series_id,
                api_key,
                ws,
                we,
                rate_gate,
                min_window_days=min_window_days,
                clamp=False,
                vintage_dates=remaining_dates,
                stats=window_stats,
            )
            preferred = gate_report_window_fetch(
                rate_gate,
                span_days=window_stats.span_days or ((we - ws).days + 1),
                bisects=window_stats.bisects,
            )
            preferred = min(max_window_days, max(min_window_days, preferred))
            total.bisects += window_stats.bisects
            total.attempts += window_stats.attempts
            total.span_days += window_stats.span_days
            total.windows += 1
            if not df.empty:
                frames.append(df)
            remaining_dates = [d for d in remaining_dates if d > we]
    else:
        cur = vintage_start
        while cur <= vintage_end:
            we = min(vintage_end, cur + timedelta(days=preferred - 1))
            window_stats = FetchStats()
            df = fetch_fred_revisions(
                series_id,
                api_key,
                cur,
                we,
                rate_gate,
                min_window_days=min_window_days,
                clamp=False,
                stats=window_stats,
            )
            preferred = gate_report_window_fetch(
                rate_gate,
                span_days=window_stats.span_days or ((we - cur).days + 1),
                bisects=window_stats.bisects,
            )
            preferred = min(max_window_days, max(min_window_days, preferred))
            total.bisects += window_stats.bisects
            total.attempts += window_stats.attempts
            total.span_days += window_stats.span_days
            total.windows += 1
            if not df.empty:
                frames.append(df)
            cur = we + timedelta(days=1)

    if not frames:
        return observations_to_df(series_id, [], mode="alfred_vintage"), total
    return pd.concat(frames, ignore_index=True), total


def fetch_release_series_ids(release_id: int, api_key: str, rate_gate) -> list[str]:
    _require_fred_gate(rate_gate)
    url = "https://api.stlouisfed.org/fred/release/series"
    params = {"release_id": release_id, "api_key": api_key, "file_type": "json", "limit": 1000}
    resp = _fred_api_get(url, params, rate_gate)
    resp.raise_for_status()
    return [s["id"] for s in resp.json().get("seriess", [])]


def resolve_fred_series_ids(cfg: MarketsConfig, rate_gate) -> list[str]:
    _require_fred_gate(rate_gate)
    ids = list(cfg.fred_series)
    for rid in cfg.fred_release_ids:
        ids.extend(fetch_release_series_ids(rid, cfg.fred_api_key, rate_gate))
    unique = sorted({s.upper() for s in ids})
    limited = cfg.resolve_fred_series(unique)
    if len(limited) < len(unique):
        logger.info(
            "fred series limit %s -> %s",
            len(unique),
            len(limited),
        )
    return limited


def write_fred_vintage_frame(lake: LakeStore, df: pd.DataFrame, *, run_id: str) -> dict:
    writer = L1Writer(lake, run_id=run_id)
    return writer.write_parts(df, shard=uuid4().hex[:8])
