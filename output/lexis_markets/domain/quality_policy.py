"""Inherent quality policy applied during L3 writes.

Trim phantom listing windows, collapse halt/copy-forward flats, quarantine junk
fills, rewrite L3 when OHLC or window changes. Serve stamps series_meta after
each new L3 span. Set ``QUALITY_INLINE_REPAIR=0`` only for emergency score-only runs.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from lexis_markets.config import MarketsConfig
from lexis_markets.domain.canonical_bar import CANONICAL_BAR_COLUMNS
from lexis_markets.domain.ohlc_fix import enforce_ohlc_bounds
from lexis_markets.domain.quality import (
    collapse_halt_flats,
    dense_observation_window,
    series_quality_window,
)
from lexis_markets.jobs.clock import as_date
from lexis_markets.lake import PgClient, write_parquet_lake, open_lake
from lexis_markets.logging_setup import get_logger
from lexis_markets.registry import (
    merge_spans,
    quarantine_series,
    update_cache_meta,
    update_series_trade_window,
)
from lexis_markets.serve.cache import invalidate_series_l3, l1_fingerprint

logger = get_logger("domain.quality_policy")


def inline_repair_enabled() -> bool:
    """Default on. Set QUALITY_INLINE_REPAIR=0 only for emergency score-only runs."""
    return os.environ.get("QUALITY_INLINE_REPAIR", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def apply_series_quality_policy(
    cfg: MarketsConfig,
    *,
    series_id: str,
    calendar_id: str,
    status: str,
    first_seen: date,
    last_seen: date,
    bars: pd.DataFrame,
    apply: bool = True,
    allow_window_trim: bool = True,
    allow_l3_rewrite: bool = True,
    allow_quarantine: bool = True,
) -> tuple[dict, dict]:
    """Score bars and (when ``apply``) persist window/quarantine/L3 repairs.

    Returns ``(quality_dict, effects)``.

    Do not enable ``allow_window_trim`` / ``allow_l3_rewrite`` on lookback-clipped
    bars — that would shrink registry windows and replace full L3 with a tip.
    """
    effects = {
        "trimmed": False,
        "ohlc_fixed": 0,
        "quarantined": False,
        "purged_l3": False,
        "rebuilt": False,
        "halt_collapsed": 0,
        "new_first": None,
        "new_last": None,
    }
    first = as_date(first_seen)
    last = as_date(last_seen)
    work = bars
    if work is None or work.empty or first is None or last is None:
        q = series_quality_window(
            pd.DataFrame() if work is None else work,
            calendar_id,
            {"status": status},
        )
        return q, effects

    if "ts" in work.columns:
        work = work.copy()
        work["ts"] = pd.to_datetime(work["ts"]).dt.date

    # FRED native calendars have month/week gaps and multi-decade missing stretches.
    # dense_observation_window would lock onto an early contiguous run (e.g. DFF→2011)
    # and purge recent L3 — never trim macros that way.
    if allow_window_trim and str(calendar_id or "").lower() != "fred_native":
        win = dense_observation_window(work["ts"]) if "ts" in work.columns else None
        if win and (win[0] != first or win[1] != last):
            effects["trimmed"] = True
            effects["new_first"] = win[0].isoformat()
            effects["new_last"] = win[1].isoformat()
            first, last = win
            work = work[(work["ts"] >= first) & (work["ts"] <= last)].reset_index(drop=True)
            if apply and inline_repair_enabled():
                pg = PgClient(cfg.postgres_url, pool_max=1)
                update_series_trade_window(pg, series_id, first, last)

    n_before = len(work)
    work = collapse_halt_flats(work)
    effects["halt_collapsed"] = max(0, n_before - len(work))

    work, n_ohlc = enforce_ohlc_bounds(work)
    effects["ohlc_fixed"] = int(n_ohlc)

    q = series_quality_window(work, calendar_id, {"status": status})

    if not apply or not inline_repair_enabled():
        return q, effects

    if allow_quarantine and q.get("quarantine") and status != "BAD_DATA":
        pg = PgClient(cfg.postgres_url, pool_max=1)
        quarantine_series(
            pg,
            series_id,
            note=f"ramp/flat junk flags={q.get('flags')}",
        )
        if allow_l3_rewrite:
            invalidate_series_l3(cfg, series_id)
            effects["purged_l3"] = True
        effects["quarantined"] = True
        logger.info("quality_policy quarantine series=%s", series_id)
        return q, effects

    if allow_l3_rewrite and (
        effects["trimmed"] or n_ohlc > 0 or effects["halt_collapsed"] > 0
    ):
        invalidate_series_l3(cfg, series_id)
        effects["purged_l3"] = True
        lake = open_lake(cfg)
        pg = PgClient(cfg.postgres_url, pool_max=1)
        key = cfg.cache_object_key(series_id)
        write_df = work.copy()
        if "series_id" not in write_df.columns:
            write_df["series_id"] = series_id
        for col in CANONICAL_BAR_COLUMNS:
            if col not in write_df.columns:
                write_df[col] = None
        write_parquet_lake(
            lake,
            key,
            write_df[CANONICAL_BAR_COLUMNS],
            sort_by=["series_id", "ts"],
        )
        merge_spans(pg, series_id, first, last)
        fp = l1_fingerprint(pg, first, last)
        update_cache_meta(pg, series_id, len(write_df), fp)
        effects["rebuilt"] = True
        logger.info(
            "quality_policy rewrite series=%s trimmed=%s ohlc_fixed=%s rows=%s",
            series_id,
            effects["trimmed"],
            n_ohlc,
            len(write_df),
        )
    return q, effects
