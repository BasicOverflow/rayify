"""Stamp series_meta quality when Serve writes new L3 spans.

Full-history writes run the full policy (trim / quarantine / halt rewrite).
Tip writes merge flags so a short window cannot zero out historical counts.
Cache hits do not persist.
"""
from __future__ import annotations

from datetime import date, timedelta

from lexis_markets.config import MarketsConfig
from lexis_markets.domain.quality_policy import apply_series_quality_policy
from lexis_markets.jobs.clock import as_date
from lexis_markets.lake import LakeStore, PgClient
from lexis_markets.logging_setup import get_logger
from lexis_markets.serve.cache import read_series_cache

logger = get_logger("serve.quality_persist")

TIP_JOIN_CALENDAR_DAYS = 40
_FLAG_KEYS = (
    "linear_ramp",
    "sparse_bridge",
    "flat_close",
    "ohlc_violation",
    "non_positive_close",
    "duplicate_ts",
    "extreme_return",
)


def quality_update_tuple(q: dict, series_id: str) -> tuple:
    fc = q.get("flag_counts") or {}
    return (
        int(q.get("gap_count") or 0),
        int(q.get("disagreement_count") or 0),
        int(q.get("suspicious_count") or 0),
        float(q.get("quality_score") or 0),
        int(fc.get("linear_ramp", 0) or 0),
        int(fc.get("sparse_bridge", 0) or 0),
        int(fc.get("flat_close", 0) or 0),
        int(fc.get("ohlc_violation", 0) or 0),
        int(fc.get("non_positive_close", 0) or 0),
        int(fc.get("duplicate_ts", 0) or 0),
        int(fc.get("extreme_return", 0) or 0),
        series_id,
    )


def apply_quality_updates(pg: PgClient, updates: list[tuple]) -> None:
    if not updates:
        return
    pg.executemany(
        """
        UPDATE series_meta
        SET gap_count = %s,
            disagreement_count = %s,
            suspicious_count = %s,
            quality_score = %s,
            flag_linear_ramp = %s,
            flag_sparse_bridge = %s,
            flag_flat_close = %s,
            flag_ohlc_violation = %s,
            flag_non_positive_close = %s,
            flag_duplicate_ts = %s,
            flag_extreme_return = %s
        WHERE series_id = %s
        """,
        updates,
    )


def _meta_row(pg: PgClient, series_id: str) -> dict | None:
    return pg.fetchone(
        """
        SELECT series_id, calendar_id, status, first_seen, last_seen,
               gap_count, disagreement_count, suspicious_count, quality_score,
               flag_linear_ramp, flag_sparse_bridge, flag_flat_close,
               flag_ohlc_violation, flag_non_positive_close,
               flag_duplicate_ts, flag_extreme_return
        FROM series_meta
        WHERE series_id = %s
        """,
        (series_id,),
    )


def _is_full_window(request_start: date, request_end: date, first: date, last: date) -> bool:
    return request_start <= first and request_end >= last


def _merge_tip_tuple(q: dict, series_id: str, existing: dict) -> tuple:
    """Keep historical counts; raise them if the tip found more issues."""
    fc = q.get("flag_counts") or {}
    old_score = existing.get("quality_score")
    new_score = q.get("quality_score")
    if old_score is None:
        score = float(new_score or 0)
    elif new_score is None:
        score = float(old_score or 0)
    else:
        score = min(float(old_score), float(new_score))
    return (
        max(int(existing.get("gap_count") or 0), int(q.get("gap_count") or 0)),
        max(int(existing.get("disagreement_count") or 0), int(q.get("disagreement_count") or 0)),
        max(int(existing.get("suspicious_count") or 0), int(q.get("suspicious_count") or 0)),
        score,
        max(int(existing.get("flag_linear_ramp") or 0), int(fc.get("linear_ramp") or 0)),
        max(int(existing.get("flag_sparse_bridge") or 0), int(fc.get("sparse_bridge") or 0)),
        max(int(existing.get("flag_flat_close") or 0), int(fc.get("flat_close") or 0)),
        max(int(existing.get("flag_ohlc_violation") or 0), int(fc.get("ohlc_violation") or 0)),
        max(int(existing.get("flag_non_positive_close") or 0), int(fc.get("non_positive_close") or 0)),
        max(int(existing.get("flag_duplicate_ts") or 0), int(fc.get("duplicate_ts") or 0)),
        max(int(existing.get("flag_extreme_return") or 0), int(fc.get("extreme_return") or 0)),
        series_id,
    )


def persist_l3_quality(
    cfg: MarketsConfig,
    series_id: str,
    request_start: date,
    request_end: date,
) -> dict:
    """Score L3 for ``series_id`` after a span write and stamp series_meta."""
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url, pool_max=1)
    meta = _meta_row(pg, series_id)
    if not meta:
        return {"series_id": series_id, "persisted": False, "reason": "missing_meta"}
    first = as_date(meta.get("first_seen"))
    last = as_date(meta.get("last_seen"))
    if first is None or last is None:
        return {"series_id": series_id, "persisted": False, "reason": "no_window"}

    full = _is_full_window(request_start, request_end, first, last)
    if full:
        bar_start, bar_end = first, last
    else:
        bar_start = max(first, request_start - timedelta(days=TIP_JOIN_CALENDAR_DAYS))
        bar_end = min(last, request_end)

    bars = read_series_cache(lake, series_id, bar_start, bar_end)
    calendar_id = meta.get("calendar_id") or "nyse"
    status = meta.get("status") or "ACTIVE"
    q, effects = apply_series_quality_policy(
        cfg,
        series_id=series_id,
        calendar_id=calendar_id,
        status=status,
        first_seen=first,
        last_seen=last,
        bars=bars,
        apply=True,
        allow_window_trim=full,
        allow_l3_rewrite=full,
        allow_quarantine=True,
    )
    if full:
        update = quality_update_tuple(q, series_id)
    else:
        update = _merge_tip_tuple(q, series_id, meta)
    apply_quality_updates(pg, [update])
    logger.info(
        "quality_persist series=%s full=%s suspicious=%s trimmed=%s quarantined=%s",
        series_id,
        full,
        update[2],
        effects.get("trimmed"),
        effects.get("quarantined"),
    )
    return {
        "series_id": series_id,
        "persisted": True,
        "full": full,
        "suspicious_count": update[2],
        "effects": effects,
    }
