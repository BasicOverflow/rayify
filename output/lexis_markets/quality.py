from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import ray

from lexis_markets.config import MarketsConfig
from lexis_markets.storage import PgClient

if TYPE_CHECKING:
    from lexis_markets.api import MarketsService


def session_span_days(first: date, last: date, calendar_id: str) -> int:
    if calendar_id == "fred_native":
        return (last - first).days + 1
    return int(np.busday_count(first, last)) + 1


def gaps_from_bars(bars: pd.DataFrame, calendar_id: str) -> tuple[int, int, float]:
    if bars.empty:
        return 0, 0, 1.0
    ts = pd.to_datetime(bars["ts"]).dt.date
    first, last = ts.min(), ts.max()
    span = session_span_days(first, last, calendar_id)
    days = int(ts.nunique())
    gap = max(0, span - days)
    disagreement = int((bars.get("source_count", 1) > 1).sum()) if "source_count" in bars.columns else 0
    score = days / span if span else 1.0
    return gap, disagreement, score


def source_segments(bars: pd.DataFrame) -> list[dict]:
    if bars.empty or "source" not in bars.columns:
        return []
    df = bars.sort_values("ts")
    sources = df["source"].astype(str).to_numpy()
    dates = pd.to_datetime(df["ts"]).dt.date.to_numpy()
    runs: list[dict] = []
    start = 0
    for i in range(1, len(df)):
        if sources[i] != sources[i - 1]:
            runs.append(
                {
                    "source": sources[start],
                    "start": str(dates[start]),
                    "end": str(dates[i - 1]),
                    "days": int(i - start),
                }
            )
            start = i
    runs.append(
        {
            "source": sources[start],
            "start": str(dates[start]),
            "end": str(dates[-1]),
            "days": int(len(df) - start),
        }
    )
    return runs


def series_quality_window(bars: pd.DataFrame, calendar_id: str, meta: dict | None = None) -> dict:
    gap, disagreement, coverage = gaps_from_bars(bars, calendar_id)
    out: dict = {
        "calendar_id": calendar_id,
        "window_gap_days": gap,
        "window_disagreement_days": disagreement,
        "window_coverage": round(coverage, 6),
        "bar_count": len(bars),
        "source_segments": source_segments(bars),
    }
    if not bars.empty and "data_quality" in bars.columns:
        out["data_quality_counts"] = {
            str(k): int(v) for k, v in bars["data_quality"].value_counts().items()
        }
    if meta:
        out["registry"] = {
            "gap_count": meta.get("gap_count"),
            "disagreement_count": meta.get("disagreement_count"),
            "quality_score": meta.get("quality_score"),
            "first_seen": str(meta["first_seen"]) if meta.get("first_seen") else None,
            "last_seen": str(meta["last_seen"]) if meta.get("last_seen") else None,
            "status": meta.get("status"),
        }
    return out


def apply_min_volume(bars: pd.DataFrame, min_volume: float | None) -> pd.DataFrame:
    if min_volume is None or bars.empty or "volume" not in bars.columns:
        return bars
    vol = pd.to_numeric(bars["volume"], errors="coerce")
    return bars[vol.notna() & (vol >= min_volume)].copy()


def _recompute_rows(rows: list[dict], svc: MarketsService) -> list[tuple]:
    updates = []
    for r in rows:
        spec = {
            "series_ids": [r["series_id"]],
            "start": str(r["first_seen"]),
            "end": str(r["last_seen"]),
            "granularity": "daily",
            "revision_mode": "as_of",
        }
        bars = svc.query_stitched_bars(spec)
        gap, disagreement, score = gaps_from_bars(bars, r["calendar_id"] or "nyse")
        updates.append((gap, disagreement, score, r["series_id"]))
    return updates


@ray.remote
def task_recompute_gaps_batch(cfg_d: dict, rows: list[dict]) -> list[tuple]:
    from lexis_markets.api import MarketsService

    svc = MarketsService(MarketsConfig.from_dict(cfg_d))
    return _recompute_rows(rows, svc)


def recompute_gaps(
    pg: PgClient,
    svc: MarketsService,
    *,
    asset_classes: tuple[str, ...] = ("equity", "etf", "macro"),
    series_ids: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 8,
) -> dict:
    if series_ids:
        rows = pg.fetchall(
            """
            SELECT series_id, calendar_id, first_seen, last_seen
            FROM series_meta
            WHERE series_id = ANY(%s) AND first_seen IS NOT NULL AND last_seen IS NOT NULL
            """,
            (series_ids,),
        )
    else:
        rows = pg.fetchall(
            """
            SELECT series_id, calendar_id, first_seen, last_seen
            FROM series_meta
            WHERE asset_class = ANY(%s) AND first_seen IS NOT NULL AND last_seen IS NOT NULL
            ORDER BY series_id
            """,
            (list(asset_classes),),
        )
    if limit:
        rows = rows[:limit]
    if not rows:
        return {"series": 0, "gap_nonzero": 0, "gap_total": 0}

    cfg_d = svc.cfg.to_dict()
    updates: list[tuple] = []
    if len(rows) <= batch_size:
        updates = _recompute_rows(rows, svc)
    else:
        batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
        refs = [task_recompute_gaps_batch.remote(cfg_d, batch) for batch in batches]
        for part in ray.get(refs):
            updates.extend(part)

    pg.executemany(
        """
        UPDATE series_meta
        SET gap_count = %s, disagreement_count = %s, quality_score = %s
        WHERE series_id = %s
        """,
        updates,
    )
    gaps = [u[0] for u in updates]
    return {
        "series": len(updates),
        "gap_nonzero": sum(1 for g in gaps if g > 0),
        "gap_total": sum(gaps),
    }
