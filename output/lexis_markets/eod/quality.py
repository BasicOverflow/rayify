"""Ray remote tasks: per-series gap/disagreement scan against L3 cache."""
from __future__ import annotations

from datetime import date

import pandas as pd
import ray

from lexis_markets.lake import LakeStore, PgClient
from lexis_markets.config import MarketsConfig
from lexis_markets.domain.quality import series_quality_window
from lexis_markets.serve.cache import read_series_cache
from lexis_markets.serve.stitch import stitch_series
from lexis_markets.logging_setup import get_logger

logger = get_logger("eod.quality")


def _bars_for_series(
    lake: LakeStore,
    pg: PgClient,
    series_id: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    cached = read_series_cache(lake, series_id, start, end)
    if not cached.empty:
        return cached
    return stitch_series(lake, pg, series_id, start, end)


def _scan_rows(rows: list[dict], lake: LakeStore, pg: PgClient) -> list[tuple]:
    updates: list[tuple] = []
    for row in rows:
        series_id = row["series_id"]
        calendar_id = row.get("calendar_id") or "nyse"
        start = row["first_seen"]
        end = row["last_seen"]
        if not start or not end:
            continue
        bars = _bars_for_series(lake, pg, series_id, start, end)
        q = series_quality_window(bars, calendar_id)
        updates.append(
            (
                q["gap_count"],
                q["disagreement_count"],
                q["suspicious_count"],
                q["quality_score"],
                series_id,
            )
        )
    return updates


@ray.remote
def task_quality_scan_batch(cfg_d: dict, rows: list[dict]) -> list[tuple]:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    updates = _scan_rows(rows, lake, pg)
    logger.info("quality_scan batch=%s updates=%s", len(rows), len(updates))
    return updates


def apply_quality_updates(pg: PgClient, updates: list[tuple]) -> None:
    if not updates:
        return
    pg.executemany(
        """
        UPDATE series_meta
        SET gap_count = %s, disagreement_count = %s, suspicious_count = %s, quality_score = %s
        WHERE series_id = %s
        """,
        updates,
    )
