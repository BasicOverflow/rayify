"""Per-series L3 canonical cache: span tracking, miss detection, Ray build dispatch."""
from __future__ import annotations

from datetime import date

import pandas as pd
import ray

from lexis_markets.config import MarketsConfig, cache_object_key
from lexis_markets.domain.canonical_bar import CANONICAL_BAR_COLUMNS
from lexis_markets.jobs.scheduler import TaskShape, run_batches
from lexis_markets.lake import LakeStore, PgClient, write_parquet_lake
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.runtime import max_in_flight
from lexis_markets.registry import merge_spans, uncached_ranges, update_cache_meta
from lexis_markets.serve.dedupe import InFlightDedupe
from lexis_markets.serve.stitch import merge_canonical_cache, stitch_series

logger = get_logger("serve.cache")


def l1_fingerprint(pg: PgClient, start: date, end: date) -> str:
    row = pg.fetchone(
        """
        SELECT COALESCE(MAX(compacted_at)::text, '') AS fp
        FROM l1_month_manifest
        WHERE (year > %s OR (year = %s AND month >= %s))
          AND (year < %s OR (year = %s AND month <= %s))
        """,
        (start.year, start.year, start.month, end.year, end.year, end.month),
    )
    return row["fp"] if row else ""


def read_series_cache(
    lake: LakeStore,
    series_id: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    key = cache_object_key(series_id)
    if not lake.exists(key):
        return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    df = lake.get_df_parquet(key, columns=CANONICAL_BAR_COLUMNS)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    return df[(df["ts"] >= start) & (df["ts"] <= end)].reset_index(drop=True)


def _read_cache(lake: LakeStore, cfg: MarketsConfig, series_id: str) -> pd.DataFrame | None:
    key = cfg.cache_object_key(series_id)
    if not lake.exists(key):
        return None
    df = lake.get_df_parquet(key, columns=CANONICAL_BAR_COLUMNS)
    return df if not df.empty else None


@ray.remote
def task_build_cache_span(
    cfg_d: dict,
    series_id: str,
    start_iso: str,
    end_iso: str,
) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)

    logger.info("cache_build series=%s span=%s..%s", series_id, start_iso, end_iso)
    new_bars = stitch_series(lake, pg, series_id, start, end)
    existing = _read_cache(lake, cfg, series_id)
    merged = merge_canonical_cache(existing, new_bars)

    key = cfg.cache_object_key(series_id)
    if merged.empty:
        write_parquet_lake(
            lake,
            key,
            pd.DataFrame(columns=CANONICAL_BAR_COLUMNS),
            sort_by=["series_id", "ts"],
        )
    else:
        write_parquet_lake(lake, key, merged[CANONICAL_BAR_COLUMNS], sort_by=["series_id", "ts"])

    span_start = start
    span_end = end
    if not merged.empty:
        ts = pd.to_datetime(merged["ts"]).dt.date
        span_start = min(ts.min(), start)
        span_end = max(ts.max(), end)

    out = {
        "series_id": series_id,
        "span_start": span_start.isoformat(),
        "span_end": span_end.isoformat(),
        "rows": len(merged),
    }
    logger.info("cache_build done series=%s rows=%s", series_id, out["rows"])
    return out


@ray.remote
def task_ensure_series_cache(cfg_d: dict, series_id: str, start_iso: str, end_iso: str) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    svc = CacheService(cfg)
    return svc._build_missing_spans(series_id, date.fromisoformat(start_iso), date.fromisoformat(end_iso))


class CacheService:
    def __init__(self, cfg: MarketsConfig, *, dedupe: InFlightDedupe | None = None):
        self.cfg = cfg
        self.pg = PgClient(cfg.postgres_url)
        self.lake = LakeStore(cfg)
        self._dedupe = dedupe or InFlightDedupe()

    def _build_missing_spans(self, series_id: str, start: date, end: date) -> dict:
        if not self.pg.fetchone(
            "SELECT 1 FROM series_meta WHERE series_id = %s", (series_id,)
        ):
            raise ValueError(f"unknown series_id: {series_id}")

        missing = uncached_ranges(self.pg, series_id, start, end)
        if not missing:
            logger.debug("cache hit series=%s %s..%s", series_id, start, end)
            return {"series_id": series_id, "built_spans": 0, "rows": 0}

        cfg_d = self.cfg.to_dict()
        jobs = [
            (series_id, dr.start.isoformat(), dr.end.isoformat())
            for dr in missing
        ]
        shape = TaskShape(batch_size=1, max_in_flight=max_in_flight())
        results = run_batches(
            f"cache_{series_id}",
            jobs,
            lambda batch: task_build_cache_span.remote(cfg_d, batch[0][0], batch[0][1], batch[0][2]),
            shape,
        )

        total_rows = 0
        for row in results:
            span_start = date.fromisoformat(row["span_start"])
            span_end = date.fromisoformat(row["span_end"])
            merge_spans(self.pg, series_id, span_start, span_end)
            total_rows = max(total_rows, int(row["rows"]))

        fp = l1_fingerprint(self.pg, start, end)
        update_cache_meta(self.pg, series_id, total_rows, fp)
        logger.info(
            "cache built series=%s spans=%s rows=%s",
            series_id,
            len(missing),
            total_rows,
        )
        return {"series_id": series_id, "built_spans": len(missing), "rows": total_rows}

    def ensure_series_cached(self, series_id: str, start: date, end: date) -> dict:
        key = (series_id, start.isoformat(), end.isoformat())
        return self._dedupe.wait(
            key,
            lambda: task_ensure_series_cache.remote(
                self.cfg.to_dict(),
                series_id,
                start.isoformat(),
                end.isoformat(),
            ),
        )
