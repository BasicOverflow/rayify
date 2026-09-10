"""Ray remote: assemble one series' bars for Serve (stitch or L3 cache + post-process)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import ray

from lexis_markets.lake import LakeStore, PgClient
from lexis_markets.serve.cache import CacheService, read_series_cache
from lexis_markets.config import MarketsConfig
from lexis_markets.domain.derived import apply_derived_features
from lexis_markets.domain.quality import apply_min_volume
from lexis_markets.domain.resample import resample_bars
from lexis_markets.serve.revision import parse_revision_mode, uses_l3_cache
from lexis_markets.serve.stitch import stitch_series
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.runtime import IO_REMOTE_OPTS

logger = get_logger("serve.series_query")


@ray.remote(**IO_REMOTE_OPTS)
def task_query_series_bars(
    cfg_d: dict,
    series_id: str,
    start_iso: str,
    end_iso: str,
    revision_mode: str,
    as_of_iso: str | None,
    granularity: str,
    min_volume: float | None,
    features: list[str] | None,
) -> pd.DataFrame:
    """CPU/IO-heavy per-series path used by Serve ``query_stitched_bars``."""
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    mode = parse_revision_mode(revision_mode)
    as_of = date.fromisoformat(as_of_iso) if as_of_iso else None

    if uses_l3_cache(mode):
        CacheService(cfg)._build_missing_spans(series_id, start, end)
        part = read_series_cache(lake, series_id, start, end)
    else:
        part = stitch_series(
            lake,
            pg,
            series_id,
            start,
            end,
            revision_mode=mode,
            as_of=as_of,
        )

    if part.empty:
        return part

    part = resample_bars(part, granularity)
    part = apply_min_volume(part, min_volume)
    part = apply_derived_features(part, features)
    logger.debug(
        "series_query done series=%s mode=%s rows=%s",
        series_id,
        mode,
        len(part),
    )
    return part
