from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import ray

from lexis_markets.cluster import TaskShape, run_batches
from lexis_markets.config import RULE_VERSION, MarketsConfig
from lexis_markets.stitch_engine import (
    load_obs_for_segments,
    load_stitch_segments,
    month_bounds,
    stitch_merged,
)
from lexis_markets.storage import (
    LakeStore,
    PgClient,
    STITCHED_BAR_COLUMNS,
    months_in_range,
    write_parquet_lake,
)


def stitch_rule_id() -> str:
    return RULE_VERSION


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


def stitched_cache_key(year: int, month: int) -> str:
    return f"layer_3/year={year:04d}/month={month:02d}/stitched.parquet"


def _l1_fp_map(pg: PgClient, months: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    if not months:
        return {}
    placeholders = ", ".join(["(%s, %s)"] * len(months))
    flat = [x for ym in months for x in ym]
    rows = pg.fetchall(
        f"""
        SELECT year, month, COALESCE(compacted_at::text, '') AS l1_fp
        FROM l1_month_manifest
        WHERE (year, month) IN ({placeholders})
        """,
        (*flat,),
    )
    return {(r["year"], r["month"]): r["l1_fp"] for r in rows}


def _l3_manifest_map(pg: PgClient, months: list[tuple[int, int]]) -> dict[tuple[int, int], dict]:
    if not months:
        return {}
    placeholders = ", ".join(["(%s, %s)"] * len(months))
    flat = [x for ym in months for x in ym]
    rows = pg.fetchall(
        f"""
        SELECT year, month, l1_fp, stitch_rule
        FROM l3_month_manifest
        WHERE (year, month) IN ({placeholders})
        """,
        (*flat,),
    )
    return {(r["year"], r["month"]): r for r in rows}


def months_needing_rebuild(
    pg: PgClient,
    months: list[tuple[int, int]],
    *,
    force: bool = False,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    months = sorted(set(months))
    if force:
        return months, []
    rule = stitch_rule_id()
    l1_map = _l1_fp_map(pg, months)
    l3_map = _l3_manifest_map(pg, months)
    need: list[tuple[int, int]] = []
    skip: list[tuple[int, int]] = []
    for ym in months:
        l1_fp = l1_map.get(ym, "")
        if not l1_fp:
            continue
        l3 = l3_map.get(ym)
        if l3 and l3.get("l1_fp") == l1_fp and l3.get("stitch_rule") == rule:
            skip.append(ym)
        else:
            need.append(ym)
    return need, skip


def cached_months_ready(pg: PgClient, months: list[tuple[int, int]]) -> bool:
    if not months:
        return False
    need, _ = months_needing_rebuild(pg, months)
    return len(need) == 0


def cache_status(pg: PgClient) -> dict:
    rule = stitch_rule_id()
    l1_row = pg.fetchone("SELECT COUNT(*) AS n FROM l1_month_manifest")
    l3_row = pg.fetchone("SELECT COUNT(*) AS n FROM l3_month_manifest")
    stale_row = pg.fetchone(
        """
        SELECT COUNT(*) AS n
        FROM l1_month_manifest l1
        LEFT JOIN l3_month_manifest l3 ON l1.year = l3.year AND l1.month = l3.month
        WHERE l3.year IS NULL
           OR l3.l1_fp IS DISTINCT FROM l1.compacted_at::text
           OR l3.stitch_rule IS DISTINCT FROM %s
        """,
        (rule,),
    )
    return {
        "l1_months": int(l1_row["n"]) if l1_row else 0,
        "l3_months": int(l3_row["n"]) if l3_row else 0,
        "stale_months": int(stale_row["n"]) if stale_row else 0,
        "stitch_rule": rule,
        "complete": int(stale_row["n"] or 0) == 0,
    }


def build_stitched_month(
    lake: LakeStore,
    pg: PgClient,
    year: int,
    month: int,
    *,
    seg_df: pd.DataFrame | None = None,
) -> dict:
    start, end = month_bounds(year, month)
    if seg_df is None:
        seg_df = load_stitch_segments(pg, None)
    merged = load_obs_for_segments(lake, pg, start, end, seg_df, revision_mode="as_of")
    bars = stitch_merged(merged)
    out_key = stitched_cache_key(year, month)
    if bars.empty:
        write_parquet_lake(lake, out_key, pd.DataFrame(columns=STITCHED_BAR_COLUMNS), sort_by=["series_id", "ts"])
    else:
        write_parquet_lake(lake, out_key, bars[STITCHED_BAR_COLUMNS], sort_by=["series_id", "ts"])
    l1_fp = _l1_fp_map(pg, [(year, month)]).get((year, month), "")
    rule = stitch_rule_id()
    pg.execute(
        """
        INSERT INTO l3_month_manifest
            (year, month, stitched_key, row_count, series_count, l1_fp, stitch_rule)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (year, month) DO UPDATE SET
            stitched_key = EXCLUDED.stitched_key,
            row_count = EXCLUDED.row_count,
            series_count = EXCLUDED.series_count,
            l1_fp = EXCLUDED.l1_fp,
            stitch_rule = EXCLUDED.stitch_rule
        """,
        (
            year,
            month,
            out_key,
            len(bars),
            int(bars["series_id"].nunique()) if not bars.empty else 0,
            l1_fp,
            rule,
        ),
    )
    return {"year": year, "month": month, "rows": len(bars), "key": out_key}


@ray.remote
def task_build_stitched_month_batch(
    cfg_d: dict,
    months: list[tuple[int, int]],
    seg_df: pd.DataFrame,
) -> list[dict]:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    return [build_stitched_month(lake, pg, y, m, seg_df=seg_df) for y, m in months]


def build_stitched_months(
    cfg: MarketsConfig,
    months: list[tuple[int, int]],
    *,
    force: bool = False,
    pg: PgClient | None = None,
) -> list[dict]:
    months = sorted(set(months))
    if not months:
        return []
    pg = pg or PgClient(cfg.postgres_url)
    need, skip = months_needing_rebuild(pg, months, force=force)
    print(
        f"stitch_cache: months={len(months)} build={len(need)} skip={len(skip)} force={force}",
        flush=True,
    )
    if not need:
        return []
    cfg_d = cfg.to_dict()
    seg_df = load_stitch_segments(pg, None)
    seg_ref = ray.put(seg_df)
    shape = TaskShape(
        batch_size=2,
        max_in_flight=max(1, cfg.minio_max_workers),
    )
    print(
        f"stitch_cache: max_in_flight={shape.max_in_flight} segment_rows={len(seg_df)}",
        flush=True,
    )
    batches = run_batches(
        "stitch_cache",
        need,
        lambda batch: task_build_stitched_month_batch.remote(cfg_d, batch, seg_ref),
        shape,
    )
    return [row for batch in batches for row in batch]


def ensure_stitched_cache(
    cfg: MarketsConfig,
    pg: PgClient,
    lake: LakeStore,
    start: date,
    end: date,
    *,
    force: bool = False,
) -> dict:
    months = months_in_range(start, end)
    need, skip = months_needing_rebuild(pg, months, force=force)
    if need:
        built = build_stitched_months(cfg, need, force=force, pg=pg)
        return {"warmed": len(built), "skipped": len(skip), "needed": len(need)}
    return {"warmed": 0, "skipped": len(skip), "needed": 0}


def months_from_l1_manifest(pg: PgClient) -> list[tuple[int, int]]:
    rows = pg.fetchall("SELECT year, month FROM l1_month_manifest ORDER BY year, month")
    return [(r["year"], r["month"]) for r in rows]


def _read_cached_month(
    lake: LakeStore,
    key: str,
    series_ids: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    filters = None
    if series_ids and len(series_ids) <= 512:
        filters = [("series_id", "in", series_ids)]
    df = lake.get_df_parquet(key, columns=STITCHED_BAR_COLUMNS, filters=filters)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    df = df[(df["ts"] >= start) & (df["ts"] <= end)]
    if series_ids and len(series_ids) > 512:
        sid_set = set(series_ids)
        df = df[df["series_id"].isin(sid_set)]
    return df


def read_stitched_cache(
    lake: LakeStore,
    pg: PgClient,
    series_ids: list[str],
    start: date,
    end: date,
    *,
    max_workers: int = 16,
) -> pd.DataFrame | None:
    months = months_in_range(start, end)
    if not cached_months_ready(pg, months):
        return None
    keys = [stitched_cache_key(y, m) for y, m in months]
    workers = min(max_workers, len(keys) or 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        frames = list(ex.map(lambda k: _read_cached_month(lake, k, series_ids, start, end), keys))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=STITCHED_BAR_COLUMNS)
    return pd.concat(frames, ignore_index=True)
