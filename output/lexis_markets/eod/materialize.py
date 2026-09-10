"""Seed-phase cluster materialize: stitch+quality on worker lake, L3 local.

After Kaggle/YF/FRED L1 lands on seed scratch actors and compact finishes,
split L1 files evenly across workers. Each worker reads its files from the
cluster lake (no MinIO), attaches ``series_id``, and shuffles rows so each
worker owns an even series shard. Stitch and quality run in-process.
Compacted L1 + L3 flush to MinIO once at the end of seed.

Daily EOD cron stays on Serve tip warm.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import ray

from lexis_markets.config import MarketsConfig
from lexis_markets.domain.canonical_bar import CANONICAL_BAR_COLUMNS
from lexis_markets.domain.quality_policy import apply_series_quality_policy
from lexis_markets.jobs.clock import as_date
from lexis_markets.lake import LakeStore, PgClient, write_parquet_lake, open_lake, lake_from_cfg_d, cfg_d_with_scratch
from lexis_markets.lake.cluster import _parquet_bytes, _read_parquet_bytes
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.runtime import CPU_REMOTE_OPTS
from lexis_markets.scratch import mark_live
from lexis_markets.registry import merge_spans, update_cache_meta
from lexis_markets.serve.cache import l1_fingerprint
from lexis_markets.serve.quality_persist import apply_quality_updates, quality_update_tuple
from lexis_markets.serve.stitch import (
    load_stitch_segments,
    stitch_series_from_obs,
)

logger = get_logger("eod.materialize")

L1_OBS_COLS = (
    "source",
    "source_symbol",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "realtime_start",
    "realtime_end",
)


def shard_index(series_id: str, n: int) -> int:
    if n <= 1:
        return 0
    return int(hashlib.sha256(str(series_id).encode()).hexdigest(), 16) % n


def even_shards(items: list, n: int) -> list[list]:
    """Split ``items`` into ``n`` shards with sizes differing by at most one."""
    if n <= 1:
        return [list(items)]
    if not items:
        return [[] for _ in range(n)]
    n = min(n, len(items))
    out: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        out[i % n].append(item)
    return out


def shard_count(*, n_items: int, requested: int | None = None) -> int:
    if n_items <= 0:
        return 1
    if requested is not None and requested > 0:
        return max(1, min(int(requested), n_items))
    cpus = 1
    if ray.is_initialized():
        cpus = int(ray.available_resources().get("CPU", 1) or 1)
    return max(1, min(cpus, n_items))


def list_l1_parquet_keys(lake: LakeStore) -> list[str]:
    raw = lake.list_keys("layer_1/")
    prefix = lake.prefix or ""
    out: list[str] = []
    for k in raw:
        if not str(k).endswith(".parquet"):
            continue
        rel = k[len(prefix) :] if prefix and str(k).startswith(prefix) else k
        out.append(rel)
    return sorted(out)


def attach_series_ids(obs: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Join L1 rows to ``stitch_segments`` (source, symbol, valid window)."""
    if obs is None or obs.empty or segments is None or segments.empty:
        return pd.DataFrame()
    need = [c for c in L1_OBS_COLS if c in obs.columns]
    if "source" not in need or "source_symbol" not in need or "ts" not in need:
        return pd.DataFrame()
    part = obs[need].copy()
    part["source_symbol"] = part["source_symbol"].astype(str).str.upper()
    part["ts"] = pd.to_datetime(part["ts"]).dt.date
    seg = segments.copy()
    seg["source_symbol"] = seg["source_symbol"].astype(str).str.upper()
    merged = part.merge(seg, on=["source", "source_symbol"], how="inner")
    if merged.empty:
        return merged
    for col in ("valid_from", "valid_to"):
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], errors="coerce").dt.date
    merged = merged[merged["valid_from"].isna() | (merged["ts"] >= merged["valid_from"])]
    merged = merged[merged["valid_to"].isna() | (merged["ts"] <= merged["valid_to"])]
    return merged.reset_index(drop=True)


def load_series_jobs(pg: PgClient) -> list[dict]:
    rows = pg.fetchall(
        """
        SELECT series_id, first_seen, last_seen, calendar_id, status, asset_class
        FROM series_meta
        WHERE first_seen IS NOT NULL
          AND last_seen IS NOT NULL
        ORDER BY series_id
        """
    )
    jobs: list[dict] = []
    for r in rows:
        first = as_date(r.get("first_seen"))
        last = as_date(r.get("last_seen"))
        if first is None or last is None or first > last:
            continue
        jobs.append(
            {
                "series_id": str(r["series_id"]),
                "first_seen": first.isoformat(),
                "last_seen": last.isoformat(),
                "calendar_id": r.get("calendar_id") or "nyse",
                "status": r.get("status") or "ACTIVE",
                "asset_class": r.get("asset_class") or "equity",
            }
        )
    return jobs


def stitch_and_score(
    series_id: str,
    first: date,
    last: date,
    seg_df: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    calendar_id: str,
    status: str,
    cfg: MarketsConfig | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Stitch one series from in-memory L1 and score (no I/O when ``cfg`` is None)."""
    rev = "as_of" if str(series_id).startswith("macro:") else "latest"
    as_of = last if rev == "as_of" else None
    bars = stitch_series_from_obs(
        series_id,
        first,
        last,
        seg_df,
        obs,
        revision_mode=rev,
        as_of=as_of,
    )
    if cfg is None:
        from lexis_markets.domain.quality import series_quality_window

        q = series_quality_window(bars, calendar_id, {"status": status})
        return bars, q, {}
    q, effects = apply_series_quality_policy(
        cfg,
        series_id=series_id,
        calendar_id=calendar_id,
        status=status,
        first_seen=first,
        last_seen=last,
        bars=bars,
        apply=True,
        allow_window_trim=True,
        allow_l3_rewrite=True,
        allow_quarantine=True,
    )
    return bars, q, effects


def persist_l3_bars(
    cfg: MarketsConfig,
    series_id: str,
    first: date,
    last: date,
    bars: pd.DataFrame,
    effects: dict,
) -> int:
    """Write L3 when quality policy did not already rewrite or purge."""
    if effects.get("purged_l3") and not effects.get("rebuilt"):
        return 0
    if effects.get("rebuilt"):
        return 0 if bars is None else len(bars)
    lake = open_lake(cfg)
    pg = PgClient(cfg.postgres_url, pool_max=1)
    key = cfg.cache_object_key(series_id)
    write_df = bars.copy() if bars is not None else pd.DataFrame()
    if write_df.empty:
        write_df = pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    else:
        if "series_id" not in write_df.columns:
            write_df["series_id"] = series_id
        for col in CANONICAL_BAR_COLUMNS:
            if col not in write_df.columns:
                write_df[col] = None
        write_df = write_df[CANONICAL_BAR_COLUMNS]
    write_parquet_lake(lake, key, write_df, sort_by=["series_id", "ts"])
    if not write_df.empty:
        ts = pd.to_datetime(write_df["ts"]).dt.date
        span_start = min(ts.min(), first)
        span_end = max(ts.max(), last)
    else:
        span_start, span_end = first, last
    merge_spans(pg, series_id, span_start, span_end)
    fp = l1_fingerprint(pg, first, last)
    update_cache_meta(pg, series_id, len(write_df), fp)
    return len(write_df)


def _obs_for_stitch(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in L1_OBS_COLS if c in frame.columns]
    return frame[cols] if cols else frame


def encode_shard_frames(buckets: list[list[pd.DataFrame]]) -> tuple[bytes, ...]:
    """Parquet bytes per dest shard. Empty bucket -> b'' (no numpy pickle)."""
    out: list[bytes] = []
    for parts in buckets:
        if parts:
            out.append(_parquet_bytes(pd.concat(parts, ignore_index=True)))
        else:
            out.append(b"")
    return tuple(out)


def decode_shard_parts(part_blobs) -> pd.DataFrame:
    frames = [_read_parquet_bytes(b) for b in part_blobs if b]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@ray.remote(**CPU_REMOTE_OPTS)
def task_map_l1_shard(
    cfg_d: dict,
    keys: list[str],
    n_shards: int,
    segments_records: list[dict],
) -> list:
    """Download this worker's L1 files, attach series_id, split frames per dest shard."""
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = lake_from_cfg_d(cfg_d)
    segs = pd.DataFrame(segments_records)
    buckets: list[list[pd.DataFrame]] = [[] for _ in range(n_shards)]
    work = Path(tempfile.mkdtemp(prefix="l1_map_"))
    mark_live(work)
    try:
        for key in keys:
            dest = work / Path(str(key).replace("/", "_"))
            lake.download_file(key, dest)
            df = pd.read_parquet(dest)
            dest.unlink(missing_ok=True)
            attached = attach_series_ids(df, segs)
            if attached.empty:
                continue
            shards = attached["series_id"].map(lambda s: shard_index(str(s), n_shards))
            attached = attached.assign(_shard=shards)
            for i, part in attached.groupby("_shard", sort=False):
                buckets[int(i)].append(part.drop(columns=["_shard"]))
        return encode_shard_frames(buckets)
    finally:
        shutil.rmtree(work, ignore_errors=True)


@ray.remote(**CPU_REMOTE_OPTS)
def task_reduce_shard(
    cfg_d: dict,
    series_jobs: list[dict],
    segments_records: list[dict],
    *part_blobs: bytes,
) -> dict:
    """Stitch+score this worker's series from shuffled L1, write L3 + quality."""
    cfg = MarketsConfig.from_dict(cfg_d)
    lake_from_cfg_d(cfg_d)
    obs_raw = _obs_for_stitch(decode_shard_parts(part_blobs))
    seg_df = pd.DataFrame(segments_records)
    updates: list[tuple] = []
    ok = 0
    empty = 0
    rows = 0
    for job in series_jobs:
        sid = job["series_id"]
        first = date.fromisoformat(job["first_seen"])
        last = date.fromisoformat(job["last_seen"])
        bars, q, effects = stitch_and_score(
            sid,
            first,
            last,
            seg_df,
            obs_raw,
            calendar_id=job.get("calendar_id") or "nyse",
            status=job.get("status") or "ACTIVE",
            cfg=cfg,
        )
        n = persist_l3_bars(cfg, sid, first, last, bars, effects)
        updates.append(quality_update_tuple(q, sid))
        rows += n
        if n > 0:
            ok += 1
        else:
            empty += 1
    if updates:
        apply_quality_updates(PgClient(cfg.postgres_url, pool_max=1), updates)
    logger.info(
        "materialize reduce series=%s ok=%s empty=%s rows=%s",
        len(series_jobs),
        ok,
        empty,
        rows,
    )
    return {
        "series": len(series_jobs),
        "ok": ok,
        "empty": empty,
        "rows": rows,
    }


def run_materialize(cfg: MarketsConfig, *, workers: int | None = None) -> dict:
    """Pull L1 onto the cluster, process even series shards, write L3 + flags."""
    lake = open_lake(cfg)
    pg = PgClient(cfg.postgres_url)
    keys = list_l1_parquet_keys(lake)
    jobs = load_series_jobs(pg)
    segs = load_stitch_segments(pg, None)
    if not keys or not jobs:
        logger.info("materialize skip keys=%s series=%s", len(keys), len(jobs))
        return {"shards": 0, "keys": len(keys), "series": len(jobs), "ok": 0, "rows": 0}

    n = shard_count(n_items=max(len(keys), len(jobs)), requested=workers)
    n = min(n, max(1, len(keys)), max(1, len(jobs)))
    key_shards = even_shards(keys, n)
    series_shards = [[] for _ in range(n)]
    for job in jobs:
        series_shards[shard_index(job["series_id"], n)].append(job)
    seg_records = segs.to_dict("records") if not segs.empty else []
    cfg_d = cfg_d_with_scratch(cfg)

    logger.info(
        "materialize start shards=%s keys=%s series=%s",
        n,
        len(keys),
        len(jobs),
    )
    map_out: list = []
    for shard in key_shards:
        if not shard:
            continue
        refs = task_map_l1_shard.options(num_returns=n).remote(
            cfg_d, shard, n, seg_records
        )
        if n == 1:
            refs = [refs]
        map_out.append(refs)
    reduce_refs = []
    for dest in range(n):
        parts = [row[dest] for row in map_out]
        reduce_refs.append(
            task_reduce_shard.remote(cfg_d, series_shards[dest], seg_records, *parts)
        )
    reduced = ray.get(reduce_refs)
    out = {
        "shards": n,
        "keys": len(keys),
        "series": len(jobs),
        "ok": sum(int(r.get("ok") or 0) for r in reduced),
        "empty": sum(int(r.get("empty") or 0) for r in reduced),
        "rows": sum(int(r.get("rows") or 0) for r in reduced),
    }
    logger.info("materialize done %s", out)
    return out
