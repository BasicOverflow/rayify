"""Ray remote tasks: L1 bulk ingest (jakewright, jacksoncrow)."""
from __future__ import annotations

import time
import zipfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import requests

from lexis_markets.lake import LakeStore, L1Writer, get_json, put_json, delete_prefix, lake_from_cfg_d, cfg_d_with_scratch
from lexis_markets.config import MarketsConfig
from lexis_markets.domain.sources.mappers import map_ohlcv, merge_details
from lexis_markets.logging_setup import get_logger
from lexis_markets.jobs.scheduler import plan_task_resources, run_batches
from lexis_markets.ray.runtime import DEFAULT_REMOTE_OPTS
from lexis_markets.scratch import scratch_dir

logger = get_logger("kaggle.ingest")

JAKEWRIGHT = "jakewright/9000-tickers-of-stock-market-data-full-history"
JACKSONCROW = "jacksoncrow/stock-market-dataset"
JC_CACHE_KEY = "ops/cache/jacksoncrow.zip"
JW_CACHE_KEY = "ops/cache/jakewright.zip"
JW_MARKER = "ops/markers/jakewright_l1.json"
JC_MARKER = "ops/markers/jacksoncrow_l1.json"
JW_STAGING = "ops/staging/jakewright/parts/"
JC_STAGING = "ops/staging/jacksoncrow/parts/"
JW_STAGING_MARKER = "ops/staging/jakewright.json"
JC_STAGING_MARKER = "ops/staging/jacksoncrow.json"
BATCH_ROWS = 250_000
FILES_PER_CHUNK = 40

__all__ = [
    "JC_CACHE_KEY",
    "JW_CACHE_KEY",
    "JW_MARKER",
    "JC_MARKER",
    "JW_STAGING",
    "JC_STAGING",
    "JW_STAGING_MARKER",
    "JC_STAGING_MARKER",
    "cache_kaggle_zip",
    "clear_staging_after_ingest",
    "ingest_jacksoncrow",
    "ingest_jakewright",
]


def clear_staging_after_ingest(
    lake: LakeStore,
    staging_prefix: str,
    staging_marker: str | None = None,
) -> int:
    n = delete_prefix(lake, staging_prefix)
    if staging_marker and lake.exists(staging_marker):
        lake.delete_keys([staging_marker])
        n += 1
    return n


def cache_kaggle_zip(lake: LakeStore, dataset: str, token: str, cache_key: str) -> str:
    if lake.exists(cache_key):
        return cache_key
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=600)
    resp.raise_for_status()
    with scratch_dir("lexis-kaggle-") as tmp:
        zpath = tmp / "dataset.zip"
        with open(zpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        lake.put_file(cache_key, zpath, "application/zip")
    return cache_key


def _write(cfg_d: dict, run_id: str, mapped: pd.DataFrame) -> list[dict]:
    lake = lake_from_cfg_d(cfg_d)
    return L1Writer(lake, run_id=run_id).write_parts(mapped, shard=uuid4().hex[:8])["details"]


@ray.remote(**DEFAULT_REMOTE_OPTS)
def ingest_staging_parts(cfg_d: dict, keys: list[str], run_id: str, source: str) -> list[dict]:
    lake = lake_from_cfg_d(cfg_d)
    details: list[dict] = []
    for key in keys:
        raw = lake.get_df_parquet(key)
        if source == "jakewright":
            symbol_col = "Ticker" if "Ticker" in raw.columns else "Symbol"
            date_col = "Date" if "Date" in raw.columns else "date"
            mapped = map_ohlcv(raw, source="jakewright", symbol_col=symbol_col, date_col=date_col)
        else:
            mapped = raw
        details.extend(_write(cfg_d, run_id, mapped))
    return details


def ingest_raydata(
    *,
    label: str,
    cfg: MarketsConfig,
    lake: LakeStore,
    marker_key: str,
    staging_prefix: str,
    staging_marker: str,
    prepare_fn,
    source: str,
) -> dict:
    t0 = time.perf_counter()
    if lake.exists(marker_key):
        out = get_json(lake, marker_key)
        chunks_n = int(out.get("chunks") or 0)
        rows = int(out.get("rows") or 0)
        avg = rows / chunks_n if chunks_n else 0
        cleared = clear_staging_after_ingest(lake, staging_prefix, staging_marker)
        logger.info(
            "%s: skip %.1fs chunks=%s avg_rows_per_chunk=%.0f cleared=%s",
            label,
            time.perf_counter() - t0,
            chunks_n,
            avg,
            cleared,
        )
        return out

    cfg_d = cfg_d_with_scratch(cfg)
    # prepare returns (n_parts, keys) so we do not rely on a second ClusterLake list_keys
    # (that list can miss keys prepare just wrote across seed workers).
    prepared = ray.get(prepare_fn.remote(cfg_d))
    if isinstance(prepared, tuple):
        n_parts, keys = int(prepared[0]), list(prepared[1])
    else:
        n_parts = int(prepared)
        keys = sorted(lake.list_keys(staging_prefix))
    if n_parts and not keys:
        keys = sorted(lake.list_keys(staging_prefix))
        logger.warning(
            "%s: prepare parts=%s but keys empty; relist keys=%s",
            label,
            n_parts,
            len(keys),
        )
    if n_parts and not keys:
        raise RuntimeError(
            f"{label}: staged {n_parts} parts but list_keys empty under {staging_prefix}"
        )
    run_id = uuid4().hex[:12]
    shape = plan_task_resources(batch_size=1)
    results = run_batches(
        label,
        keys,
        lambda batch: ingest_staging_parts.options(max_retries=2).remote(cfg_d, batch, run_id, source),
        shape,
    )
    details = merge_details(d for r in results for d in r)
    rows = sum(int(d.get("rows") or 0) for d in details)
    avg = rows / n_parts if n_parts else 0
    elapsed = time.perf_counter() - t0
    out = {
        "source": label,
        "symbols": len(details),
        "rows": rows,
        "months_written": sum(int(d.get("months_written") or 0) for d in details),
        "details": details,
        "run_id": run_id,
        "chunks": n_parts,
        "elapsed_s": elapsed,
        "avg_rows_per_chunk": avg,
    }
    put_json(lake, marker_key, out)
    cleared = clear_staging_after_ingest(lake, staging_prefix, staging_marker)
    logger.info(
        "%s_total: %.1fs parts=%s tasks=%s avg_rows_per_part=%.0f rows=%s cleared=%s",
        label,
        elapsed,
        n_parts,
        len(keys),
        avg,
        rows,
        cleared,
    )
    return out


def _member_data_name(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if n.endswith(".parquet") and not n.endswith("/")]
    if not names:
        names = [n for n in zf.namelist() if "stock" in n.lower() and n.endswith(".csv")]
    if not names:
        raise FileNotFoundError("no jakewright parquet/csv in zip")
    return names[0]


@ray.remote(**DEFAULT_REMOTE_OPTS)
def prepare_jakewright_staging(cfg_d: dict) -> tuple[int, list[str]]:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = lake_from_cfg_d(cfg_d)
    if lake.exists(JW_STAGING_MARKER):
        n = int(get_json(lake, JW_STAGING_MARKER)["parts"])
        return n, sorted(lake.list_keys(JW_STAGING))
    cache_kaggle_zip(lake, JAKEWRIGHT, cfg.kaggle_api_token, JW_CACHE_KEY)
    keys: list[str] = []
    with scratch_dir("lexis-jw-stage-") as tmp:
        zpath = tmp / "dataset.zip"
        lake.download_file(JW_CACHE_KEY, zpath)
        with zipfile.ZipFile(zpath, "r") as zf:
            member = _member_data_name(zf)
            data_path = tmp / Path(member).name
            with zf.open(member) as src, open(data_path, "wb") as dst:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)
        zpath.unlink(missing_ok=True)
        if data_path.suffix == ".parquet":
            pf = pq.ParquetFile(data_path)
            for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_ROWS)):
                part = tmp / f"part-{i:05d}.parquet"
                pq.write_table(pa.Table.from_batches([batch]), part)
                key = f"{JW_STAGING}part-{i:05d}.parquet"
                lake.put_file(key, part)
                keys.append(key)
                part.unlink(missing_ok=True)
        else:
            for i, chunk in enumerate(pd.read_csv(data_path, chunksize=BATCH_ROWS)):
                part = tmp / f"part-{i:05d}.parquet"
                chunk.to_parquet(part, index=False)
                key = f"{JW_STAGING}part-{i:05d}.parquet"
                lake.put_file(key, part)
                keys.append(key)
                part.unlink(missing_ok=True)
        data_path.unlink(missing_ok=True)
    put_json(lake, JW_STAGING_MARKER, {"parts": len(keys)})
    logger.info("jakewright staging: %s parts -> %s", len(keys), JW_STAGING)
    return len(keys), keys


@ray.remote(**DEFAULT_REMOTE_OPTS)
def prepare_jacksoncrow_staging(cfg_d: dict) -> tuple[int, list[str]]:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = lake_from_cfg_d(cfg_d)
    if lake.exists(JC_STAGING_MARKER):
        n = int(get_json(lake, JC_STAGING_MARKER)["parts"])
        return n, sorted(lake.list_keys(JC_STAGING))
    cache_kaggle_zip(lake, JACKSONCROW, cfg.kaggle_api_token, JC_CACHE_KEY)
    keys: list[str] = []
    with scratch_dir("lexis-jc-stage-") as tmp:
        zpath = tmp / "dataset.zip"
        lake.download_file(JC_CACHE_KEY, zpath)
        root = tmp / "data"
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(root)
        zpath.unlink(missing_ok=True)
        files: list[tuple[Path, str]] = []
        for path in sorted(root.rglob("*.csv")):
            parent = path.parent.name.lower()
            if parent == "stocks":
                stype = "equity"
            elif parent == "etfs":
                stype = "etf"
            else:
                continue
            files.append((path, stype))
        parts = 0
        for i in range(0, len(files), FILES_PER_CHUNK):
            frames = []
            for path, stype in files[i : i + FILES_PER_CHUNK]:
                raw = pd.read_csv(path)
                raw["Symbol"] = path.stem.upper()
                date_col = "Date" if "Date" in raw.columns else "date"
                frames.append(
                    map_ohlcv(
                        raw,
                        source="jacksoncrow",
                        symbol_col="Symbol",
                        date_col=date_col,
                        series_type=stype,
                    )
                )
                path.unlink(missing_ok=True)
            mapped = pd.concat(frames, ignore_index=True)
            part = tmp / f"part-{parts:05d}.parquet"
            mapped.to_parquet(part, index=False)
            key = f"{JC_STAGING}part-{parts:05d}.parquet"
            lake.put_file(key, part)
            keys.append(key)
            part.unlink(missing_ok=True)
            parts += 1
    put_json(lake, JC_STAGING_MARKER, {"parts": len(keys)})
    logger.info("jacksoncrow staging: %s parts -> %s", len(keys), JC_STAGING)
    return len(keys), keys


def ingest_jakewright(cfg: MarketsConfig, lake: LakeStore) -> dict:
    return ingest_raydata(
        label="jakewright",
        cfg=cfg,
        lake=lake,
        marker_key=JW_MARKER,
        staging_prefix=JW_STAGING,
        staging_marker=JW_STAGING_MARKER,
        prepare_fn=prepare_jakewright_staging,
        source="jakewright",
    )


def ingest_jacksoncrow(cfg: MarketsConfig, lake: LakeStore) -> dict:
    return ingest_raydata(
        label="jacksoncrow",
        cfg=cfg,
        lake=lake,
        marker_key=JC_MARKER,
        staging_prefix=JC_STAGING,
        staging_marker=JC_STAGING_MARKER,
        prepare_fn=prepare_jacksoncrow_staging,
        source="jacksoncrow",
    )
