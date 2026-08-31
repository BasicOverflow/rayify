from __future__ import annotations

import json
import tempfile
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import requests

from lexis_markets.cluster import plan_task_resources, run_batches
from lexis_markets.cleanup import clear_kaggle_source_build
from lexis_markets.config import MarketsConfig
from lexis_markets.storage import (
    LakeStore,
    ObsWriter,
    get_json,
    put_json,
    utcnow,
)

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

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_PER_MINUTE = 120.0
FRED_DONE_PREFIX = "ops/fred/done/"


def cache_kaggle_zip(lake: LakeStore, dataset: str, token: str, cache_key: str) -> str:
    if lake.exists(cache_key):
        return cache_key
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=600)
    resp.raise_for_status()
    with tempfile.TemporaryDirectory(prefix="lexis-kaggle-") as tmp:
        zpath = Path(tmp) / "dataset.zip"
        with open(zpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        lake.put_file(cache_key, zpath, "application/zip")
    return cache_key


def map_ohlcv(
    df: pd.DataFrame,
    *,
    source: str,
    symbol_col: str,
    date_col: str,
    series_type: str = "equity",
) -> pd.DataFrame:
    colmap = {c.lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n.lower() in colmap:
                return colmap[n.lower()]
        return None

    open_c, high_c, low_c, close_c = col("Open"), col("High"), col("Low"), col("Close")
    vol_c, adj_c = col("Volume"), col("Adj Close", "adj_close", "AdjClose")
    div_c, split_c = col("Dividends", "dividend"), col("Stock Splits", "split", "Splits")
    n = len(df)
    return pd.DataFrame(
        {
            "source": [source] * n,
            "source_symbol": df[symbol_col].astype(str).str.upper().to_numpy(),
            "series_type": [series_type] * n,
            "ts": pd.to_datetime(df[date_col]).dt.date.to_numpy(),
            "open": df[open_c].astype(float).to_numpy() if open_c else None,
            "high": df[high_c].astype(float).to_numpy() if high_c else None,
            "low": df[low_c].astype(float).to_numpy() if low_c else None,
            "close": df[close_c].astype(float).to_numpy() if close_c else None,
            "volume": df[vol_c].astype(float).to_numpy() if vol_c else None,
            "adj_close": df[adj_c].astype(float).to_numpy() if adj_c else None,
            "dividend": df[div_c].astype(float).to_numpy() if div_c else None,
            "split": df[split_c].astype(float).to_numpy() if split_c else None,
            "currency": ["USD"] * n,
            "fetched_at": [utcnow()] * n,
            "realtime_start": [None] * n,
            "realtime_end": [None] * n,
            "extras": [json.dumps({"ingest": "kaggle"})] * n,
        }
    )


def merge_details(rows: list[dict]) -> list[dict]:
    by: dict[str, dict] = {}
    for r in rows:
        if not r.get("symbol"):
            continue
        sym = str(r["symbol"]).upper()
        if sym not in by:
            by[sym] = {**r, "symbol": sym}
            continue
        cur = by[sym]
        cur["rows"] = int(cur.get("rows") or 0) + int(r.get("rows") or 0)
        cur["months_written"] = int(cur.get("months_written") or 0) + int(r.get("months_written") or 0)
        if r.get("first"):
            cur["first"] = min(str(cur.get("first") or r["first"]), str(r["first"]))
        if r.get("last"):
            cur["last"] = max(str(cur.get("last") or r["last"]), str(r["last"]))
    return list(by.values())


def _write(cfg_d: dict, run_id: str, mapped: pd.DataFrame) -> list[dict]:
    return ObsWriter(LakeStore(MarketsConfig.from_dict(cfg_d)), run_id=run_id).write_parts(
        mapped, shard=uuid4().hex[:8]
    )["details"]


@ray.remote
def ingest_staging_parts(cfg_d: dict, keys: list[str], run_id: str, source: str) -> list[dict]:
    lake = LakeStore(MarketsConfig.from_dict(cfg_d))
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
    cache_key: str,
    prepare_fn,
    source: str,
) -> dict:
    t0 = time.perf_counter()
    if lake.exists(marker_key):
        out = get_json(lake, marker_key)
        chunks_n = int(out.get("chunks") or 0)
        rows = int(out.get("rows") or 0)
        avg = rows / chunks_n if chunks_n else 0
        cleared = clear_kaggle_source_build(lake, staging_prefix, staging_marker, cache_key)
        print(
            f"{label}: skip {time.perf_counter() - t0:.1f}s chunks={chunks_n} "
            f"avg_rows_per_chunk={avg:.0f} cleared={cleared}"
        )
        return out

    cfg_d = cfg.to_dict()
    n_parts = int(ray.get(prepare_fn.remote(cfg_d)))
    keys = sorted(lake.list_keys(staging_prefix))
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
    cleared = clear_kaggle_source_build(lake, staging_prefix, staging_marker, cache_key)
    print(
        f"{label}_total: {elapsed:.1f}s parts={n_parts} tasks={len(keys)} "
        f"avg_rows_per_part={avg:.0f} rows={rows} cleared={cleared}"
    )
    return out


def _member_data_name(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if n.endswith(".parquet") and not n.endswith("/")]
    if not names:
        names = [n for n in zf.namelist() if "stock" in n.lower() and n.endswith(".csv")]
    if not names:
        raise FileNotFoundError("no jakewright parquet/csv in zip")
    return names[0]


@ray.remote
def prepare_jakewright_staging(cfg_d: dict) -> int:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    if lake.exists(JW_STAGING_MARKER):
        return int(get_json(lake, JW_STAGING_MARKER)["parts"])
    cache_kaggle_zip(lake, JAKEWRIGHT, cfg.kaggle_api_token, JW_CACHE_KEY)
    parts = 0
    with tempfile.TemporaryDirectory(prefix="lexis-jw-stage-") as tmp:
        tmp = Path(tmp)
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
        if data_path.suffix == ".parquet":
            pf = pq.ParquetFile(data_path)
            for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_ROWS)):
                part = tmp / f"part-{i:05d}.parquet"
                pq.write_table(pa.Table.from_batches([batch]), part)
                lake.put_file(f"{JW_STAGING}part-{i:05d}.parquet", part)
                parts += 1
        else:
            for i, chunk in enumerate(pd.read_csv(data_path, chunksize=BATCH_ROWS)):
                part = tmp / f"part-{i:05d}.parquet"
                chunk.to_parquet(part, index=False)
                lake.put_file(f"{JW_STAGING}part-{i:05d}.parquet", part)
                parts += 1
    put_json(lake, JW_STAGING_MARKER, {"parts": parts})
    print(f"jakewright staging: {parts} parts -> {JW_STAGING}")
    return parts


@ray.remote
def prepare_jacksoncrow_staging(cfg_d: dict) -> int:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    if lake.exists(JC_STAGING_MARKER):
        return int(get_json(lake, JC_STAGING_MARKER)["parts"])
    cache_kaggle_zip(lake, JACKSONCROW, cfg.kaggle_api_token, JC_CACHE_KEY)
    parts = 0
    with tempfile.TemporaryDirectory(prefix="lexis-jc-stage-") as tmp:
        tmp = Path(tmp)
        zpath = tmp / "dataset.zip"
        lake.download_file(JC_CACHE_KEY, zpath)
        root = tmp / "data"
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(root)
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
            mapped = pd.concat(frames, ignore_index=True)
            part = tmp / f"part-{parts:05d}.parquet"
            mapped.to_parquet(part, index=False)
            lake.put_file(f"{JC_STAGING}part-{parts:05d}.parquet", part)
            parts += 1
    put_json(lake, JC_STAGING_MARKER, {"parts": parts})
    print(f"jacksoncrow staging: {parts} parts -> {JC_STAGING}")
    return parts


def ingest_jakewright(cfg: MarketsConfig, lake: LakeStore) -> dict:
    return ingest_raydata(
        label="jakewright",
        cfg=cfg,
        lake=lake,
        marker_key=JW_MARKER,
        staging_prefix=JW_STAGING,
        staging_marker=JW_STAGING_MARKER,
        cache_key=JW_CACHE_KEY,
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
        cache_key=JC_CACHE_KEY,
        prepare_fn=prepare_jacksoncrow_staging,
        source="jacksoncrow",
    )


@ray.remote
class FredRateGate:
    def __init__(self, per_minute: float = FRED_PER_MINUTE):
        self.min_interval = 60.0 / per_minute
        self._next = 0.0


    def acquire(self):
        now = time.monotonic()
        wait = self._next - now
        if wait > 0:
            time.sleep(wait)
        self._next = time.monotonic() + self.min_interval


def fred_done_key(series_id: str) -> str:
    return f"{FRED_DONE_PREFIX}{series_id.upper()}.json"


def fred_done_symbols(lake: LakeStore) -> set[str]:
    return {k.rsplit("/", 1)[-1].removesuffix(".json") for k in lake.list_keys(FRED_DONE_PREFIX)}


def fred_done_details(lake: LakeStore) -> list[dict]:
    return [get_json(lake, key) for key in lake.list_keys(FRED_DONE_PREFIX)]


def fred_mark_done(lake: LakeStore, detail: dict) -> None:
    sym = str(detail["symbol"]).upper()
    put_json(lake, fred_done_key(sym), detail)


@ray.remote
def _fred_stats_in_month(cfg_d: dict, month_prefix: str) -> dict[str, dict]:
    lake = LakeStore(MarketsConfig.from_dict(cfg_d))
    stats: dict[str, dict] = {}
    for key in lake.list_keys(month_prefix):
        if not key.endswith(".parquet"):
            continue
        df = lake.get_df_parquet(key, columns=["source", "source_symbol", "ts"])
        fred = df[df["source"] == "fred"]
        if fred.empty:
            continue
        fred = fred.copy()
        fred["ts"] = pd.to_datetime(fred["ts"])
        for sym, g in fred.groupby("source_symbol", sort=False):
            sid = str(sym).upper()
            first, last = g["ts"].min().date(), g["ts"].max().date()
            rows = int(len(g))
            if sid not in stats:
                stats[sid] = {"first": first, "last": last, "rows": rows}
            else:
                cur = stats[sid]
                cur["first"] = min(cur["first"], first)
                cur["last"] = max(cur["last"], last)
                cur["rows"] += rows
    return stats


def bootstrap_fred_done_from_lake(cfg_d: dict) -> int:
    lake = LakeStore(MarketsConfig.from_dict(cfg_d))
    if fred_done_symbols(lake):
        return 0
    if not lake.list_common_prefixes("layer_1/"):
        return 0
    print("fred: bootstrapping done markers from lake...", flush=True)
    months: list[str] = []
    for yp in lake.list_common_prefixes("layer_1/"):
        months.extend(lake.list_common_prefixes(yp))
    merged: dict[str, dict] = {}
    for part in ray.get([_fred_stats_in_month.remote(cfg_d, mp) for mp in months]):
        for sid, st in part.items():
            if sid not in merged:
                merged[sid] = st
            else:
                cur = merged[sid]
                cur["first"] = min(cur["first"], st["first"])
                cur["last"] = max(cur["last"], st["last"])
                cur["rows"] += st["rows"]
    for sid, st in merged.items():
        fred_mark_done(
            lake,
            {
                "symbol": sid,
                "source": "fred",
                "series_type": "macro",
                "rows": st["rows"],
                "months_written": 0,
                "first": str(st["first"]),
                "last": str(st["last"]),
            },
        )
    print(f"fred: bootstrapped {len(merged)} done markers", flush=True)
    return len(merged)


def _fred_get(params: dict, rate_gate=None) -> requests.Response:
    for attempt in range(8):
        if rate_gate is not None:
            ray.get(rate_gate.acquire.remote())
        resp = requests.get(FRED_OBS_URL, params=params, timeout=120)
        if resp.status_code == 429:
            time.sleep(min(2 ** attempt, 60))
            continue
        return resp
    return resp


def _observations_to_df(series_id: str, observations: list, *, mode: str, as_of: str | None) -> pd.DataFrame:
    rows = []
    for o in observations:
        if o.get("value") in (".", None, ""):
            continue
        rows.append(
            {
                "source": "fred",
                "source_symbol": series_id,
                "series_type": "macro",
                "ts": o["date"],
                "open": None,
                "high": None,
                "low": None,
                "close": float(o["value"]),
                "volume": None,
                "adj_close": None,
                "dividend": None,
                "split": None,
                "currency": "USD",
                "fetched_at": utcnow(),
                "realtime_start": o.get("realtime_start", as_of),
                "realtime_end": o.get("realtime_end", as_of),
                "extras": json.dumps({"mode": mode, "as_of": as_of}),
            }
        )
    return pd.DataFrame(rows)


def fetch_alfred_series(series_id: str, api_key: str, rate_gate=None) -> pd.DataFrame:
    as_of = (date.today() - timedelta(days=1)).isoformat()
    base = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    resp = _fred_get({**base, "realtime_start": as_of, "realtime_end": as_of}, rate_gate)
    if resp.status_code == 400:
        resp = _fred_get(base, rate_gate)
        resp.raise_for_status()
        return _observations_to_df(series_id, resp.json().get("observations", []), mode="fred_full", as_of=None)
    resp.raise_for_status()
    return _observations_to_df(
        series_id, resp.json().get("observations", []), mode="alfred_as_of", as_of=as_of
    )


def fetch_release_series_ids(release_id: int, api_key: str, rate_gate=None) -> list[str]:
    url = "https://api.stlouisfed.org/fred/release/series"
    params = {"release_id": release_id, "api_key": api_key, "file_type": "json", "limit": 1000}
    if rate_gate is not None:
        ray.get(rate_gate.acquire.remote())
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return [s["id"] for s in resp.json().get("seriess", [])]


def resolve_fred_series_ids(cfg: MarketsConfig, rate_gate=None) -> list[str]:
    ids = list(cfg.fred_series)
    for rid in cfg.fred_release_ids:
        ids.extend(fetch_release_series_ids(rid, cfg.fred_api_key, rate_gate=rate_gate))
    return sorted(set(ids))


def ingest_fred_series(
    lake: LakeStore,
    api_key: str,
    series_ids: list[str],
    *,
    run_id: str | None = None,
    rate_gate=None,
) -> dict:
    writer = ObsWriter(lake, run_id=run_id or uuid4().hex[:12])
    details = []
    rows = months = ok = failed = 0
    frames = []
    for sid in series_ids:
        df = fetch_alfred_series(sid, api_key, rate_gate=rate_gate)
        if df.empty:
            failed += 1
            details.append(
                {"symbol": sid, "rows": 0, "months_written": 0, "source": "fred", "series_type": "macro"}
            )
            continue
        frames.append(df)
        ok += 1
    if frames:
        meta = writer.write_parts(pd.concat(frames, ignore_index=True), shard=uuid4().hex[:8])
        rows = meta["rows"]
        months = meta["months_written"]
        for d in meta["details"]:
            fred_mark_done(lake, d)
        details.extend(meta["details"])
    return {
        "source": "fred",
        "symbols": len(series_ids),
        "ok": ok,
        "failed": failed,
        "rows": rows,
        "months_written": months,
        "details": details,
    }


@ray.remote
def task_ingest_fred_batch(cfg_d: dict, series_ids: list[str], run_id: str, rate_gate) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    return ingest_fred_series(
        LakeStore(cfg), cfg.fred_api_key, series_ids, run_id=run_id, rate_gate=rate_gate
    )
