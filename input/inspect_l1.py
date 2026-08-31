"""Sample stitched L1 series plots + lake-wide quality stats. Run from output/."""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lexis_markets.api import MarketsService
from lexis_markets.config import MarketsConfig
from lexis_markets.storage import LakeStore, PgClient

SOURCE_COLORS = {
    "jakewright": "#2563eb",
    "jacksoncrow": "#dc2626",
    "fred": "#16a34a",
    "marketparquet": "#0d9488",
    "yfinance": "#ea580c",
}
DEFAULT_COLOR = "#6b7280"

SCAN_COLS = ["source", "source_symbol", "series_type", "ts", "close"]
PLOT_TYPES = ("equity", "etf", "macro")


def l1_month_prefixes(lake: LakeStore) -> list[str]:
    out: list[str] = []
    for yp in lake.list_common_prefixes("layer_1/"):
        out.extend(lake.list_common_prefixes(yp))
    return sorted(out)


def merge_symbol_stats(master: dict, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    g = df.groupby(["source", "series_type", "source_symbol"], as_index=False).agg(
        rows=("ts", "count"),
        days=("ts", "nunique"),
        null_close=("close", lambda s: int(s.isna().sum())),
        first=("ts", "min"),
        last=("ts", "max"),
    )
    for r in g.itertuples(index=False):
        key = (r.source, r.series_type, r.source_symbol)
        dup = int(r.rows - r.days)
        if key not in master:
            master[key] = {
                "rows": 0,
                "days": 0,
                "null_close": 0,
                "dup_rows": 0,
                "first": r.first,
                "last": r.last,
            }
        m = master[key]
        m["rows"] += int(r.rows)
        m["days"] += int(r.days)
        m["null_close"] += int(r.null_close)
        m["dup_rows"] += dup
        if r.first < m["first"]:
            m["first"] = r.first
        if r.last > m["last"]:
            m["last"] = r.last


def merge_masters(into: dict, part: dict) -> None:
    for key, m in part.items():
        if key not in into:
            into[key] = {
                "rows": m["rows"],
                "days": m["days"],
                "null_close": m["null_close"],
                "dup_rows": m["dup_rows"],
                "first": m["first"],
                "last": m["last"],
            }
            continue
        dst = into[key]
        dst["rows"] += m["rows"]
        dst["days"] += m["days"]
        dst["null_close"] += m["null_close"]
        dst["dup_rows"] += m["dup_rows"]
        if m["first"] < dst["first"]:
            dst["first"] = m["first"]
        if m["last"] > dst["last"]:
            dst["last"] = m["last"]


def _scan_month(lake: LakeStore, mp: str, file_workers: int) -> tuple[int, dict]:
    keys = [k for k in lake.list_keys(mp) if k.endswith(".parquet")]
    partial: dict = {}
    for df in lake.get_dfs_parquet_parallel(keys, columns=SCAN_COLS, max_workers=file_workers):
        merge_symbol_stats(partial, df)
    return len(keys), partial


def scan_l1(lake: LakeStore, *, month_workers: int = 16, file_workers: int = 16) -> dict:
    prefixes = l1_month_prefixes(lake)
    master: dict = {}
    file_count = 0
    t0 = time.perf_counter()
    workers = min(month_workers, len(prefixes) or 1)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_scan_month, lake, mp, file_workers): mp for mp in prefixes}
        for fut in as_completed(futs):
            nfiles, partial = fut.result()
            file_count += nfiles
            merge_masters(master, partial)
            done += 1
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed else 0
            print(
                f"scan: {done}/{len(prefixes)} months parts={file_count} "
                f"series={len(master)} ({rate:.1f} mo/s, {elapsed:.0f}s)",
                flush=True,
            )
    print(f"scan: done {file_count} parts, {len(master)} source-symbol series", flush=True)
    return {"parquet_files": file_count, "symbols": master}


def enrich_stats(symbols: dict) -> list[dict]:
    rows_out = []
    for (source, series_type, symbol), m in symbols.items():
        first = pd.Timestamp(m["first"]).date()
        last = pd.Timestamp(m["last"]).date()
        if series_type in ("equity", "etf"):
            span = int(np.busday_count(first, last)) + 1
        else:
            span = (last - first).days + 1
        span = max(span, 1)
        missing = max(0, span - int(m["days"]))
        density = m["days"] / span if span else 0.0
        rows_out.append(
            {
                "source": source,
                "series_type": series_type,
                "symbol": symbol,
                "rows": m["rows"],
                "unique_days": m["days"],
                "span_days": span,
                "missing_est": missing,
                "density": round(density, 4),
                "dup_rows": m["dup_rows"],
                "null_close": m["null_close"],
                "first": str(first),
                "last": str(last),
            }
        )
    return rows_out


def registry_gap_summary(pg: PgClient) -> dict:
    row = pg.fetchone(
        """
        SELECT COUNT(*) AS series,
               COALESCE(SUM(gap_count), 0) AS gap_total,
               COALESCE(SUM(CASE WHEN gap_count > 0 THEN 1 ELSE 0 END), 0) AS gap_nonzero,
               AVG(quality_score) AS avg_quality
        FROM series_meta
        """
    )
    by_class = pg.fetchall(
        """
        SELECT asset_class,
               COUNT(*) AS series,
               COALESCE(SUM(gap_count), 0) AS gap_total,
               COALESCE(SUM(CASE WHEN gap_count > 0 THEN 1 ELSE 0 END), 0) AS gap_nonzero,
               AVG(quality_score) AS avg_quality
        FROM series_meta
        GROUP BY asset_class
        ORDER BY asset_class
        """
    )
    return {"overall": row, "by_asset_class": by_class}


def summarize(stats_rows: list[dict], parquet_files: int, stitched: dict | None = None) -> dict:
    df = pd.DataFrame(stats_rows)
    by_source = df.groupby("source").agg(
        symbols=("symbol", "count"),
        rows=("rows", "sum"),
        missing=("missing_est", "sum"),
        dup_rows=("dup_rows", "sum"),
        null_close=("null_close", "sum"),
        avg_density=("density", "mean"),
    )
    by_type = df.groupby("series_type").agg(
        symbols=("symbol", "count"),
        rows=("rows", "sum"),
        missing=("missing_est", "sum"),
        avg_density=("density", "mean"),
    )
    low_density = df[df["density"] < 0.85].sort_values("density").head(25)
    high_dup = df[df["dup_rows"] > 0].sort_values("dup_rows", ascending=False).head(25)
    return {
        "parquet_files": parquet_files,
        "source_symbol_series": len(df),
        "total_rows": int(df["rows"].sum()),
        "total_unique_days": int(df["unique_days"].sum()),
        "total_missing_est": int(df["missing_est"].sum()),
        "total_dup_rows": int(df["dup_rows"].sum()),
        "total_null_close": int(df["null_close"].sum()),
        "by_source": by_source.reset_index().to_dict(orient="records"),
        "by_series_type": by_type.reset_index().to_dict(orient="records"),
        "low_density_sample": low_density.to_dict(orient="records"),
        "high_dup_sample": high_dup.to_dict(orient="records"),
        "date_range": {
            "first": str(df["first"].min()) if len(df) else None,
            "last": str(df["last"].max()) if len(df) else None,
        },
        "stitched_gaps": stitched,
    }


def write_quality_report(out_dir: Path, stats_rows: list[dict], summary: dict) -> None:
    (out_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "L1 ingest quality report",
        "=" * 40,
        f"parquet files: {summary['parquet_files']}",
        f"source-symbol series: {summary['source_symbol_series']}",
        f"total rows: {summary['total_rows']:,}",
        f"missing days (est): {summary['total_missing_est']:,}",
        f"duplicate rows (same-day): {summary['total_dup_rows']:,}",
        f"null close: {summary['total_null_close']:,}",
        f"date range: {summary['date_range']['first']} .. {summary['date_range']['last']}",
        "",
        "by source:",
    ]
    for r in summary["by_source"]:
        lines.append(
            f"  {r['source']:12s} symbols={r['symbols']:5d} rows={r['rows']:10,d} "
            f"missing={r['missing']:8,d} dup={r['dup_rows']:6,d} avg_density={r['avg_density']:.3f}"
        )
    lines.append("")
    lines.append("by series_type:")
    for r in summary["by_series_type"]:
        lines.append(
            f"  {r['series_type']:8s} symbols={r['symbols']:5d} rows={r['rows']:10,d} "
            f"missing={r['missing']:8,d} avg_density={r['avg_density']:.3f}"
        )
    lines.append("")
    if summary.get("stitched_gaps"):
        sg = summary["stitched_gaps"]["overall"]
        lines.append("stitched L3 gaps (registry):")
        lines.append(
            f"  series={sg['series']} gap_nonzero={sg['gap_nonzero']} gap_total={sg['gap_total']} "
            f"avg_quality={float(sg['avg_quality'] or 0):.3f}"
        )
        for r in summary["stitched_gaps"]["by_asset_class"]:
            lines.append(
                f"  {r['asset_class']:8s} series={r['series']:5d} gap_nonzero={r['gap_nonzero']:5d} "
                f"gap_total={r['gap_total']:8d} avg_quality={float(r['avg_quality'] or 0):.3f}"
            )
        lines.append("")
    lines.append("worst density (likely gaps):")
    for r in summary["low_density_sample"][:15]:
        lines.append(
            f"  {r['source']}/{r['series_type']}/{r['symbol']} density={r['density']:.3f} "
            f"days={r['unique_days']}/{r['span_days']} {r['first']}..{r['last']}"
        )
    (out_dir / "quality_report.txt").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    src_df = pd.DataFrame(summary["by_source"])
    if not src_df.empty:
        axes[0].bar(src_df["source"], src_df["rows"], color=[SOURCE_COLORS.get(s, DEFAULT_COLOR) for s in src_df["source"]])
        axes[0].set_title("rows by source")
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].bar(src_df["source"], src_df["missing"], color=[SOURCE_COLORS.get(s, DEFAULT_COLOR) for s in src_df["source"]])
        axes[1].set_title("missing days (est) by source")
        axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_by_source.png", dpi=120)
    plt.close(fig)

    type_df = pd.DataFrame(stats_rows)
    if not type_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(type_df["density"], bins=40, color="#64748b", edgecolor="white")
        ax.set_title("density distribution (unique_days / span)")
        ax.set_xlabel("density")
        fig.tight_layout()
        fig.savefig(out_dir / "density_histogram.png", dpi=120)
        plt.close(fig)


def _series_spec(series_id: str, start: date, end: date) -> dict:
    return {
        "series_ids": [series_id],
        "start": str(start),
        "end": str(end),
        "granularity": "daily",
        "revision_mode": "as_of",
    }


def fetch_l3_series_local(
    cfg: MarketsConfig, lake: LakeStore, pg: PgClient, series_id: str, start: date, end: date
) -> pd.DataFrame:
    svc = MarketsService(cfg)
    svc.lake = lake
    svc.pg = pg
    bars = svc.query_stitched_bars(_series_spec(series_id, start, end))
    if bars.empty:
        return pd.DataFrame(columns=["ts", "close", "source", "source_count"])
    df = bars.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts")


async def fetch_l3_series_api(
    client: httpx.AsyncClient, cfg: MarketsConfig, series_id: str, start: date, end: date
) -> pd.DataFrame:
    url = f"{cfg.ray_serve_url.rstrip('/')}/v1/series/{quote(series_id, safe='')}"
    resp = await client.get(url, params={"start": str(start), "end": str(end)}, timeout=600.0)
    resp.raise_for_status()
    rows = resp.json().get("rows") or []
    if not rows:
        return pd.DataFrame(columns=["ts", "close", "source", "source_count"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts")


def source_runs(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    df = df.sort_values("ts")
    if df.empty:
        return []
    dates = df["ts"].to_numpy()
    sources = df["source"].astype(str).to_numpy()
    runs = []
    start = 0
    for i in range(1, len(dates)):
        if sources[i] != sources[i - 1]:
            runs.append((dates[start], dates[i - 1], sources[start]))
            start = i
    runs.append((dates[start], dates[-1], sources[start]))
    return runs


def plot_price(df: pd.DataFrame) -> pd.Series:
    return df["close"].astype(float)


def plot_stitched(ax, df: pd.DataFrame, title: str) -> None:
    df = df.sort_values("ts")
    y = plot_price(df)
    for a, b, who in source_runs(df):
        mask = (df["ts"] >= a) & (df["ts"] <= b)
        color = SOURCE_COLORS.get(who, DEFAULT_COLOR)
        ax.plot(df.loc[mask, "ts"], y.loc[mask], color=color, lw=2.0)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)
    if len(df) > 0:
        multi = int((df.get("source_count", 1) > 1).sum()) if "source_count" in df.columns else 0
        ax.text(
            0.02, 0.98,
            f"days={len(df)} sources={df['source'].nunique()} multi={multi}",
            transform=ax.transAxes, va="top", fontsize=7,
        )


def sample_pool(pg: PgClient, asset_class: str, *, multi_source: bool, prefer_eod: bool) -> pd.DataFrame:
    if prefer_eod:
        rows = pg.fetchall(
            """
            SELECT m.series_id, m.canonical_symbol, m.first_seen, m.last_seen
            FROM series_meta m
            WHERE m.asset_class = %s AND m.last_seen > '2020-04-01'
              AND EXISTS (
                SELECT 1 FROM stitch_segments s
                WHERE s.series_id = m.series_id AND s.source IN ('marketparquet', 'yfinance')
              )
            """,
            (asset_class,),
        )
        if rows:
            return pd.DataFrame(rows)
    if asset_class == "equity" and multi_source:
        rows = pg.fetchall(
            """
            SELECT series_id, canonical_symbol, first_seen, last_seen
            FROM series_meta m
            WHERE asset_class = 'equity'
              AND EXISTS (SELECT 1 FROM symbol_aliases a WHERE a.series_id = m.series_id AND a.source = 'jakewright')
              AND EXISTS (SELECT 1 FROM symbol_aliases a WHERE a.series_id = m.series_id AND a.source = 'jacksoncrow')
            """
        )
    else:
        rows = pg.fetchall(
            """
            SELECT series_id, canonical_symbol, first_seen, last_seen
            FROM series_meta WHERE asset_class = %s
            """,
            (asset_class,),
        )
    return pd.DataFrame(rows)


def _save_plot(df: pd.DataFrame, out_path: Path, sid: str, sym: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_stitched(ax, df, f"{sid} L3 stitched (close)")
    ax.set_xlabel("date")
    ax.set_ylabel("close")
    handles = [
        Line2D([0], [0], color=SOURCE_COLORS.get(s, DEFAULT_COLOR), lw=2, label=s)
        for s in sorted(df["source"].astype(str).unique())
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_window(row, plot_years: int | None) -> tuple[date, date]:
    end = row["last_seen"]
    start = row["first_seen"]
    if plot_years:
        start = max(start, end - timedelta(days=plot_years * 365))
    return start, end


async def plot_samples(
    cfg: MarketsConfig,
    lake: LakeStore,
    pg: PgClient,
    out_dir: Path,
    *,
    n: int,
    seed: int,
    multi_source: bool,
    prefer_eod: bool,
    plot_years: int | None,
    via_api: bool,
    workers: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tasks_meta: list[tuple] = []

    for asset_class in PLOT_TYPES:
        pool = sample_pool(pg, asset_class, multi_source=multi_source, prefer_eod=prefer_eod)
        if pool.empty:
            print(f"plots: skip {asset_class} (none in registry)")
            continue
        pick = pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)
        class_dir = plot_dir / asset_class
        class_dir.mkdir(parents=True, exist_ok=True)
        for _, row in pick.iterrows():
            safe = str(row["canonical_symbol"]).replace("/", "_").replace("\\", "_")
            out_path = class_dir / f"{safe}_{row['series_id'].replace(':', '_')}.png"
            tasks_meta.append((asset_class, row, out_path))

    total = len(tasks_meta)
    workers_label = workers if workers > 0 else total
    print(f"plots: {total} jobs workers={workers_label}", flush=True)
    by_class: dict[str, int] = {}
    sem = asyncio.Semaphore(workers) if workers > 0 else None

    async with httpx.AsyncClient() as client:

        async def run_one(asset_class: str, row, out_path: Path):
            async def _work():
                sid = row["series_id"]
                sym = row["canonical_symbol"]
                pstart, pend = plot_window(row, plot_years)
                try:
                    if via_api:
                        df = await fetch_l3_series_api(client, cfg, sid, pstart, pend)
                    else:
                        df = await asyncio.to_thread(
                            fetch_l3_series_local, cfg, lake, pg, sid, pstart, pend
                        )
                except Exception as e:
                    return asset_class, sym, None, str(e)
                if df.empty:
                    return asset_class, sym, None, "empty"
                await asyncio.to_thread(_save_plot, df, out_path, sid, sym)
                return asset_class, sym, int(df["source"].nunique()), None

            if sem is None:
                return await _work()
            async with sem:
                return await _work()

        tasks = [asyncio.create_task(run_one(ac, row, p)) for ac, row, p in tasks_meta]
        done = 0
        for fut in asyncio.as_completed(tasks):
            asset_class, sym, ns, err = await fut
            done += 1
            if ns is None:
                print(f"plots: [{done}/{total}] skip {asset_class}/{sym} ({err})", flush=True)
                continue
            by_class[asset_class] = by_class.get(asset_class, 0) + 1
            print(f"plots: [{done}/{total}] {asset_class}/{sym} sources={ns}", flush=True)
    for asset_class, n_written in by_class.items():
        print(f"plots: {asset_class} wrote {n_written} images", flush=True)


def run_cluster_inspect(
    cfg: MarketsConfig,
    pg: PgClient,
    out_dir: Path,
    *,
    n: int,
    seed: int,
    prefer_eod: bool,
    plot_years: int | None,
    via_api: bool = True,
) -> None:
    from lexis_markets.api import ensure_markets_api
    from lexis_markets.cluster import init_ray
    from lexis_markets.cluster_plots import run_cluster_plots

    init_ray(cfg)
    if via_api:
        ensure_markets_api(cfg)
    plot_dir = out_dir / "plots"
    jobs: list[tuple[str, str, date, date, Path]] = []

    for asset_class in PLOT_TYPES:
        pool = sample_pool(pg, asset_class, multi_source=True, prefer_eod=prefer_eod)
        if pool.empty:
            print(f"plots: skip {asset_class} (none in registry)", flush=True)
            continue
        pick = pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)
        class_dir = plot_dir / asset_class
        class_dir.mkdir(parents=True, exist_ok=True)
        for _, row in pick.iterrows():
            pstart, pend = plot_window(row, plot_years)
            safe = str(row["canonical_symbol"]).replace("/", "_").replace("\\", "_")
            fname = f"{safe}_{row['series_id'].replace(':', '_')}.png"
            jobs.append((asset_class, row["series_id"], pstart, pend, class_dir / fname))

    print(f"plots: cluster render {len(jobs)} jobs via_api={via_api}", flush=True)
    results = run_cluster_plots(
        cfg,
        [(sid, start, end, path.name) for _, sid, start, end, path in jobs],
        via_api=via_api,
    )
    by_name = {(r["series_id"], r["fname"]): r for r in results}
    done = 0
    total = len(jobs)
    for asset_class, sid, _, _, path in jobs:
        done += 1
        r = by_name.get((sid, path.name))
        if not r or not r.get("png"):
            print(f"plots: [{done}/{total}] skip {asset_class}/{sid}", flush=True)
            continue
        path.write_bytes(r["png"])
        print(
            f"plots: [{done}/{total}] {asset_class}/{sid} rows={r.get('rows')} -> {path.name}",
            flush=True,
        )


def plot_samples_sync(*args, **kwargs) -> None:
    asyncio.run(plot_samples(*args, **kwargs))


def main():
    p = argparse.ArgumentParser(description="Inspect L1 ingest quality and sample stitched plots")
    p.add_argument("--out", type=Path, default=Path("l1_inspect"))
    p.add_argument("--samples", type=int, default=35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh-scan", action="store_true", help="ignore scan cache")
    p.add_argument("--skip-scan", action="store_true", help="skip quality scan (use cache or existing reports)")
    p.add_argument("--local", action="store_true", help="plot on this machine instead of Ray workers")
    p.add_argument("--direct", action="store_true", help="skip HTTP API; query MarketsService directly")
    p.add_argument("--workers", type=int, default=0, help="local plot concurrency (0 = all at once)")
    p.add_argument("--scan-workers", type=int, default=0, help="parallel month scans (0 = MINIO_MAX_WORKERS)")
    p.add_argument("--file-workers", type=int, default=0, help="parallel MinIO reads (0 = MINIO_MAX_WORKERS)")
    p.add_argument("--plot-years", type=int, default=5, help="cap history per plot (0 = full first_seen..last_seen)")
    p.add_argument("--prefer-eod", action="store_true", help="bias samples toward series with live EOD data")
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args()

    cfg = MarketsConfig.from_env()
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    args.out.mkdir(parents=True, exist_ok=True)
    scan_workers = args.scan_workers or cfg.minio_max_workers
    file_workers = args.file_workers or cfg.minio_max_workers

    if not args.skip_scan:
        cache_path = args.out / "scan_master.pkl"
        if cache_path.exists() and not args.refresh_scan:
            master = pickle.loads(cache_path.read_bytes())
            print(f"scan: loaded cache {len(master)} series", flush=True)
            scanned = {"parquet_files": 0, "symbols": master}
        else:
            print(f"scan: start month_workers={scan_workers} file_workers={file_workers}", flush=True)
            scanned = scan_l1(lake, month_workers=scan_workers, file_workers=file_workers)
            cache_path.write_bytes(pickle.dumps(scanned["symbols"]))
        print("quality: enrich + report", flush=True)
        stats_rows = enrich_stats(scanned["symbols"])
        stitched = registry_gap_summary(pg)
        summary = summarize(stats_rows, scanned["parquet_files"], stitched=stitched)
        write_quality_report(args.out, stats_rows, summary)
        pd.DataFrame(stats_rows).to_csv(args.out / "symbol_stats.csv", index=False)
        print(f"quality: {args.out / 'quality_report.txt'}", flush=True)

    if not args.skip_plots:
        via_api = not args.direct
        if args.local:
            plot_samples_sync(
                cfg, lake, pg, args.out,
                n=args.samples, seed=args.seed, multi_source=True,
                prefer_eod=args.prefer_eod,
                plot_years=args.plot_years or None,
                via_api=via_api, workers=args.workers,
            )
        else:
            run_cluster_inspect(
                cfg, pg, args.out,
                n=args.samples, seed=args.seed, prefer_eod=args.prefer_eod,
                plot_years=args.plot_years or None,
                via_api=via_api,
            )
        print(f"plots: {args.out / 'plots'}", flush=True)


if __name__ == "__main__":
    main()
