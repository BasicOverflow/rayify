"""Generate stitched-series plot samples for visual QA. Writes PNGs under ``--out/plots/``."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from lexis_markets.lake import PgClient
from lexis_markets.registry.filters import effective_last_seen as registry_last_seen
from lexis_markets.cli._runner import cli_main
from lexis_markets.domain.quality import (
    SPARSE_BRIDGE_MIN_DAYS,
    suspicious_from_bars,
)
from lexis_markets.jobs.clock import eod_target_date

SOURCE_COLORS = {
    "jakewright": "#2563eb",
    "jacksoncrow": "#dc2626",
    "fred": "#16a34a",
    "marketparquet": "#0d9488",
    "yfinance": "#ea580c",
}
DEFAULT_COLOR = "#6b7280"
FLAG_COLOR = "#f59e0b"
PLOT_TYPES = ("equity", "etf", "macro")


def effective_last_seen(row: dict) -> date:
    """Plot end: registry effective last, capped at the current EOD target day."""
    last = registry_last_seen(row)
    if last is None:
        last = row["last_seen"]
    return min(last, eod_target_date())


def plot_window(row: dict, plot_years: int | None) -> tuple[date, date]:
    end = effective_last_seen(row)
    start = row["first_seen"]
    if plot_years:
        start = max(start, end - timedelta(days=plot_years * 365))
    return start, end


def source_runs(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    df = df.sort_values("ts")
    if df.empty:
        return []
    dates = df["ts"].to_numpy()
    sources = df["source"].astype(str).to_numpy()
    runs: list[tuple] = []
    start = 0
    for i in range(1, len(dates)):
        if sources[i] != sources[i - 1]:
            runs.append((dates[start], dates[i - 1], sources[start]))
            start = i
    runs.append((dates[start], dates[-1], sources[start]))
    return runs


def plot_stitched(ax, df: pd.DataFrame, title: str) -> None:
    from lexis_markets.domain.quality import linear_ramp_mask, sparse_bridge_gap_days

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    y = df["close"].astype(float)
    suspicious, flag_names = suspicious_from_bars(df)
    if suspicious:
        ramp = linear_ramp_mask(y.to_numpy(dtype=float))
        _bridge_days, bridge_touch = sparse_bridge_gap_days(df["ts"].to_numpy())
        highlight = ramp | bridge_touch
        i = 0
        while i < len(highlight):
            if not highlight[i]:
                i += 1
                continue
            j = i
            while j < len(highlight) and highlight[j]:
                j += 1
            ax.axvspan(df["ts"].iloc[i], df["ts"].iloc[j - 1], color=FLAG_COLOR, alpha=0.2, lw=0)
            i = j
        # Draw dashed lines across sparse bridges so the fake diagonal is explicit.
        ts_vals = df["ts"].to_numpy()
        for k in range(1, len(ts_vals)):
            gap = int(pd.Timedelta(ts_vals[k] - ts_vals[k - 1]).days)
            if gap >= SPARSE_BRIDGE_MIN_DAYS:
                ax.plot(
                    [df["ts"].iloc[k - 1], df["ts"].iloc[k]],
                    [y.iloc[k - 1], y.iloc[k]],
                    color=FLAG_COLOR,
                    lw=1.5,
                    ls="--",
                    alpha=0.9,
                    zorder=3,
                )
    for a, b, who in source_runs(df):
        run_mask = (df["ts"] >= a) & (df["ts"] <= b)
        ax.plot(
            df.loc[run_mask, "ts"],
            y.loc[run_mask],
            color=SOURCE_COLORS.get(who, DEFAULT_COLOR),
            lw=2.0,
        )
    if flag_names:
        title = f"{title}  [QUALITY: {','.join(flag_names)} n={suspicious}]"
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)
    if len(df):
        multi = int((df.get("source_count", 1) > 1).sum()) if "source_count" in df.columns else 0
        ax.text(
            0.02,
            0.98,
            f"days={len(df)} sources={df['source'].nunique()} multi={multi} suspicious={suspicious}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )


def sample_pool(pg: PgClient, asset_class: str, *, multi_source: bool) -> pd.DataFrame:
    cols = "m.series_id, m.canonical_symbol, m.first_seen, m.last_seen, m.extras"
    if asset_class == "equity" and multi_source:
        rows = pg.fetchall(
            f"""
            SELECT {cols}
            FROM series_meta m
            WHERE m.asset_class = 'equity'
              AND m.first_seen IS NOT NULL AND m.last_seen IS NOT NULL
              AND EXISTS (SELECT 1 FROM symbol_aliases a WHERE a.series_id = m.series_id AND a.source = 'jakewright')
              AND EXISTS (SELECT 1 FROM symbol_aliases a WHERE a.series_id = m.series_id AND a.source = 'jacksoncrow')
              AND m.extras->>'eod_filled_through' IS NOT NULL
              AND (m.extras->>'eod_filled_through')::date
                  > COALESCE((m.extras->>'primary_last_seen')::date, m.first_seen)
            """
        )
    else:
        rows = pg.fetchall(
            f"""
            SELECT {cols}
            FROM series_meta m
            WHERE m.asset_class = %s AND m.first_seen IS NOT NULL AND m.last_seen IS NOT NULL
            """,
            (asset_class,),
        )
    return pd.DataFrame(rows)


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "close", "source", "source_count"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts")


def write_plot(df: pd.DataFrame, out_path: Path, series_id: str, symbol: str) -> dict:
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_stitched(ax, df, f"{series_id} stitched (close)")
    ax.set_xlabel("date")
    ax.set_ylabel("close")
    handles = [
        Line2D([0], [0], color=SOURCE_COLORS.get(s, DEFAULT_COLOR), lw=2, label=s)
        for s in sorted(df["source"].astype(str).unique())
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    suspicious, flags = suspicious_from_bars(df)
    return {"series_id": series_id, "suspicious_count": suspicious, "flags": flags}


def run_visuals(
    cfg,
    *,
    out_dir: Path,
    samples: int,
    total: int | None,
    seed: int,
    plot_years: int | None,
    features: list[str] | None,
) -> dict:
    from lexis_markets.registry import ensure_eod_aliases, seed_default_stitch
    from lexis_markets.serve.app import deploy_serve
    from lexis_markets.serve.dedupe import InFlightDedupe
    from lexis_markets.serve.handlers import MarketsService

    deploy_serve(cfg)
    pg = PgClient(cfg.postgres_url)
    ensure_eod_aliases(pg)
    seed_default_stitch(pg)
    svc = MarketsService(cfg, dedupe=InFlightDedupe())
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    flagged: list[str] = []

    picks: list[tuple[str, pd.Series]] = []
    if total is not None:
        frames: list[pd.DataFrame] = []
        for asset_class in PLOT_TYPES:
            pool = sample_pool(pg, asset_class, multi_source=asset_class == "equity")
            if pool.empty:
                continue
            pool = pool.copy()
            pool["asset_class"] = asset_class
            frames.append(pool)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            n = min(total, len(combined))
            for _, row in combined.sample(n=n, random_state=seed).iterrows():
                picks.append((str(row["asset_class"]), row))
    else:
        for asset_class in PLOT_TYPES:
            pool = sample_pool(pg, asset_class, multi_source=asset_class == "equity")
            if pool.empty:
                continue
            for _, row in (
                pool.sample(n=min(samples, len(pool)), random_state=seed).reset_index(drop=True).iterrows()
            ):
                picks.append((asset_class, row))

    for asset_class, row in picks:
        sid = row["series_id"]
        sym = str(row["canonical_symbol"]).replace("/", "_").replace("\\", "_")
        pstart, pend = plot_window(row, plot_years)
        safe_id = sid.replace(":", "_")
        class_dir = plot_dir / asset_class
        out_path = class_dir / f"{sym}_{safe_id}.png"
        try:
            body = svc.query_series(
                sid,
                pstart.isoformat(),
                pend.isoformat(),
                revision_mode="latest",
                features=features,
            )
            df = rows_to_df(body.get("rows") or [])
        except Exception as exc:
            skipped += 1
            print(f"plots: skip {sid} ({exc})", flush=True)
            continue
        if df.empty:
            skipped += 1
            print(f"plots: skip {sid} (empty)", flush=True)
            continue
        meta = write_plot(df, out_path, sid, sym)
        written += 1
        if meta["suspicious_count"]:
            flagged.append(f"{sid} suspicious={meta['suspicious_count']} flags={','.join(meta['flags'])}")
            print(f"plots: QUALITY FLAG {asset_class}/{sym} -> {out_path.name} ({meta})", flush=True)
        else:
            print(f"plots: {asset_class}/{sym} -> {out_path.name}", flush=True)

    summary_path = out_dir / "summary.txt"
    lines = [f"written={written} skipped={skipped} flagged={len(flagged)}"]
    lines.extend(flagged)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"written": written, "skipped": skipped, "flagged": len(flagged), "out": str(plot_dir)}


def main():
    p = argparse.ArgumentParser(description="Generate stitched-series plot samples")
    p.add_argument("--out", type=Path, default=Path("plot_samples"))
    p.add_argument("--samples", type=int, default=12, help="per asset_class when --total unset")
    p.add_argument("--total", type=int, default=None, help="sample this many series across all asset classes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-years", type=int, default=5, help="0 = full first_seen..last_seen range")
    p.add_argument(
        "--features",
        default="",
        help="comma-separated derived features for overlay data (e.g. sma_20,ema_50)",
    )
    args = p.parse_args()

    feats = [f.strip() for f in args.features.split(",") if f.strip()] or None
    plot_years = args.plot_years if args.plot_years > 0 else None

    def run(cfg):
        return run_visuals(
            cfg,
            out_dir=args.out,
            samples=args.samples,
            total=args.total,
            seed=args.seed,
            plot_years=plot_years,
            features=feats,
        )

    cli_main("visuals", run)


if __name__ == "__main__":
    main()
