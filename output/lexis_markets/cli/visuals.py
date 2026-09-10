"""Plot helpers for stitched-series QA PNGs."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from lexis_markets.registry.filters import effective_last_seen as registry_last_seen
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
