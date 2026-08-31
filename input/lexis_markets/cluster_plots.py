from __future__ import annotations

import io
from datetime import date
from urllib.parse import quote

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import ray
from matplotlib.lines import Line2D

from lexis_markets.api import MarketsService
from lexis_markets.cluster import plan_task_resources, run_batches
from lexis_markets.config import MarketsConfig

SOURCE_COLORS = {
    "jakewright": "#2563eb",
    "jacksoncrow": "#dc2626",
    "fred": "#16a34a",
    "marketparquet": "#0d9488",
    "yfinance": "#ea580c",
}
DEFAULT_COLOR = "#6b7280"


def _plot_price(df: pd.DataFrame) -> pd.Series:
    return df["close"].astype(float)


def _source_runs(df: pd.DataFrame) -> list[tuple]:
    df = df.sort_values("ts")
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


def _render_png(df: pd.DataFrame, series_id: str) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4))
    y = _plot_price(df)
    for a, b, who in _source_runs(df):
        mask = (df["ts"] >= a) & (df["ts"] <= b)
        ax.plot(df.loc[mask, "ts"], y.loc[mask], color=SOURCE_COLORS.get(who, DEFAULT_COLOR), lw=2.0)
    ax.set_title(f"{series_id} L3 stitched (close)", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("date")
    ax.set_ylabel("close")
    handles = [
        Line2D([0], [0], color=SOURCE_COLORS.get(s, DEFAULT_COLOR), lw=2, label=s)
        for s in sorted(df["source"].astype(str).unique())
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _fetch_bars(cfg: MarketsConfig, series_id: str, start: str, end: str, *, via_api: bool) -> pd.DataFrame:
    if via_api:
        url = f"{cfg.ray_serve_url.rstrip('/')}/v1/series/{quote(series_id, safe='')}"
        resp = httpx.get(url, params={"start": start, "end": end}, timeout=600.0)
        resp.raise_for_status()
        rows = resp.json().get("rows") or []
        if not rows:
            return pd.DataFrame(columns=["ts", "close", "source", "source_count"])
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"])
        return df
    spec = {
        "series_ids": [series_id],
        "start": start,
        "end": end,
        "granularity": "daily",
        "revision_mode": "as_of",
    }
    return MarketsService(cfg).query_stitched_bars(spec)


@ray.remote
def task_render_plot_batch(
    cfg_d: dict, jobs: list[tuple[str, str, str, str]], via_api: bool = True
) -> list[dict]:
    cfg = MarketsConfig.from_dict(cfg_d)
    out = []
    for series_id, start, end, fname in jobs:
        bars = _fetch_bars(cfg, series_id, start, end, via_api=via_api)
        if bars.empty:
            out.append({"series_id": series_id, "fname": fname, "png": None, "rows": 0})
            continue
        df = bars.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        out.append(
            {
                "series_id": series_id,
                "fname": fname,
                "png": _render_png(df, series_id),
                "rows": len(df),
            }
        )
    return out


def run_cluster_plots(
    cfg: MarketsConfig,
    jobs: list[tuple[str, date, date, str]],
    *,
    via_api: bool = True,
) -> list[dict]:
    if not jobs:
        return []
    cfg_d = cfg.to_dict()
    payload = [(sid, str(start), str(end), fname) for sid, start, end, fname in jobs]
    shape = plan_task_resources(batch_size=4)
    batches = run_batches(
        "cluster_plots",
        payload,
        lambda batch: task_render_plot_batch.remote(cfg_d, batch, via_api),
        shape,
    )
    return [row for batch in batches for row in batch]
