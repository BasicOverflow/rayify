"""Normalize vendor DataFrames into L1 ``RAW_BAR_COLUMNS`` layout."""
from __future__ import annotations

import json
import os

import pandas as pd

from lexis_markets.lake import utcnow


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
            "extras": [json.dumps({"ingest": source})] * n,
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
