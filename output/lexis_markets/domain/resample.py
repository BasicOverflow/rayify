"""Weekly/monthly OHLCV aggregation from daily canonical bars."""
from __future__ import annotations

import pandas as pd


def resample_bars(bars: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "daily" or bars.empty:
        return bars
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"])
    rule = {"weekly": "W-FRI", "monthly": "ME"}[granularity]
    bars = (
        bars.set_index("ts")
        .groupby("series_id")
        .resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "adj_close": "last",
                "source": "last",
                "source_count": "max",
                "data_quality": "last",
            }
        )
        .reset_index()
    )
    bars["ts"] = bars["ts"].dt.date
    return bars
