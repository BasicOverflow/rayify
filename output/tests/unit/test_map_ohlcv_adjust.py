"""Unit tests: map_ohlcv scales OHLC when adj_close is present."""
from __future__ import annotations

import pandas as pd
import pytest

from lexis_markets.domain.quality import ohlc_violation_count
from lexis_markets.domain.sources.mappers import map_ohlcv, scale_ohlc_to_adj_close


@pytest.mark.unit
def test_scale_ohlc_to_adj_close_keeps_envelope():
    df = pd.DataFrame(
        {
            "open": [36.95],
            "high": [37.20],
            "low": [36.89],
            "close": [37.00],
            "adj_close": [35.96],
        }
    )
    out = scale_ohlc_to_adj_close(df)
    assert ohlc_violation_count(out) == 0
    assert abs(float(out.loc[0, "close"]) - 35.96) < 1e-9
    assert float(out.loc[0, "low"]) <= float(out.loc[0, "close"]) <= float(out.loc[0, "high"])


@pytest.mark.unit
def test_map_ohlcv_adjusts_when_adj_close_present():
    raw = pd.DataFrame(
        {
            "Symbol": ["PWS"],
            "Date": ["2026-03-06"],
            "Open": [32.40],
            "High": [32.57],
            "Low": [32.39],
            "Close": [32.50],
            "Adj Close": [32.17],
            "Volume": [1000.0],
        }
    )
    mapped = map_ohlcv(raw, source="yfinance", symbol_col="Symbol", date_col="Date")
    assert ohlc_violation_count(mapped) == 0
    assert abs(float(mapped.loc[0, "close"]) - 32.17) < 1e-6


@pytest.mark.unit
def test_map_ohlcv_can_skip_adjust():
    raw = pd.DataFrame(
        {
            "Symbol": ["X"],
            "Date": ["2026-03-06"],
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.5],
            "Adj Close": [5.0],
            "Volume": [1.0],
        }
    )
    mapped = map_ohlcv(
        raw, source="yfinance", symbol_col="Symbol", date_col="Date", adjust_to_adj_close=False
    )
    assert float(mapped.loc[0, "close"]) == 10.5
    assert float(mapped.loc[0, "adj_close"]) == 5.0
