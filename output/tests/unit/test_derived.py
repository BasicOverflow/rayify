"""Unit tests: SMA/EMA derived features."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from lexis_markets.domain.derived import add_ema, add_sma, apply_derived_features


def _bars(closes: list[float], *, series_id: str = "equity:AAPL") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": [series_id] * len(closes),
            "ts": [date(2020, 1, i + 1) for i in range(len(closes))],
            "close": closes,
            "open": closes,
            "high": closes,
            "low": closes,
            "volume": [100.0] * len(closes),
            "adj_close": closes,
            "source": ["yfinance"] * len(closes),
            "source_count": [1] * len(closes),
            "data_quality": ["ok"] * len(closes),
        }
    )


@pytest.mark.unit
def test_sma_3():
    out = add_sma(_bars([1.0, 2.0, 3.0, 4.0]), 3)
    assert pd.isna(out.loc[0, "sma_3"])
    assert pd.isna(out.loc[1, "sma_3"])
    assert out.loc[2, "sma_3"] == 2.0
    assert out.loc[3, "sma_3"] == 3.0


@pytest.mark.unit
def test_ema_3_first_value():
    out = add_ema(_bars([10.0, 20.0, 30.0]), 3)
    assert out.loc[0, "ema_3"] == 10.0
    assert out.loc[1, "ema_3"] == 15.0


@pytest.mark.unit
def test_apply_multiple_features():
    out = apply_derived_features(_bars([1.0, 2.0, 3.0, 4.0, 5.0]), ["sma_2", "ema_2"])
    assert "sma_2" in out.columns
    assert "ema_2" in out.columns
