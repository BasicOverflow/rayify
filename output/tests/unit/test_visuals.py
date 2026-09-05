"""Unit tests: visuals helpers (no Serve / no PNG IO)."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lexis_markets.cli.visuals import rows_to_df, sample_pool


@pytest.mark.unit
def test_rows_to_df_empty():
    df = rows_to_df([])
    assert list(df.columns) == ["ts", "close", "source", "source_count"]
    assert df.empty


@pytest.mark.unit
def test_rows_to_df_sorts_by_ts():
    rows = [
        {"ts": "2024-01-03", "close": 3.0, "source": "yfinance", "source_count": 1},
        {"ts": "2024-01-01", "close": 1.0, "source": "jakewright", "source_count": 1},
    ]
    df = rows_to_df(rows)
    assert df.iloc[0]["close"] == 1.0
    assert df.iloc[-1]["close"] == 3.0


@pytest.mark.unit
def test_sample_pool_equity_multi_source_query():
    pg = MagicMock()
    pg.fetchall.return_value = [
        {
            "series_id": "equity:AAPL",
            "canonical_symbol": "AAPL",
            "first_seen": date(2020, 1, 1),
            "last_seen": date(2024, 1, 1),
            "extras": {},
        }
    ]
    df = sample_pool(pg, "equity", multi_source=True)
    assert len(df) == 1
    sql = pg.fetchall.call_args[0][0]
    assert "jakewright" in sql and "jacksoncrow" in sql
    assert "eod_filled_through" in sql


@pytest.mark.unit
def test_sample_pool_macro_simple():
    pg = MagicMock()
    pg.fetchall.return_value = []
    df = sample_pool(pg, "macro", multi_source=False)
    assert df.empty
    assert pg.fetchall.call_args[0][1] == ("macro",)
