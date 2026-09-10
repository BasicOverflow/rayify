"""Unit tests: visuals helpers (no Serve / no PNG IO)."""
from __future__ import annotations

import pytest

from lexis_markets.cli.visuals import rows_to_df


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
