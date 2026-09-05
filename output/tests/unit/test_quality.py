"""Unit tests: gap/quality metrics and anomaly flags."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from lexis_markets.domain.quality import (
    apply_min_volume,
    gaps_from_bars,
    series_quality_window,
    suspicious_from_bars,
)


@pytest.mark.unit
def test_gaps_from_bars_weekday_calendar():
    # Mon–Fri week with Wednesday missing → 1 gap
    bars = pd.DataFrame(
        {
            "ts": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5)],
            "close": [1.0, 1.1, 1.2, 1.3],
            "source_count": [1, 1, 2, 1],
        }
    )
    gap, disagreement, score = gaps_from_bars(bars, "us_equity")
    assert gap == 1
    assert disagreement == 1
    assert 0.7 <= score <= 0.9


@pytest.mark.unit
def test_gaps_from_bars_empty():
    gap, disagreement, score = gaps_from_bars(pd.DataFrame(), "us_equity")
    assert (gap, disagreement, score) == (0, 0, 1.0)


@pytest.mark.unit
def test_series_quality_window_includes_status():
    bars = pd.DataFrame({"ts": [date(2024, 1, 2), date(2024, 1, 3)], "close": [1.0, 1.1]})
    out = series_quality_window(bars, "us_equity", {"status": "ACTIVE"})
    assert out["status"] == "ACTIVE"
    assert "gap_count" in out and "quality_score" in out
    assert out["suspicious_count"] == 0
    assert out["flags"] == []


@pytest.mark.unit
def test_suspicious_linear_ramp_flags_itmr_style():
    # Perfect arithmetic progression over ~60 sessions (dense fill).
    n = 60
    closes = np.linspace(0.5, 11.5, n)
    bars = pd.DataFrame(
        {
            "ts": [date(2017, 1, 3) + timedelta(days=i) for i in range(n)],
            "close": closes,
        }
    )
    count, flags = suspicious_from_bars(bars)
    assert count >= 50
    assert "linear_ramp" in flags
    out = series_quality_window(bars, "us_equity")
    assert out["suspicious_count"] == count
    assert "linear_ramp" in out["flags"]


@pytest.mark.unit
def test_suspicious_sparse_bridge_flags_itmr_real_shape():
    # Real ITMR: three prints years apart, then dense trading (plot draws diagonals).
    early = [
        (date(2016, 1, 27), 25.69),
        (date(2016, 9, 28), 0.315),
        (date(2019, 3, 8), 11.40),
    ]
    dense = [(date(2019, 3, 8) + timedelta(days=i), 11.0 + (i % 5) * 0.1) for i in range(1, 40)]
    bars = pd.DataFrame(early + dense, columns=["ts", "close"])
    count, flags = suspicious_from_bars(bars)
    assert "sparse_bridge" in flags
    assert count >= 21
    assert count > 200  # multi-year bridges dominate


@pytest.mark.unit
def test_suspicious_fred_native_skips_sparse_bridge():
    # Monthly FRED spacing must not look like ITMR bridges.
    bars = pd.DataFrame(
        {
            "ts": [date(2020, m, 1) for m in range(1, 13)],
            "close": [100.0 + i for i in range(12)],
        }
    )
    count, flags = suspicious_from_bars(bars, calendar_id="fred_native")
    assert count == 0
    assert flags == []


@pytest.mark.unit
def test_suspicious_noisy_market_path_clean():
    rng = np.random.default_rng(0)
    n = 60
    closes = 10 + np.cumsum(rng.normal(0, 0.3, size=n))
    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2) + timedelta(days=i) for i in range(n)],
            "close": closes,
        }
    )
    count, flags = suspicious_from_bars(bars, calendar_id="nyse")
    assert count == 0
    assert flags == []


@pytest.mark.unit
def test_suspicious_flat_close_run():
    n = 30
    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2) + timedelta(days=i) for i in range(n)],
            "close": [7.5] * n,
        }
    )
    count, flags = suspicious_from_bars(bars, calendar_id="nyse")
    assert count >= 21
    assert "flat_close" in flags
    assert "linear_ramp" in flags  # zero slope is also constant-slope


@pytest.mark.unit
def test_suspicious_ohlc_violation():
    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2), date(2020, 1, 3)],
            "open": [10.0, 10.0],
            "high": [9.0, 11.0],  # first bar: high < open/close
            "low": [10.5, 9.5],  # first bar: low > open
            "close": [10.0, 10.5],
        }
    )
    count, flags = suspicious_from_bars(bars, calendar_id="nyse")
    assert "ohlc_violation" in flags
    assert count >= 1


@pytest.mark.unit
def test_suspicious_non_positive_close():
    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)],
            "close": [10.0, 0.0, -1.0],
        }
    )
    count, flags = suspicious_from_bars(bars, calendar_id="nyse")
    assert "non_positive_close" in flags
    assert count >= 2


@pytest.mark.unit
def test_suspicious_duplicate_ts():
    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 3)],
            "close": [10.0, 10.1, 10.2],
        }
    )
    count, flags = suspicious_from_bars(bars, calendar_id="nyse")
    assert "duplicate_ts" in flags
    assert count >= 1


@pytest.mark.unit
def test_suspicious_extreme_return():
    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)],
            "close": [1.0, 10.0, 10.1],  # 900% jump
        }
    )
    count, flags = suspicious_from_bars(bars, calendar_id="nyse")
    assert "extreme_return" in flags
    assert count >= 1


@pytest.mark.unit
def test_apply_min_volume_filters():
    bars = pd.DataFrame({"ts": [1, 2, 3], "volume": [10, 100, 50], "close": [1, 2, 3]})
    filtered = apply_min_volume(bars, 60)
    assert list(filtered["volume"]) == [100]
