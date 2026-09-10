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
def test_linear_ramp_skips_split_adjusted_micro_history():
    # USB-style: 3000 days of tick steps at sub-cent adjusted prices.
    n = 80
    closes = np.linspace(0.0008, 0.004, n)
    bars = pd.DataFrame(
        {
            "ts": [date(1973, 5, 3) + timedelta(days=i) for i in range(n)],
            "close": closes,
        }
    )
    count, flags = suspicious_from_bars(bars)
    assert "linear_ramp" not in flags
    assert count == 0


@pytest.mark.unit
def test_linear_ramp_skips_quiet_drift():
    n = 80
    closes = np.linspace(17.48, 18.90, n)
    bars = pd.DataFrame(
        {
            "ts": [date(1987, 11, 11) + timedelta(days=i) for i in range(n)],
            "close": closes,
        }
    )
    count, flags = suspicious_from_bars(bars)
    assert "linear_ramp" not in flags


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
def test_suspicious_halt_gap_same_close_is_not_sparse_bridge():
    bars = pd.DataFrame(
        {
            "ts": [date(2016, 1, 4), date(2019, 3, 8), date(2019, 3, 11)],
            "close": [7.5, 7.5, 8.0],
        }
    )
    count, flags = suspicious_from_bars(bars, calendar_id="nyse")
    assert "sparse_bridge" not in flags
    assert count == 0


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
    assert "linear_ramp" not in flags  # identical closes are halt/flat, not a ramp


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


@pytest.mark.unit
def test_fred_native_daily_uses_busdays_not_calendar():
    # Ten weekdays across two Mon–Fri weeks → score ~1.0 (weekends not gaps).
    days = [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
        date(2024, 1, 9),
        date(2024, 1, 10),
        date(2024, 1, 11),
        date(2024, 1, 12),
    ]
    bars = pd.DataFrame({"ts": days, "close": [1.0 + i for i in range(10)]})
    gap, _dis, coverage = gaps_from_bars(bars, "fred_native")
    assert gap == 0
    assert coverage >= 0.99


@pytest.mark.unit
def test_fred_native_monthly_score_not_calendar_inflated():
    # 12 monthly prints across a year should score ~1.0 under fred_native.
    bars = pd.DataFrame(
        {
            "ts": [date(2020, m, 1) for m in range(1, 13)],
            "close": [100.0 + i for i in range(12)],
            "source_count": [1] * 12,
        }
    )
    gap, _dis, score = gaps_from_bars(bars, "fred_native")
    assert gap == 0
    assert score >= 0.95
    gap_nyse, _, score_nyse = gaps_from_bars(bars, "nyse")
    assert gap_nyse > 200
    assert score_nyse < 0.1


@pytest.mark.unit
def test_dense_observation_window_trims_sparse_prefix():
    from lexis_markets.domain.quality import dense_observation_window

    early = [date(2010, 1, 4), date(2012, 6, 1)]
    dense = [date(2019, 1, 2) + timedelta(days=i) for i in range(60)]
    ts = pd.Series(early + dense)
    win = dense_observation_window(ts, max_gap_days=21, min_bars=40)
    assert win is not None
    assert win[0] == dense[0]
    assert win[1] == dense[-1]


@pytest.mark.unit
def test_composite_quality_penalizes_suspicious():
    from lexis_markets.domain.quality import composite_quality_score

    assert composite_quality_score(1.0, 0, 100) == 1.0
    assert composite_quality_score(1.0, 100, 100) == 0.5
    assert composite_quality_score(0.8, 40, 100) == pytest.approx(0.64)


@pytest.mark.unit
def test_enforce_ohlc_bounds_fixes_violation():
    from lexis_markets.domain.ohlc_fix import enforce_ohlc_bounds
    from lexis_markets.domain.quality import ohlc_violation_count

    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2)],
            "open": [10.0],
            "high": [9.0],  # invalid: high < open/close
            "low": [11.0],  # invalid
            "close": [10.5],
        }
    )
    assert ohlc_violation_count(bars) == 1
    fixed, n = enforce_ohlc_bounds(bars)
    assert n == 1
    assert ohlc_violation_count(fixed) == 0
    assert fixed.iloc[0]["high"] >= fixed.iloc[0]["close"]
    assert fixed.iloc[0]["low"] <= fixed.iloc[0]["open"]


@pytest.mark.unit
def test_series_quality_window_flag_counts():
    n = 30
    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2) + timedelta(days=i) for i in range(n)],
            "close": [7.5] * n,
        }
    )
    out = series_quality_window(bars, "nyse")
    assert out["flag_counts"]["flat_close"] >= 21
    assert out["flag_counts"]["linear_ramp"] == 0
    assert out["suspicious_count"] > 0
    assert "flat_close" in out["flags"]


@pytest.mark.unit
def test_fred_native_skips_dense_window_trim():
    """Historical FRED gaps must not truncate last_seen / purge recent L3."""
    from types import SimpleNamespace

    from lexis_markets.domain.quality_policy import apply_series_quality_policy

    # Dense early run, then a 30-day gap, then recent bars (DFF-style).
    early = [date(2000, 1, 3) + timedelta(days=i) for i in range(60)]
    late = [date(2024, 1, 2) + timedelta(days=i) for i in range(40)]
    bars = pd.DataFrame(
        {
            "ts": early + late,
            "open": [1.0] * 100,
            "high": [1.1] * 100,
            "low": [0.9] * 100,
            "close": [1.0] * 100,
        }
    )
    cfg = SimpleNamespace(postgres_url="postgresql://unused")
    q, effects = apply_series_quality_policy(
        cfg,  # type: ignore[arg-type]
        series_id="macro:DFF",
        calendar_id="fred_native",
        status="ACTIVE",
        first_seen=early[0],
        last_seen=late[-1],
        bars=bars,
        apply=False,
        allow_window_trim=True,
    )
    assert effects["trimmed"] is False
    assert effects.get("new_last") is None
    assert "quality_score" in q


@pytest.mark.unit
def test_apply_series_quality_policy_score_only_no_cfg_io():
    """apply=False must not touch PG/MinIO; still returns fixed OHLC counts."""
    from types import SimpleNamespace

    from lexis_markets.domain.quality_policy import apply_series_quality_policy

    bars = pd.DataFrame(
        {
            "ts": [date(2020, 1, 2), date(2020, 1, 3)],
            "open": [10.0, 10.0],
            "high": [9.0, 11.0],
            "low": [11.0, 9.0],
            "close": [10.5, 10.2],
        }
    )
    cfg = SimpleNamespace(postgres_url="postgresql://unused")
    q, effects = apply_series_quality_policy(
        cfg,  # type: ignore[arg-type]
        series_id="equity:TEST",
        calendar_id="nyse",
        status="ACTIVE",
        first_seen=date(2020, 1, 2),
        last_seen=date(2020, 1, 3),
        bars=bars,
        apply=False,
    )
    assert effects["ohlc_fixed"] >= 1
    assert effects["rebuilt"] is False
    assert "quality_score" in q


@pytest.mark.unit
def test_apply_series_quality_policy_collapses_halt_flats():
    from types import SimpleNamespace

    from lexis_markets.domain.quality_policy import apply_series_quality_policy

    n = 30
    bars = pd.DataFrame(
        {
            "series_id": ["equity:T"] * n,
            "ts": [date(2020, 1, 2) + timedelta(days=i) for i in range(n)],
            "open": [7.5] * n,
            "high": [7.5] * n,
            "low": [7.5] * n,
            "close": [7.5] * n,
        }
    )
    cfg = SimpleNamespace(postgres_url="postgresql://unused")
    q, effects = apply_series_quality_policy(
        cfg,  # type: ignore[arg-type]
        series_id="equity:T",
        calendar_id="nyse",
        status="ACTIVE",
        first_seen=date(2020, 1, 2),
        last_seen=date(2020, 1, 2) + timedelta(days=n - 1),
        bars=bars,
        apply=False,
    )
    assert effects["halt_collapsed"] >= 21
    assert q["flag_counts"]["flat_close"] == 0
    assert q["suspicious_count"] == 0
