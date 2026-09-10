"""Multi-hop stitch calibration: prior scale must chain into the next fill."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from lexis_markets.serve.stitch import (
    calibrate_fill_gaps,
    collapse_halt_flats,
    drop_contradicted_jw_flats,
    pick_winners,
)


def _bar(series_id, ts, source, close, **extra):
    row = {
        "series_id": series_id,
        "ts": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000.0,
        "adj_close": close,
        "source": source,
        "source_symbol": "X",
        "series_type": "etf",
        "fetched_at": pd.Timestamp("2026-01-01", tz="UTC"),
        "data_quality": "ok",
    }
    row.update(extra)
    return row


@pytest.mark.unit
def test_yf_mp_tip_inherits_jc_yf_scale():
    """JC→YF scales YF up; MP tip overlaps YF at raw parity — must not leave an MP cliff."""
    sid = "etf:X"
    # JC only on day 0
    d0 = date(2020, 4, 1)
    # YF from day 1..10; raw YF starts at 100 while JC ended at 120 → k=1.2
    yf_days = [d0 + timedelta(days=i) for i in range(1, 11)]
    # MP on last 5 days, identical raw closes to YF (tip overlap)
    mp_days = yf_days[-5:]

    rows = [_bar(sid, d0, "jacksoncrow", 120.0)]
    for i, d in enumerate(yf_days):
        rows.append(_bar(sid, d, "yfinance", 100.0 + i))
    for i, d in enumerate(mp_days):
        # same raw as YF on those days
        yf_close = 100.0 + (len(yf_days) - 5 + i)
        rows.append(_bar(sid, d, "marketparquet", yf_close))

    merged = pd.DataFrame(rows)
    bars = pick_winners(merged)
    out = calibrate_fill_gaps(bars, merged)
    assert not out.empty
    out = out.sort_values("ts").reset_index(drop=True)
    sources = out["source"].tolist()
    assert sources[0] == "jacksoncrow"
    assert "yfinance" in sources
    assert sources[-1] == "marketparquet"

    # YF scaled by 1.2 onto JC
    yf = out[out["source"] == "yfinance"]
    assert abs(float(yf.iloc[0]["close"]) - 120.0) < 1e-6

    # MP tip must inherit that 1.2 scale (raw overlap ratio ~1 is not enough alone)
    mp = out[out["source"] == "marketparquet"]
    yf_last = float(yf.iloc[-1]["close"])
    mp_first = float(mp.iloc[0]["close"])
    assert abs(mp_first / yf_last - 1.0) < 0.02, (yf_last, mp_first)


@pytest.mark.unit
def test_near_2x_jc_yf_junction_kept():
    """Denomination / split-like JC→YF ratios just above 2.0 must not drop the fill."""
    sid = "etf:Z"
    d0 = date(2020, 4, 1)
    rows = [_bar(sid, d0, "jacksoncrow", 86.0)]
    for i in range(1, 6):
        rows.append(_bar(sid, d0 + timedelta(days=i), "yfinance", 40.0 + i))
    merged = pd.DataFrame(rows)
    out = calibrate_fill_gaps(pick_winners(merged), merged)
    assert "yfinance" in set(out["source"])
    assert abs(float(out[out["source"] == "yfinance"].iloc[0]["close"]) - 86.0) < 1e-6


@pytest.mark.unit
def test_4x_split_jc_fill_is_kept():
    """4:1 JC hole-fill is a split, not a stitch_break."""
    sid = "equity:S"
    d0 = date(1990, 1, 2)
    rows = [_bar(sid, d0, "jakewright", 80.0)]
    for i in range(1, 8):
        rows.append(_bar(sid, d0 + timedelta(days=i), "jacksoncrow", 20.0 + i * 0.1))
    out = calibrate_fill_gaps(pick_winners(pd.DataFrame(rows)), pd.DataFrame(rows))
    jc = out[out["source"] == "jacksoncrow"]
    assert not jc.empty
    assert abs(float(jc.iloc[0]["close"]) - 80.0) < 1e-6


@pytest.mark.unit
def test_insane_jc_fill_is_dropped():
    sid = "equity:S"
    d0 = date(1990, 1, 2)
    rows = [_bar(sid, d0, "jakewright", 80.0)]
    for i in range(1, 6):
        rows.append(_bar(sid, d0 + timedelta(days=i), "jacksoncrow", 0.02 + i * 0.001))
    out = calibrate_fill_gaps(pick_winners(pd.DataFrame(rows)), pd.DataFrame(rows))
    assert "jacksoncrow" not in set(out["source"])


@pytest.mark.unit
def test_single_hop_overlap_still_calibrates():
    sid = "etf:Y"
    days = [date(2026, 8, 20) + timedelta(days=i) for i in range(10)]
    rows = []
    for d in days[:6]:
        rows.append(_bar(sid, d, "yfinance", 200.0))
    for d in days[4:]:  # overlap 2+ days
        rows.append(_bar(sid, d, "marketparquet", 100.0))
    merged = pd.DataFrame(rows)
    bars = pick_winners(merged)
    out = calibrate_fill_gaps(bars, merged)
    mp = out[out["source"] == "marketparquet"]
    assert abs(float(mp.iloc[0]["close"]) - 200.0) < 1e-6


@pytest.mark.unit
def test_jw_flat_demoted_when_secondary_disagrees():
    sid = "equity:VL"
    days = [date(2020, 1, 2) + timedelta(days=i) for i in range(30)]
    rows = []
    for d in days:
        rows.append(_bar(sid, d, "jakewright", 10.0))
        rows.append(_bar(sid, d, "jacksoncrow", 12.0 + (d - days[0]).days * 0.1))
    merged = drop_contradicted_jw_flats(pd.DataFrame(rows))
    bars = pick_winners(merged)
    assert set(bars["source"]) == {"jacksoncrow"}
    assert abs(float(bars.sort_values("ts").iloc[0]["close"]) - 12.0) < 1e-6


@pytest.mark.unit
def test_jw_flat_kept_when_secondary_agrees():
    sid = "equity:AI"
    days = [date(2020, 1, 2) + timedelta(days=i) for i in range(30)]
    rows = []
    for d in days:
        rows.append(_bar(sid, d, "jakewright", 10.0))
        rows.append(_bar(sid, d, "jacksoncrow", 10.0))
    merged = drop_contradicted_jw_flats(pd.DataFrame(rows))
    winners = pick_winners(merged)
    assert set(winners["source"]) == {"jakewright"}
    out = collapse_halt_flats(winners)
    assert len(out) == 2
    assert list(out["close"]) == [10.0, 10.0]


@pytest.mark.unit
def test_jw_ramp_demoted_when_secondary_disagrees():
    sid = "equity:TX"
    days = [date(2020, 1, 2) + timedelta(days=i) for i in range(30)]
    rows = []
    for i, d in enumerate(days):
        rows.append(_bar(sid, d, "jakewright", 1.0 + i * 0.2))
        rows.append(_bar(sid, d, "yfinance", 10.0 + i * 0.05))
    merged = drop_contradicted_jw_flats(pd.DataFrame(rows))
    bars = pick_winners(merged)
    assert set(bars["source"]) == {"yfinance"}


@pytest.mark.unit
def test_collapse_halt_identical_close_even_with_ohlc_range():
    sid = "equity:AI"
    days = [date(2020, 1, 2) + timedelta(days=i) for i in range(30)]
    rows = [
        _bar(sid, d, "jakewright", 10.0, open=9.5, high=10.8, low=9.2)
        for d in days
    ]
    out = collapse_halt_flats(pick_winners(pd.DataFrame(rows)))
    assert len(out) == 2


@pytest.mark.unit
def test_collapse_halt_keeps_moving_tape():
    sid = "equity:M"
    days = [date(2020, 1, 2) + timedelta(days=i) for i in range(30)]
    rows = [_bar(sid, d, "jakewright", 10.0 + i * 0.2) for i, d in enumerate(days)]
    out = collapse_halt_flats(pick_winners(pd.DataFrame(rows)))
    assert len(out) == 30
