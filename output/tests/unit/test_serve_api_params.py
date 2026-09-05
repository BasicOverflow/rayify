"""Unit tests: Serve query params (revision, features, volume, granularity, clip)."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lexis_markets.domain.canonical_bar import CANONICAL_BAR_COLUMNS
from lexis_markets.domain.derived import apply_derived_features
from lexis_markets.domain.quality import apply_min_volume
from lexis_markets.domain.resample import resample_bars
from lexis_markets.serve.handlers import clip_range
from lexis_markets.serve.revision import parse_revision_mode, resolve_collapse_as_of, uses_l3_cache


def _bars(n: int = 40, *, start: str = "2020-01-02", volume: float = 100.0) -> pd.DataFrame:
    start_d = date.fromisoformat(start)
    rows = []
    for i in range(n):
        ts = start_d.fromordinal(start_d.toordinal() + i)
        if ts.weekday() >= 5:
            continue
        rows.append(
            {
                "series_id": "equity:E000",
                "ts": ts,
                "open": 10.0 + i * 0.1,
                "high": 11.0 + i * 0.1,
                "low": 9.0 + i * 0.1,
                "close": 10.5 + i * 0.1,
                "volume": volume if i % 3 else volume / 10,
                "adj_close": 10.5 + i * 0.1,
                "source": "yfinance",
                "source_count": 1,
                "data_quality": "ok",
            }
        )
    return pd.DataFrame(rows)[CANONICAL_BAR_COLUMNS]


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "as_of"),
        ("", "as_of"),
        ("as_of", "as_of"),
        ("LATEST", "latest"),
        ("latest", "latest"),
    ],
)
def test_parse_revision_mode_ok(raw, expected):
    assert parse_revision_mode(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["bogus", "both", "asof", "live"])
def test_parse_revision_mode_rejects_unknown(raw):
    with pytest.raises(ValueError, match="revision_mode"):
        parse_revision_mode(raw)


@pytest.mark.unit
def test_resolve_collapse_as_of_modes():
    end = date(2020, 12, 31)
    assert resolve_collapse_as_of(revision_mode="latest", request_end=end, as_of=None) is None
    assert resolve_collapse_as_of(revision_mode="as_of", request_end=end, as_of=None) == end
    assert (
        resolve_collapse_as_of(revision_mode="as_of", request_end=end, as_of=date(2020, 6, 1))
        == date(2020, 6, 1)
    )
    assert uses_l3_cache("latest") is True
    assert uses_l3_cache("as_of") is False


@pytest.mark.unit
def test_features_sma_ema_added():
    out = apply_derived_features(_bars(60), ["sma_5", "ema_5"])
    assert "sma_5" in out.columns and "ema_5" in out.columns
    assert out["sma_5"].notna().sum() >= 1


@pytest.mark.unit
@pytest.mark.parametrize("feat", ["rsi", "macd"])
def test_features_planned_not_implemented(feat):
    with pytest.raises(NotImplementedError):
        apply_derived_features(_bars(30), [feat])


@pytest.mark.unit
@pytest.mark.parametrize("feat", ["bogus", "sma", "ema_", "close", "vwap_20"])
def test_features_unknown_raises(feat):
    with pytest.raises(ValueError, match="unknown derived feature"):
        apply_derived_features(_bars(30), [feat])


@pytest.mark.unit
@pytest.mark.parametrize("feat", ["sma_1", "ema_1", "sma_501", "ema_999"])
def test_features_window_out_of_range(feat):
    with pytest.raises(ValueError, match="MA window"):
        apply_derived_features(_bars(30), [feat])


@pytest.mark.unit
def test_min_volume_filters_rows():
    bars = _bars(30, volume=100.0)
    filtered = apply_min_volume(bars, 50.0)
    assert len(filtered) < len(bars)
    assert (filtered["volume"] >= 50.0).all()


@pytest.mark.unit
def test_min_volume_none_is_noop():
    bars = _bars(20)
    assert len(apply_min_volume(bars, None)) == len(bars)


@pytest.mark.unit
def test_min_volume_too_high_yields_empty():
    bars = _bars(20, volume=10.0)
    assert apply_min_volume(bars, 1e9).empty


@pytest.mark.unit
@pytest.mark.parametrize("granularity", ["daily", "weekly", "monthly"])
def test_resample_granularities(granularity):
    bars = _bars(60)
    out = resample_bars(bars, granularity)
    assert not out.empty
    if granularity == "daily":
        assert len(out) == len(bars)
    else:
        assert len(out) < len(bars)


@pytest.mark.unit
def test_resample_invalid_granularity_raises():
    with pytest.raises(KeyError):
        resample_bars(_bars(10), "hourly")


@pytest.mark.unit
def test_clip_range_multi_series_unchanged():
    pg = MagicMock()
    start, end = date(2010, 1, 1), date(2020, 12, 31)
    assert clip_range(pg, ["equity:A", "equity:B"], start, end) == (start, end)
    pg.fetchone.assert_not_called()


@pytest.mark.unit
def test_clip_range_single_series_clips_to_meta():
    pg = MagicMock()
    pg.fetchone.return_value = {
        "first_seen": date(2015, 1, 1),
        "last_seen": date(2019, 6, 1),
        "extras": {"eod_filled_through": "2019-12-31"},
    }
    start, end = clip_range(pg, ["equity:E000"], date(2010, 1, 1), date(2020, 12, 31))
    assert start == date(2015, 1, 1)
    assert end == date(2019, 12, 31)


@pytest.mark.unit
def test_combo_features_then_min_volume():
    """Post-stitch pipeline order: features on full bars, then volume filter still works."""
    bars = apply_derived_features(_bars(40, volume=100.0), ["sma_5"])
    out = apply_min_volume(bars, 50.0)
    assert "sma_5" in out.columns
    assert (out["volume"] >= 50.0).all()


@pytest.mark.unit
def test_combo_weekly_then_min_volume():
    bars = resample_bars(_bars(60, volume=100.0), "weekly")
    out = apply_min_volume(bars, 1.0)
    assert not out.empty
