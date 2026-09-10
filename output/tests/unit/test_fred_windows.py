"""Unit tests: vintage-date packing and adaptive ALFRED window sizing."""
from __future__ import annotations

from datetime import date

import pytest

from lexis_markets.fred.alfred import (
    AdaptiveWindowDays,
    pack_vintage_date_windows,
    plan_ingest_windows,
    vintage_windows,
)
from lexis_markets.fred.client import _bisect_mid


@pytest.mark.unit
def test_pack_vintage_dates_skips_empty_calendar_gaps():
    dates = [
        date(2000, 1, 15),
        date(2000, 6, 1),
        date(2010, 3, 1),  # decade gap
        date(2010, 9, 1),
    ]
    windows = pack_vintage_date_windows(dates, max_days=365)
    assert windows == [
        (date(2000, 1, 15), date(2000, 6, 1)),
        (date(2010, 3, 1), date(2010, 9, 1)),
    ]


@pytest.mark.unit
def test_pack_vs_calendar_fewer_windows_for_sparse_series():
    dates = [date(1995, 1, 1), date(2005, 1, 1), date(2015, 1, 1)]
    packed = pack_vintage_date_windows(dates, max_days=1825)
    calendar = vintage_windows(date(1995, 1, 1), date(2015, 1, 1), max_days=1825)
    assert len(packed) == 3
    assert len(calendar) > len(packed)


@pytest.mark.unit
def test_plan_ingest_windows_falls_back_to_calendar_without_vintages():
    windows = plan_ingest_windows(
        vintage_start=date(2020, 1, 1),
        vintage_end=date(2020, 2, 10),
        vintage_dates=[],
        max_days=31,
    )
    assert windows == vintage_windows(date(2020, 1, 1), date(2020, 2, 10), max_days=31)


@pytest.mark.unit
def test_plan_ingest_windows_packs_real_vintages():
    windows = plan_ingest_windows(
        vintage_start=date(2000, 1, 1),
        vintage_end=date(2010, 12, 31),
        vintage_dates=[date(2000, 1, 15), date(2000, 2, 15), date(2010, 1, 15)],
        max_days=400,
    )
    assert windows == [
        (date(2000, 1, 15), date(2000, 2, 15)),
        (date(2010, 1, 15), date(2010, 1, 15)),
    ]


@pytest.mark.unit
def test_adaptive_repack_after_shrink():
    """Simulate the live loop: shrink preferred days, then re-pack remaining vintages."""
    remaining = [
        date(2000, 1, 1),
        date(2000, 6, 1),
        date(2001, 1, 1),
        date(2001, 6, 1),
        date(2002, 1, 1),
    ]
    sizer = AdaptiveWindowDays(current=400, min_days=31, max_days=1825)
    first = plan_ingest_windows(
        vintage_start=remaining[0],
        vintage_end=remaining[-1],
        vintage_dates=remaining,
        max_days=sizer.preferred(),
    )[0]
    assert first == (date(2000, 1, 1), date(2001, 1, 1))
    sizer.observe(span_days=400, bisects=1)
    assert sizer.preferred() == 200
    remaining = [d for d in remaining if d > first[1]]
    next_windows = plan_ingest_windows(
        vintage_start=remaining[0],
        vintage_end=remaining[-1],
        vintage_dates=remaining,
        max_days=sizer.preferred(),
    )
    assert next_windows == [
        (date(2001, 6, 1), date(2001, 6, 1)),
        (date(2002, 1, 1), date(2002, 1, 1)),
    ]


@pytest.mark.unit
def test_adaptive_window_shrinks_on_bisect_grows_after_clean():
    sizer = AdaptiveWindowDays(current=1825, min_days=31, max_days=1825, grow_after=2)
    assert sizer.observe(span_days=1825, bisects=1) == 912
    assert sizer.observe(span_days=400, bisects=0) == 912
    grown = sizer.observe(span_days=400, bisects=0)
    assert grown > 912
    assert grown <= 1825


@pytest.mark.unit
def test_bisect_mid_prefers_middle_vintage_date():
    start = date(2020, 1, 1)
    end = date(2020, 12, 31)
    dates = [date(2020, 1, 15), date(2020, 6, 15), date(2020, 12, 1)]
    mid = _bisect_mid(start, end, dates)
    assert mid == date(2020, 6, 15)


@pytest.mark.unit
def test_fred_fetch_wrote_nothing_ignores_clamped_empty_jobs():
    from lexis_markets.fred.tasks import _fred_fetch_wrote_nothing

    clamped = [{"empty_after_clamp": True, "vintage_dates": []}]
    assert _fred_fetch_wrote_nothing(clamped, ok=0, rows=0) is False
    fetchable = [{"empty_after_clamp": False, "vintage_dates": ["2020-01-01"]}]
    assert _fred_fetch_wrote_nothing(fetchable, ok=0, rows=0) is True
    assert _fred_fetch_wrote_nothing(fetchable, ok=1, rows=10) is False


@pytest.mark.unit
def test_patch_macro_skips_zero_row_details():
    from lexis_markets.registry.meta import patch_macro_eod_registry

    class Pg:
        def __init__(self):
            self.rows = None

        def executemany(self, _sql, rows):
            self.rows = rows

    pg = Pg()
    out = patch_macro_eod_registry(
        pg,
        [
            {"series_id": "macro:DFF", "vintage_through": "2026-09-08", "last": "2026-09-08", "rows": 10},
            {"series_id": "macro:DJIA", "vintage_through": "2026-09-08", "rows": 0},
        ],
    )
    assert out == {"updated": 1}
    assert pg.rows == [(date(2026, 9, 8), "2026-09-08", "macro:DFF")]


@pytest.mark.unit
def test_seed_launches_fred_with_force():
    import inspect

    from lexis_markets.kaggle.seed import run_seed

    src = inspect.getsource(run_seed)
    assert "remote_fred_backfill.remote(cfg_d, compact=False, force=True)" in src
