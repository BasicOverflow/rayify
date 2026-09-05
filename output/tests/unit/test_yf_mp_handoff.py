"""YF/MP EOD handoff: yfinance must cover the full target window."""
from __future__ import annotations

from datetime import date

import pytest

from lexis_markets.eod.ingest import _yf_jobs, mp_cutoff


@pytest.mark.unit
def test_yf_jobs_covers_full_window_even_when_mp_partial():
    """Regression: truncating YF at cutoff-1 left holes when MP skipped a day."""
    today = date(2026, 9, 4)
    cutoff = mp_cutoff(today)
    assert cutoff == date(2026, 8, 28)

    targets = [
        {
            "series_id": "etf:SPY",
            "symbol": "SPY",
            "series_type": "etf",
            "start": date(2020, 4, 2),
            "end": date(2026, 9, 3),
            "primary_last": date(2020, 4, 1),
            "yf_sym": "SPY",
        }
    ]
    jobs = _yf_jobs(targets)
    assert len(jobs) == 1
    assert jobs[0]["start"] == date(2020, 4, 2)
    assert jobs[0]["end"] == date(2026, 9, 3)
    assert jobs[0]["end"] >= cutoff


@pytest.mark.unit
def test_yf_jobs_dedupes_series_windows():
    targets = [
        {
            "series_id": "equity:AAPL",
            "symbol": "AAPL",
            "series_type": "equity",
            "start": date(2024, 1, 1),
            "end": date(2024, 1, 31),
            "yf_sym": "AAPL",
        },
        {
            "series_id": "equity:AAPL",
            "symbol": "AAPL",
            "series_type": "equity",
            "start": date(2024, 1, 1),
            "end": date(2024, 1, 31),
            "yf_sym": "AAPL",
        },
    ]
    jobs = _yf_jobs(targets)
    assert len(jobs) == 1
