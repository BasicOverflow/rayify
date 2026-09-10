"""Unit tests: EOD eligibility SQL fragment and L3 uncached span math."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from lexis_markets.jobs.clock import eod_target_date
from lexis_markets.registry import EOD_ELIGIBLE_WHERE
from lexis_markets.registry.meta import DateRange, uncached_ranges
from lexis_markets.supervisor.cron import eod_window_open


@pytest.mark.unit
def test_eod_eligible_where_is_universe_strong_sql():
    """Package export must be universe SQL (listing + yf_skip), not meta's weak copy."""
    from lexis_markets.registry import universe as uni

    sql = EOD_ELIGIBLE_WHERE.lower()
    assert EOD_ELIGIBLE_WHERE == uni.EOD_ELIGIBLE_WHERE
    assert "jacksoncrow" in sql
    assert "marketparquet" in sql
    assert "listing_exchange" in sql
    assert "yf_skip" in sql


@pytest.mark.unit
def test_eod_target_date_is_yesterday():
    assert eod_target_date(date(2025, 3, 11)) == date(2025, 3, 10)


@pytest.mark.unit
def test_uncached_ranges_full_miss():
    pg = MagicMock()
    pg.fetchall.return_value = []
    missing = uncached_ranges(pg, "equity:AAA", date(2024, 1, 1), date(2024, 1, 10))
    assert missing == [DateRange(date(2024, 1, 1), date(2024, 1, 10))]


@pytest.mark.unit
def test_uncached_ranges_partial_hit():
    pg = MagicMock()
    pg.fetchall.return_value = [
        {"span_start": date(2024, 1, 3), "span_end": date(2024, 1, 5)},
    ]
    missing = uncached_ranges(pg, "equity:AAA", date(2024, 1, 1), date(2024, 1, 10))
    assert missing == [
        DateRange(date(2024, 1, 1), date(2024, 1, 2)),
        DateRange(date(2024, 1, 6), date(2024, 1, 10)),
    ]


@pytest.mark.unit
def test_uncached_ranges_full_hit():
    pg = MagicMock()
    pg.fetchall.return_value = [
        {"span_start": date(2024, 1, 1), "span_end": date(2024, 1, 10)},
    ]
    missing = uncached_ranges(pg, "equity:AAA", date(2024, 1, 1), date(2024, 1, 10))
    assert missing == []


@pytest.mark.unit
def test_eod_window_open_after_close_plus_delay(monkeypatch):
    from datetime import datetime, timezone

    from lexis_markets.config import MarketsConfig

    cfg = MagicMock(spec=MarketsConfig)
    cfg.eod_delay_hours = 2
    # 19:00 ET = after 16:00 + 2h
    now = datetime(2025, 3, 11, 23, 0, tzinfo=timezone.utc)  # 19:00 ET
    assert eod_window_open(cfg, now) is True
    # 17:00 ET = before open
    early = datetime(2025, 3, 11, 21, 0, tzinfo=timezone.utc)  # 17:00 ET
    assert eod_window_open(cfg, early) is False


@pytest.mark.unit
def test_catchup_through_is_day_before_latest_session():
    from datetime import datetime, timezone

    from lexis_markets.supervisor.cron import catchup_through_date, latest_session_target

    now = datetime(2025, 3, 12, 18, 0, tzinfo=timezone.utc)  # 14:00 ET
    assert latest_session_target(now) == date(2025, 3, 11)
    assert catchup_through_date(now) == date(2025, 3, 10)


@pytest.mark.unit
def test_eod_catchup_schedules_without_window(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from lexis_markets.jobs.queue import SupervisorState
    from lexis_markets.supervisor.cron import maybe_schedule_eod, maybe_schedule_eod_catchup

    cfg = MagicMock()
    cfg.eod_delay_hours = 3
    state = SupervisorState(str(tmp_path / "catchup.db"))
    lake = MagicMock()
    lake.list_keys.return_value = []
    lake.exists.return_value = False
    monkeypatch.setattr("lexis_markets.supervisor.cron.LakeStore", lambda _cfg: lake)

    early = datetime(2025, 3, 12, 20, 0, tzinfo=timezone.utc)  # 16:00 ET, window closed
    assert eod_window_open(cfg, early) is False
    assert maybe_schedule_eod(cfg, state, early) is False
    assert maybe_schedule_eod_catchup(cfg, state, early) is True
    assert maybe_schedule_eod_catchup(cfg, state, early) is False
    pending = state.fetch_pending()
    assert len(pending) == 1
    assert pending[0]["payload"]["mode"] == "catchup"
    assert pending[0]["payload"]["target_date"] == "2025-03-10"


@pytest.mark.unit
def test_eod_catchup_skips_when_marker_current(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from lexis_markets.jobs.queue import SupervisorState
    from lexis_markets.supervisor.cron import maybe_schedule_eod_catchup

    cfg = MagicMock()
    state = SupervisorState(str(tmp_path / "caught_up.db"))
    lake = MagicMock()
    lake.list_keys.return_value = ["ops/markers/eod_l1/2025-03-11.json"]
    lake.exists.return_value = True
    monkeypatch.setattr("lexis_markets.supervisor.cron.LakeStore", lambda _cfg: lake)

    now = datetime(2025, 3, 12, 20, 0, tzinfo=timezone.utc)
    assert maybe_schedule_eod_catchup(cfg, state, now) is False


@pytest.mark.unit
def test_daily_eod_cron_only_after_window(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from lexis_markets.jobs.queue import SupervisorState
    from lexis_markets.supervisor.cron import maybe_schedule_eod

    cfg = MagicMock()
    cfg.eod_delay_hours = 3
    state = SupervisorState(str(tmp_path / "daily.db"))
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.is_seed_complete",
        lambda _cfg: True,
    )

    early = datetime(2025, 3, 12, 20, 0, tzinfo=timezone.utc)  # 16:00 ET
    late = datetime(2025, 3, 12, 23, 30, tzinfo=timezone.utc)  # 19:30 ET
    assert maybe_schedule_eod(cfg, state, early) is False
    assert maybe_schedule_eod(cfg, state, late) is True
    assert state.fetch_pending()[0]["payload"]["mode"] == "daily"
    assert state.fetch_pending()[0]["payload"]["target_date"] == "2025-03-11"
