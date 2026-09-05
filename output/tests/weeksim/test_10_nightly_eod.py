"""Nightly EOD under simulated calendar (migrated from test_05 + cron)."""
from __future__ import annotations

from datetime import timedelta

import pytest

from lexis_markets.eod.ingest import fetch_mp_daily
from lexis_markets.supervisor.cron import eod_window_open, maybe_schedule_eod
from lexis_markets.jobs.queue import SupervisorState
from tests.weeksim.asserts_goal import list_eod_markers


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(10)
def test_marketparquet_probe_in_window(require_day0, lake, clock, ray_session):
    probe = clock.week_start - timedelta(days=3)
    df = fetch_mp_daily(probe, lake)
    assert df is not None


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(11)
def test_nightly_spine_completed(week_nights, lake, clock):
    assert len(week_nights.nights_done) == clock.week_days
    markers = list_eod_markers(lake)
    assert markers
    # Weekday EOD targets must leave markers; weekend targets may be empty no-ops.
    for night in clock.nights():
        target = night - timedelta(days=1)
        if clock.is_weekend(target):
            continue
        assert any(target.isoformat() in m for m in markers), f"missing marker for {target}"


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(12)
def test_weekend_targets_present_in_week(week_nights, clock):
    weekend_targets = [
        night - timedelta(days=1)
        for night in clock.nights()
        if clock.is_weekend(night - timedelta(days=1))
    ]
    assert weekend_targets, "week window must include at least one weekend EOD target"
    assert all(t in week_nights.eod_targets for t in weekend_targets)


@pytest.mark.weeksim
@pytest.mark.order(13)
def test_same_day_cron_blocked_after_schedule(week_cfg, clock, tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "night_cron.db"))
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.is_seed_complete",
        lambda cfg: True,
    )
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.pipeline_ready",
        lambda cfg, st: True,
    )
    now = clock.eod_window_now()
    assert eod_window_open(week_cfg, now) is True
    assert maybe_schedule_eod(week_cfg, state, now) is True
    assert maybe_schedule_eod(week_cfg, state, now) is False
