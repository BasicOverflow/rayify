"""Cron gating: seed / EOD, no quality job."""
from __future__ import annotations

import pytest
from freezegun import freeze_time

from lexis_markets.jobs.queue import SupervisorState
from lexis_markets.supervisor.cron import (
    cron_tick,
    eod_window_open,
    maybe_schedule_eod,
    maybe_schedule_seed,
)


@pytest.mark.weeksim
@pytest.mark.order(20)
def test_cron_does_not_schedule_quality(week_cfg, clock, tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "no_quality_cron.db"))
    monkeypatch.setattr("lexis_markets.supervisor.cron.is_seed_complete", lambda cfg: True)
    monkeypatch.setattr("lexis_markets.supervisor.cron.pipeline_ready", lambda cfg, st: True)
    now = clock.eod_window_now()
    tick = cron_tick(week_cfg, state, now)
    assert "quality_scheduled" not in tick
    pending = state.fetch_pending()
    assert all(p["job_type"] != "quality" for p in pending)


@pytest.mark.weeksim
@pytest.mark.order(21)
def test_cron_window_and_seed_gate(week_cfg, clock, tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "cron_gate.db"))
    now = clock.eod_window_now()
    assert eod_window_open(week_cfg, now) is True

    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.is_seed_complete",
        lambda cfg: False,
    )
    tick = cron_tick(week_cfg, state, now)
    assert tick["eod_scheduled"] is False


@pytest.mark.weeksim
@pytest.mark.order(22)
def test_cron_schedules_when_ready(week_cfg, clock, tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "cron_ready.db"))
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.is_seed_complete",
        lambda cfg: True,
    )
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.pipeline_ready",
        lambda cfg, st: True,
    )
    now = clock.eod_window_now()
    assert maybe_schedule_eod(week_cfg, state, now) is True
    assert maybe_schedule_eod(week_cfg, state, now) is False


@pytest.mark.weeksim
@pytest.mark.order(23)
def test_seed_scheduled_when_incomplete(week_cfg, tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "cron_seed.db"))
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.is_seed_complete",
        lambda cfg: False,
    )
    with freeze_time("2025-03-11 12:00:00"):
        assert maybe_schedule_seed(week_cfg, state) is True
    assert state.has_active_job("seed")
