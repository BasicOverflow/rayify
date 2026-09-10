"""Cron no longer enqueues a quality job."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from lexis_markets.jobs.queue import SupervisorState
from lexis_markets.supervisor.cron import cron_tick


@pytest.mark.unit
def test_cron_does_not_enqueue_quality(tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "q.db"))
    cfg = MagicMock()
    cfg.eod_delay_hours = 0
    cfg.test.enabled = True
    now = datetime(2025, 3, 11, 20, 0, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr("lexis_markets.supervisor.cron.is_seed_complete", lambda _c: True)
    monkeypatch.setattr("lexis_markets.supervisor.cron.pipeline_ready", lambda _c, _s: True)
    monkeypatch.setattr("lexis_markets.supervisor.cron.maybe_schedule_eod_catchup", lambda *_a, **_k: False)
    monkeypatch.setattr("lexis_markets.supervisor.cron.maybe_schedule_fred_catchup", lambda *_a, **_k: False)
    monkeypatch.setattr("lexis_markets.supervisor.cron.maybe_schedule_seed", lambda *_a, **_k: False)
    monkeypatch.setattr("lexis_markets.supervisor.cron.maybe_prune_worker_disk", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.assess_markets_state",
        lambda _c: MagicMock(to_log_dict=lambda: {}, seed_complete=True),
    )
    tick = cron_tick(cfg, state, now)
    assert "quality_scheduled" not in tick
    pending = state.fetch_pending()
    assert all(p["job_type"] != "quality" for p in pending)
    assert any(p["job_type"] == "eod" for p in pending)
