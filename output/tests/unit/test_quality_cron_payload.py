"""Cron quality payload includes test-profile limit."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from lexis_markets.supervisor.cron import maybe_schedule_quality


@pytest.mark.unit
def test_quality_cron_passes_test_limit(tmp_path):
    from lexis_markets.jobs.queue import SupervisorState

    state = SupervisorState(str(tmp_path / "q.db"))
    cfg = MagicMock()
    cfg.eod_delay_hours = 0
    cfg.test.enabled = True
    cfg.test.quality_limit = 6
    now = datetime(2025, 3, 11, 20, 0, tzinfo=ZoneInfo("America/New_York"))
    assert maybe_schedule_quality(cfg, state, now) is True
    pending = state.fetch_pending()
    assert len(pending) == 1
    assert pending[0]["job_type"] == "quality"
    assert pending[0]["payload"].get("limit") == 6
