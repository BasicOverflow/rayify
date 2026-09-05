"""Simulate Ray cluster offline while supervisor + NAS stay up; catch-up on reconnect."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from lexis_markets.jobs.dispatch import submit_loop
from lexis_markets.jobs.queue import SupervisorState
from lexis_markets.supervisor.cron import (
    catchup_through_date,
    maybe_schedule_eod_catchup,
    maybe_schedule_fred_catchup,
)


def _et_afternoon(d: date) -> datetime:
    # 16:00 ET ≈ 20:00 UTC (EST); window-closed for daily EOD, catch-up still allowed.
    return datetime(d.year, d.month, d.day, 20, 0, tzinfo=timezone.utc)


@pytest.mark.unit
def test_submit_loop_leaves_pending_when_ray_down(tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "ray_down.db"))
    state.enqueue_pending(
        "eod",
        {"target_date": "2025-03-10", "mode": "catchup", "run_key": "eod_catchup:2025-03-10"},
    )
    monkeypatch.setattr("lexis_markets.jobs.dispatch.ray_cluster_ready", lambda: False)
    cfg = MagicMock()
    gates = MagicMock()
    assert submit_loop(cfg, state, gates=gates) == 0
    pending = state.fetch_pending()
    assert len(pending) == 1
    assert pending[0]["status"] == "pending"


@pytest.mark.unit
def test_cluster_off_multi_day_refreshes_catchup_then_ready_to_drain(tmp_path, monkeypatch):
    """NAS markers lag across several calendar days while Ray submit is skipped.

    Supervisor cron keeps a single catch-up job and bumps its target as days pass.
    When Ray returns, that pending job is still there with the latest through date.
    """
    state = SupervisorState(str(tmp_path / "multi_day.db"))
    lake = MagicMock()
    # Stuck at Mar 7 while real calendar advances Mar 10 → Mar 13.
    lake.list_keys.return_value = ["ops/markers/eod_l1/2025-03-07.json"]
    lake.exists.side_effect = lambda key: key.endswith("2025-03-07.json")
    monkeypatch.setattr("lexis_markets.supervisor.cron.LakeStore", lambda _cfg: lake)

    cfg = MagicMock()
    day0 = date(2025, 3, 10)
    assert maybe_schedule_eod_catchup(cfg, state, _et_afternoon(day0)) is True
    pending = state.fetch_pending()
    assert len(pending) == 1
    assert pending[0]["payload"]["mode"] == "catchup"
    first_through = catchup_through_date(_et_afternoon(day0))
    assert pending[0]["payload"]["target_date"] == first_through.isoformat()

    monkeypatch.setattr("lexis_markets.jobs.dispatch.ray_cluster_ready", lambda: False)
    gates = MagicMock()
    assert submit_loop(cfg, state, gates=gates) == 0

    # Three more days offline: catch-up target must advance, still one pending job.
    for offset in (1, 2, 3):
        day = day0 + timedelta(days=offset)
        now = _et_afternoon(day)
        through = catchup_through_date(now)
        refreshed = maybe_schedule_eod_catchup(cfg, state, now)
        assert refreshed is True
        assert maybe_schedule_eod_catchup(cfg, state, now) is False  # already at through
        pending = state.fetch_pending()
        assert len(pending) == 1
        assert pending[0]["payload"]["target_date"] == through.isoformat()
        assert pending[0]["payload"]["run_key"] == f"eod_catchup:{through.isoformat()}"
        assert submit_loop(cfg, state, gates=gates) == 0

    final_through = catchup_through_date(_et_afternoon(day0 + timedelta(days=3)))
    assert final_through == date(2025, 3, 11)
    assert state.fetch_pending()[0]["payload"]["target_date"] == "2025-03-11"

    # Ray returns: submit_loop is allowed to see the pending catch-up (dispatch mocked).
    monkeypatch.setattr("lexis_markets.jobs.dispatch.ray_cluster_ready", lambda: True)
    monkeypatch.setattr("lexis_markets.jobs.dispatch.max_in_flight", lambda: 4)
    remote = MagicMock()
    remote.remote.return_value = "ref-1"
    monkeypatch.setattr("lexis_markets.jobs.dispatch._remote_dispatch", remote)
    with (
        patch("lexis_markets.jobs.dispatch.ray.wait", return_value=(["ref-1"], [])),
        patch(
            "lexis_markets.jobs.dispatch.ray.get",
            return_value={"target_date": "2025-03-11", "rows": 1},
        ),
    ):
        n = submit_loop(cfg, state, gates=gates)
    assert n == 1
    assert state.fetch_pending() == []
    last = state.get_last_run("eod_catchup")
    assert last is not None
    assert last["detail"].get("target_date") == "2025-03-11"


@pytest.mark.unit
def test_failed_catchup_requeues_when_markers_still_lag(tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "failed_catchup.db"))
    lake = MagicMock()
    lake.list_keys.return_value = []
    lake.exists.return_value = False
    monkeypatch.setattr("lexis_markets.supervisor.cron.LakeStore", lambda _cfg: lake)

    cfg = MagicMock()
    now = _et_afternoon(date(2025, 3, 12))
    assert maybe_schedule_eod_catchup(cfg, state, now) is True
    item = state.fetch_pending()[0]
    state.mark_running(item["id"])
    state.mark_failed(item["id"], "RayActorError: cluster gone")
    assert state.fetch_pending() == []

    assert maybe_schedule_eod_catchup(cfg, state, now) is True
    pending = state.fetch_pending()
    assert len(pending) == 1
    assert pending[0]["id"] == item["id"]
    assert pending[0]["payload"]["mode"] == "catchup"


@pytest.mark.unit
def test_fred_catchup_refreshes_while_pending(tmp_path):
    state = SupervisorState(str(tmp_path / "fred_off.db"))
    cfg = MagicMock()
    day0 = date(2025, 3, 10)
    assert maybe_schedule_fred_catchup(cfg, state, _et_afternoon(day0)) is True
    day2 = day0 + timedelta(days=2)
    assert maybe_schedule_fred_catchup(cfg, state, _et_afternoon(day2)) is True
    pending = state.fetch_pending()
    assert len(pending) == 1
    through = catchup_through_date(_et_afternoon(day2))
    assert pending[0]["payload"]["vintage_end"] == through.isoformat()
