"""Mid-week chaos hooks on real weeksim SQLite / markers."""
from __future__ import annotations

import pytest

from lexis_markets.jobs.dispatch import WorkItem, dispatch_work_item
from lexis_markets.jobs.queue import SupervisorState
from lexis_markets.supervisor.cron import maybe_schedule_eod
from tests.weeksim.asserts_goal import assert_eod_marker


@pytest.mark.weeksim
@pytest.mark.chaos
@pytest.mark.order(30)
def test_failed_eod_blocks_same_day_reschedule(week_cfg, clock, tmp_path, monkeypatch):
    state = SupervisorState(str(tmp_path / "chaos_eod.db"))
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
    item = state.fetch_pending()[0]
    state.mark_running(item["id"])
    state.mark_failed(item["id"], "simulated death")
    assert maybe_schedule_eod(week_cfg, state, now) is False


@pytest.mark.weeksim
@pytest.mark.chaos
@pytest.mark.order(31)
def test_recover_interrupted_eod_requeues(tmp_path):
    state = SupervisorState(str(tmp_path / "chaos_recover.db"))
    item_id = state.enqueue_pending(
        "eod", {"target_date": "2025-03-10", "run_key": "eod:2025-03-11"}
    )
    state.mark_running(item_id)
    out = state.recover_interrupted_jobs(seed_complete=True)
    assert out["requeued"] == 1
    assert any(p["id"] == item_id for p in state.fetch_pending())


@pytest.mark.weeksim
@pytest.mark.chaos
@pytest.mark.order(32)
def test_recover_seed_complete_finishes_job(tmp_path):
    state = SupervisorState(str(tmp_path / "chaos_seed.db"))
    item_id = state.enqueue_pending("seed", {})
    state.mark_running(item_id)
    out = state.recover_interrupted_jobs(seed_complete=True)
    assert out["completed"] == 1
    assert not state.has_active_job("seed")


@pytest.mark.weeksim
@pytest.mark.chaos
@pytest.mark.order(33)
def test_dedupe_pending_seed(tmp_path):
    state = SupervisorState(str(tmp_path / "chaos_dedupe.db"))
    state.enqueue_pending("seed", {})
    state.enqueue_pending("seed", {})
    assert state.dedupe_pending_jobs("seed") == 1


@pytest.mark.weeksim
@pytest.mark.chaos
@pytest.mark.io
@pytest.mark.order(34)
def test_midweek_recover_preserves_prior_marker(
    week_nights, week_cfg, lake, gates, clock, week_state, ray_session, tmp_path
):
    prior = week_state.eod_targets[0]
    assert_eod_marker(lake, prior)

    state = SupervisorState(str(tmp_path / "chaos_mid.db"))
    item_id = state.enqueue_pending(
        "eod",
        {"target_date": prior.isoformat(), "run_key": f"eod:{prior.isoformat()}"},
    )
    state.mark_running(item_id)
    state.recover_interrupted_jobs(seed_complete=True)
    out = dispatch_work_item(
        week_cfg,
        WorkItem(
            item_id=item_id,
            job_type="eod",
            payload={"target_date": prior.isoformat()},
        ),
        gates=gates,
    )
    assert out.get("target_date") == prior.isoformat()
    assert_eod_marker(lake, prior)


@pytest.mark.weeksim
@pytest.mark.chaos
@pytest.mark.order(35)
def test_wipe_guard_refuses_prod_namespace(monkeypatch):
    from lexis_markets.config import MarketsConfig
    from tests.support.config import for_tests, load_test_env
    from tests.support.wipe_test_env import assert_test_safe

    load_test_env()
    for_tests()
    monkeypatch.setenv("RAY_NAMESPACE", "lexis-markets")
    monkeypatch.setenv("MARKETS_LAKE_PREFIX", "test/")
    bad = MarketsConfig.from_env()
    with pytest.raises(RuntimeError, match="production"):
        assert_test_safe(bad)


@pytest.mark.weeksim
@pytest.mark.chaos
@pytest.mark.order(36)
def test_gate_recreate_midweek(week_cfg, week_nights, ray_session):
    """Optional mid-week gate kill: recreate handles without breaking later work."""
    from lexis_markets.ray.markets_actors import bootstrap_gate_actors, reset_gate_actors
    from lexis_markets.ray.gate_client import resolve_gates

    reset_gate_actors(week_cfg)
    bootstrap_gate_actors(week_cfg)
    gates = resolve_gates(week_cfg)
    assert gates.yf_gate is not None
    assert gates.fred_gate is not None


@pytest.mark.weeksim
@pytest.mark.chaos
@pytest.mark.order(37)
def test_cluster_off_keeps_catchup_pending_until_ray_returns(
    week_cfg, week_nights, lake, clock, tmp_path, monkeypatch
):
    """Simulate Ray off for several days: cron refreshes catch-up; submit does not burn jobs."""
    from lexis_markets.jobs.dispatch import submit_loop
    from lexis_markets.jobs.queue import SupervisorState
    from lexis_markets.ray.gate_client import resolve_gates
    from lexis_markets.supervisor.cron import (
        catchup_through_date,
        maybe_schedule_eod_catchup,
    )
    from tests.weeksim.asserts_goal import assert_eod_marker

    # Markers exist through last night; force "behind" by asking catch-up for a later through.
    prior = week_nights.eod_targets[-1]
    assert_eod_marker(lake, prior)

    state = SupervisorState(str(tmp_path / "cluster_off.db"))
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.is_seed_complete",
        lambda cfg: True,
    )
    monkeypatch.setattr(
        "lexis_markets.supervisor.cron.needs_eod_through",
        lambda _lake, through: through > prior,
    )

    # Advance simulated calendar 3 days while Ray submit is disabled.
    monkeypatch.setattr("lexis_markets.jobs.dispatch.ray_cluster_ready", lambda: False)
    gates = resolve_gates(week_cfg)
    base_day = clock.today
    for i in range(3):
        clock.day_index = (base_day - clock.week_start).days + i + 1
        now = clock.morning()
        through = catchup_through_date(now)
        maybe_schedule_eod_catchup(week_cfg, state, now)
        assert submit_loop(week_cfg, state, gates=gates) == 0
        pending = state.fetch_pending()
        assert len(pending) == 1
        assert pending[0]["payload"]["mode"] == "catchup"
        assert pending[0]["payload"]["target_date"] == through.isoformat()

    final_through = catchup_through_date(clock.morning())
    assert final_through > prior
    assert state.fetch_pending()[0]["payload"]["target_date"] == final_through.isoformat()
