"""Session spine for weeksim: wipe, Ray, seed day0, shared clock/universe."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pytest

from lexis_markets.lake import LakeStore, PgClient, ensure_schema
from lexis_markets.kaggle.seed import run_seed
from lexis_markets.ray.gate_client import GateHandles
from lexis_markets.ray.markets_actors import bootstrap_gate_actors
from lexis_markets.ray.runtime import init_ray
from tests.support.config import for_tests, load_test_env
from tests.support.wipe_test_env import wipe_test_env
from tests.weeksim.asserts_goal import assert_pool_in_registry, assert_seed_markers
from tests.weeksim.clock import SimClock
from tests.weeksim.universe import DEFAULT_UNIVERSE, WeekUniverse


@dataclass
class WeekState:
    seed_done: bool = False
    nights_done: list[date] = field(default_factory=list)
    eod_targets: list[date] = field(default_factory=list)
    serve_ready: bool = False


def pytest_configure(config):
    try:
        load_test_env()
    except FileNotFoundError:
        # Collect-only / docs without .env.test; fixtures fail fast when a test runs.
        pass


@pytest.fixture(scope="session")
def universe() -> WeekUniverse:
    u = DEFAULT_UNIVERSE
    u.apply_env()
    return u


@pytest.fixture(scope="session")
def clock(universe) -> SimClock:
    universe.apply_env()
    c = SimClock.from_env()
    c.apply_eod_start_env()
    return c


@pytest.fixture(scope="session")
def week_cfg(universe, clock):
    universe.apply_env()
    clock.apply_eod_start_env()
    import os

    os.environ["MARKETS_TEST_EOD_FULL_PATH"] = "1"
    cfg = for_tests()
    assert cfg.test.eod_full_path, "MARKETS_TEST_EOD_FULL_PATH=1 required for weeksim"
    wipe_test_env(cfg)
    ensure_schema(PgClient(cfg.postgres_url))
    return cfg


@pytest.fixture(scope="session")
def week_state() -> WeekState:
    return WeekState()


@pytest.fixture(scope="session")
def ray_session(week_cfg):
    init_ray(week_cfg)
    bootstrap_gate_actors(week_cfg)
    yield
    try:
        import ray

        ray.shutdown()
    except Exception:
        pass


@pytest.fixture(scope="session")
def lake(week_cfg):
    return LakeStore(week_cfg)


@pytest.fixture(scope="session")
def pg(week_cfg):
    return PgClient(week_cfg.postgres_url)


@pytest.fixture(scope="session")
def gates(week_cfg, ray_session) -> GateHandles:
    from lexis_markets.ray.gate_client import resolve_gates

    return resolve_gates(week_cfg)


@pytest.fixture(scope="session")
def day0_seed(week_cfg, lake, pg, gates, clock, universe, week_state, ray_session):
    """Seed under simulated week_start morning; EOD fills through week_start-1."""
    from datetime import timedelta

    target = clock.week_start - timedelta(days=1)
    out = run_seed(
        week_cfg,
        deploy_serve_app=False,
        gates=gates,
        eod_target_date=target,
    )
    assert_seed_markers(lake)
    assert_pool_in_registry(pg, universe)
    week_state.seed_done = True
    week_state.eod_targets.append(target)
    return out


@pytest.fixture(scope="session")
def require_day0(day0_seed, week_state):
    assert week_state.seed_done
    return week_state


@pytest.fixture(scope="session")
def week_nights(require_day0, week_cfg, lake, pg, gates, clock, universe, week_state, ray_session):
    """Advance calendar night-by-night via cron enqueue + ``jobs.dispatch`` (prod spine)."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from lexis_markets.jobs.dispatch import WorkItem, dispatch_work_item
    from lexis_markets.jobs.queue import SupervisorState
    from lexis_markets.supervisor.cron import eod_window_open, maybe_schedule_eod
    from tests.weeksim.asserts_goal import (
        assert_eod_fill_through,
        assert_eod_marker,
        assert_fred_vintage_through,
        assert_weekend_eod_noop,
    )
    from tests.weeksim.spine import drain_queue

    db = Path(tempfile.mkdtemp()) / "week_cron.db"
    state = SupervisorState(str(db))
    with patch("lexis_markets.supervisor.cron.is_seed_complete", lambda cfg: True):
        for night in clock.nights():
            clock.day_index = (night - clock.week_start).days
            now = clock.eod_window_now()
            assert eod_window_open(week_cfg, now) is True
            assert maybe_schedule_eod(week_cfg, state, now) is True
            target = clock.eod_target()
            started = drain_queue(week_cfg, state, gates=gates)
            assert started >= 1, f"expected eod dispatch for {target}"

            # Idempotent second EOD same night (direct dispatch, same runner as queue).
            out = dispatch_work_item(
                week_cfg,
                WorkItem(
                    item_id=0,
                    job_type="eod",
                    payload={"target_date": target.isoformat()},
                ),
                gates=gates,
            )
            assert out.get("target_date") == target.isoformat()
            if clock.is_weekend(target):
                assert_weekend_eod_noop(out, target)
            else:
                assert_eod_marker(lake, target)
                assert_eod_fill_through(pg, universe, target)
                assert_fred_vintage_through(pg, universe, target)
            week_state.nights_done.append(night)
            if target not in week_state.eod_targets:
                week_state.eod_targets.append(target)
    assert len(week_state.nights_done) == clock.week_days
    return week_state
