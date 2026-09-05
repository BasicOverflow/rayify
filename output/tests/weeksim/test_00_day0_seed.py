"""Day0 seed stage asserts (migrated from integration test_00..04)."""
from __future__ import annotations

import pytest

from lexis_markets.kaggle.seed import is_seed_complete
from lexis_markets.kaggle.ingest import JC_MARKER, JW_MARKER
from tests.weeksim.asserts_goal import assert_pool_in_registry, assert_seed_markers


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(0)
def test_day0_seed_complete(require_day0, week_cfg, lake, pg, universe, day0_seed):
    assert is_seed_complete(week_cfg)
    assert_seed_markers(lake)
    assert lake.exists(JW_MARKER) and lake.exists(JC_MARKER)
    assert_pool_in_registry(pg, universe)
    assert day0_seed.get("rows") is not None or day0_seed.get("series") is not None


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(1)
def test_day0_seed_idempotent_markers(require_day0, week_cfg, lake, gates, clock, ray_session):
    """Re-running seed must skip JW/JC download when markers exist."""
    from lexis_markets.kaggle.seed import run_seed

    target = clock.week_start - __import__("datetime").timedelta(days=1)
    out = run_seed(
        week_cfg,
        reset=False,
        deploy_serve_app=False,
        gates=gates,
        eod_target_date=target,
    )
    assert lake.exists(JW_MARKER)
    assert lake.exists(JC_MARKER)
    assert out is not None
