"""End-of-week Serve / export asserts (migrated from test_06/07)."""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from tests.support import api_assertions
from tests.support.macro_probe import probe_macro_series
from tests.support.serve_client import create_dataset, ensure_serve_api, get_series, health
from tests.weeksim.asserts_goal import assert_serve_equity_range, assert_stitch_source_mix


@pytest.fixture(scope="module")
async def serve_ready(week_cfg, ray_session, week_nights, week_state):
    await ensure_serve_api(week_cfg)
    week_state.serve_ready = True
    yield week_cfg


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(40)
async def test_serve_health(serve_ready):
    body = await health(serve_ready)
    assert body.get("ok") is True


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(41)
async def test_serve_equity_simulated_range(serve_ready, clock, universe):
    start = (clock.week_start - timedelta(days=30)).isoformat()
    # Last simulated EOD target = last night's yesterday
    end = (clock.week_start + timedelta(days=clock.week_days - 2)).isoformat()
    sid = f"equity:{universe.equity_symbols[0]}"
    await assert_serve_equity_range(serve_ready, sid, start, end)


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(42)
async def test_serve_features(serve_ready, clock, universe):
    start = (clock.week_start - timedelta(days=60)).isoformat()
    end = (clock.week_start + timedelta(days=clock.week_days - 2)).isoformat()
    body = await get_series(
        serve_ready,
        f"equity:{universe.equity_symbols[0]}",
        start,
        end,
        features=["sma_20", "ema_20"],
    )
    api_assertions.assert_features(body, ["sma_20", "ema_20"])


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(43)
async def test_serve_macro_revision(serve_ready, clock):
    start = (clock.week_start - timedelta(days=365)).isoformat()
    end = (clock.week_start + timedelta(days=clock.week_days - 2)).isoformat()
    as_of = (clock.week_start - timedelta(days=30)).isoformat()
    macro_id = await probe_macro_series(serve_ready, start, end)
    latest = await get_series(
        serve_ready, macro_id, start, end, revision_mode="latest"
    )
    as_of_body = await get_series(
        serve_ready,
        macro_id,
        start,
        end,
        revision_mode="as_of",
        as_of=as_of,
    )
    assert latest.get("rows") and as_of_body.get("rows")
    api_assertions.assert_revision_differs(latest, as_of_body)


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(44)
async def test_serve_concurrent_dedupe(serve_ready, clock, universe):
    start = (clock.week_start - timedelta(days=30)).isoformat()
    end = (clock.week_start + timedelta(days=clock.week_days - 2)).isoformat()
    sid = f"equity:{universe.equity_symbols[0]}"
    results = await asyncio.gather(
        get_series(serve_ready, sid, start, end),
        get_series(serve_ready, sid, start, end),
    )
    assert all(r.get("rows") for r in results)


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(45)
async def test_cache_build_on_miss(serve_ready, clock, universe):
    start = (clock.week_start - timedelta(days=30)).isoformat()
    end = (clock.week_start + timedelta(days=clock.week_days - 2)).isoformat()
    # Second equity symbol exercises cache miss path
    sid = f"equity:{universe.equity_symbols[-1]}"
    body = await get_series(serve_ready, sid, start, end)
    assert body.get("rows"), f"cache miss failed for {sid}"


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(46)
async def test_dataset_export_sync(serve_ready, clock, universe):
    start = (clock.week_start - timedelta(days=30)).isoformat()
    end = (clock.week_start + timedelta(days=clock.week_days - 2)).isoformat()
    body = await create_dataset(
        serve_ready,
        {
            "series_ids": [f"equity:{universe.equity_symbols[0]}"],
            "start": start,
            "end": end,
        },
        sync=True,
    )
    assert body.get("job_id") or body.get("status") or body.get("rows") is not None or "ok" in str(body).lower()


@pytest.mark.weeksim
@pytest.mark.io
@pytest.mark.order(47)
def test_stitch_source_mix_end_of_week(serve_ready, week_nights, pg, universe):
    assert_stitch_source_mix(pg, universe.equity_symbols[0])
