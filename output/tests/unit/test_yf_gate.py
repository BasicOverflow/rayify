"""Unit tests: adaptive yfinance pace / chunk / slop knobs."""
from __future__ import annotations

import pytest

from lexis_markets.config import EOD_CHUNK_SIZE_DEFAULT, EOD_START_SLOP_DAYS_DEFAULT
from lexis_markets.domain.gates import YfinanceGate


@pytest.mark.unit
def test_yf_gate_product_ceiling():
    assert EOD_CHUNK_SIZE_DEFAULT == 250
    assert EOD_START_SLOP_DAYS_DEFAULT == 365
    gate = YfinanceGate(
        2.5,
        chunk_size=EOD_CHUNK_SIZE_DEFAULT,
        min_chunk_size=30,
        start_slop_days=EOD_START_SLOP_DAYS_DEFAULT,
        min_start_slop_days=30,
    )
    knobs = gate.batch_knobs()
    assert knobs["chunk_size"] == 250
    assert knobs["start_slop_days"] == 365


@pytest.mark.unit
def test_yf_gate_starts_aggressive():
    gate = YfinanceGate(
        2.5,
        chunk_size=150,
        min_chunk_size=30,
        start_slop_days=150,
        min_start_slop_days=30,
    )
    knobs = gate.batch_knobs()
    assert knobs["chunk_size"] == 150
    assert knobs["start_slop_days"] == 150
    assert knobs["interval"] == 2.5


@pytest.mark.unit
def test_yf_gate_rate_limit_shrinks_all_knobs():
    gate = YfinanceGate(
        2.5,
        max_interval=30.0,
        backoff_factor=2.0,
        chunk_size=150,
        min_chunk_size=30,
        start_slop_days=150,
        min_start_slop_days=30,
    )
    gate.record_rate_limited()
    knobs = gate.batch_knobs()
    assert knobs["interval"] == 5.0
    assert knobs["chunk_size"] == 75
    assert knobs["start_slop_days"] == 75


@pytest.mark.unit
def test_yf_gate_rate_limit_respects_floors():
    gate = YfinanceGate(
        2.5,
        chunk_size=40,
        min_chunk_size=30,
        start_slop_days=40,
        min_start_slop_days=30,
    )
    gate.record_rate_limited()
    gate.record_rate_limited()
    knobs = gate.batch_knobs()
    assert knobs["chunk_size"] == 30
    assert knobs["start_slop_days"] == 30


@pytest.mark.unit
def test_yf_gate_success_grows_chunk_and_slop():
    gate = YfinanceGate(
        2.5,
        recovery_step=0.1,
        chunk_size=150,
        min_chunk_size=30,
        start_slop_days=150,
        min_start_slop_days=30,
        grow_after=2,
    )
    gate.record_rate_limited()
    assert gate.chunk_size == 75
    gate.record_success()
    gate.record_success()
    assert gate.chunk_size > 75
    assert gate.start_slop_days > 75
    assert gate.interval <= 5.0


@pytest.mark.unit
def test_run_yf_remaining_keeps_sibling_date_ranges():
    """Same series_id with two ranges must not drop the second after the first wave."""
    from datetime import date

    from lexis_markets.eod.ingest import _batch_yf_jobs

    targets = [
        {
            "series_id": "equity:AAA",
            "yf_sym": "AAA",
            "symbol": "AAA",
            "series_type": "equity",
            "start": date(2020, 1, 1),
            "end": date(2020, 6, 30),
        },
        {
            "series_id": "equity:AAA",
            "yf_sym": "AAA",
            "symbol": "AAA",
            "series_type": "equity",
            "start": date(2024, 1, 1),
            "end": date(2024, 6, 30),
        },
        {
            "series_id": "equity:BBB",
            "yf_sym": "BBB",
            "symbol": "BBB",
            "series_type": "equity",
            "start": date(2024, 1, 1),
            "end": date(2024, 6, 30),
        },
    ]
    jobs = _batch_yf_jobs(targets, chunk_size=1, start_slop_days=30)
    assert len(jobs) >= 2
    wave = jobs[:1]
    done = {(t["series_id"], t["start"], t["end"]) for j in wave for t in j["targets"]}
    remaining = [t for t in targets if (t["series_id"], t["start"], t["end"]) not in done]
    assert len(remaining) == len(targets) - len(wave[0]["targets"])
    assert any(t["series_id"] == "equity:AAA" for t in remaining)


@pytest.mark.unit
def test_batch_yf_jobs_365d_slop_packs_spread_starts():
    from datetime import date

    from lexis_markets.eod.ingest import _batch_yf_jobs

    targets = [
        {
            "series_id": f"equity:S{i}",
            "yf_sym": f"S{i}",
            "symbol": f"S{i}",
            "series_type": "equity",
            "start": date(2020, 1, 1) if i % 2 == 0 else date(2020, 6, 1),
            "end": date(2024, 12, 31),
        }
        for i in range(8)
    ]
    tight = _batch_yf_jobs(targets, chunk_size=400, start_slop_days=30)
    wide = _batch_yf_jobs(targets, chunk_size=400, start_slop_days=365)
    assert len(wide) == 1
    assert len(wide[0]["targets"]) == 8
    assert len(tight) > 1
