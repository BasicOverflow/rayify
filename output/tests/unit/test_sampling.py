"""Unit tests: TestProfile sampling."""
from __future__ import annotations

import pytest

from lexis_markets.config import DevProfile, MarketsConfig, TestProfile as MarketsTestProfile


@pytest.mark.unit
def test_sample_fred_series_respects_limit_and_seed():
    dev = DevProfile(
        yf_backfill_limit=None,
        yf_backfill_seed=0,
        fred_backfill_limit=3,
        fred_backfill_seed=42,
    )
    ids = [f"S{i}" for i in range(20)]
    a = dev.sample_fred_series(ids)
    b = dev.sample_fred_series(ids)
    assert len(a) == 3
    assert a == b
    assert set(a).issubset(set(ids))


@pytest.mark.unit
def test_test_profile_yf_sampling_deterministic():
    tp = MarketsTestProfile(enabled=True, yf_limit=5, yf_seed=42)
    targets = [{"symbol": f"SYM{i}", "series_id": f"equity:SYM{i}"} for i in range(100)]
    a = tp.sample_targets(targets)
    b = tp.sample_targets(targets)
    assert len(a) == 5
    assert a == b


@pytest.mark.unit
def test_test_profile_fred_fixed_series():
    tp = MarketsTestProfile(enabled=True, fred_series=("DFF", "DGS10", "UNRATE"))
    assert tp.fred_series_ids(("A", "B", "C")) == ("DFF", "DGS10", "UNRATE")


@pytest.mark.unit
def test_test_profile_eod_sampling_uses_separate_limit():
    tp = MarketsTestProfile(enabled=True, yf_limit=5, eod_yf_limit=80, yf_seed=42)
    targets = [{"symbol": f"SYM{i}", "series_id": f"equity:SYM{i}"} for i in range(200)]
    a = tp.sample_eod_targets(targets, symbol_filter=None)
    b = tp.sample_eod_targets(targets, symbol_filter=None)
    assert len(a) == 80
    assert a == b


@pytest.mark.unit
def test_config_roundtrip_preserves_profiles():
    cfg = MarketsConfig.from_env()
    cfg.dev.fred_backfill_limit = 5
    cfg.test.enabled = True
    restored = MarketsConfig.from_dict(cfg.to_dict())
    assert restored.dev.fred_backfill_limit == 5
    assert restored.test.enabled is True


@pytest.mark.unit
def test_lake_key_prefix():
    from lexis_markets.config import MarketsConfig

    cfg = MarketsConfig.from_dict(
        {
            **MarketsConfig.from_env().to_dict(),
            "lake_prefix": "test/",
        }
    )
    assert cfg.lake_key("layer_1/foo") == "test/layer_1/foo"
