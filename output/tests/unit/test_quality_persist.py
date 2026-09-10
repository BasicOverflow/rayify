"""Unit tests: L3 quality persist (full replace vs tip merge)."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lexis_markets.serve.quality_persist import (
    _is_full_window,
    _merge_tip_tuple,
    persist_l3_quality,
    quality_update_tuple,
)


@pytest.mark.unit
def test_full_window_covers_registry():
    assert _is_full_window(date(2020, 1, 1), date(2024, 12, 31), date(2020, 1, 1), date(2024, 12, 31))
    assert not _is_full_window(date(2024, 11, 1), date(2024, 12, 31), date(2020, 1, 1), date(2024, 12, 31))


@pytest.mark.unit
def test_tip_merge_keeps_historical_suspicious():
    q = {
        "gap_count": 0,
        "disagreement_count": 0,
        "suspicious_count": 0,
        "quality_score": 1.0,
        "flag_counts": {"extreme_return": 0, "ohlc_violation": 1},
    }
    existing = {
        "gap_count": 12,
        "disagreement_count": 3,
        "suspicious_count": 875,
        "quality_score": 0.2,
        "flag_linear_ramp": 10,
        "flag_ohlc_violation": 0,
        "flag_extreme_return": 7,
    }
    tup = _merge_tip_tuple(q, "equity:TENX", existing)
    assert tup[0] == 12
    assert tup[2] == 875
    assert tup[3] == 0.2
    assert tup[7] == 1  # ohlc max(0, 1)
    assert tup[10] == 7  # extreme kept
    assert tup[-1] == "equity:TENX"


@pytest.mark.unit
def test_tip_merge_raises_new_flags():
    q = {
        "gap_count": 1,
        "disagreement_count": 0,
        "suspicious_count": 2,
        "quality_score": 0.5,
        "flag_counts": {"extreme_return": 2},
    }
    existing = {
        "gap_count": 0,
        "disagreement_count": 0,
        "suspicious_count": 0,
        "quality_score": 1.0,
        "flag_extreme_return": 0,
    }
    tup = _merge_tip_tuple(q, "equity:AAPL", existing)
    assert tup[2] == 2
    assert tup[3] == 0.5
    assert tup[10] == 2


@pytest.mark.unit
def test_persist_full_writes_policy_tuple(monkeypatch):
    cfg = MagicMock()
    meta = {
        "series_id": "equity:AAPL",
        "calendar_id": "nyse",
        "status": "ACTIVE",
        "first_seen": date(2020, 1, 1),
        "last_seen": date(2024, 12, 31),
        "suspicious_count": 0,
        "quality_score": 1.0,
    }
    q = {
        "gap_count": 4,
        "disagreement_count": 0,
        "suspicious_count": 4,
        "quality_score": 0.8,
        "flag_counts": {},
    }
    captured = {}

    monkeypatch.setattr("lexis_markets.serve.quality_persist.LakeStore", lambda _cfg: MagicMock())
    monkeypatch.setattr(
        "lexis_markets.serve.quality_persist.PgClient",
        lambda *_a, **_k: MagicMock(),
    )
    monkeypatch.setattr("lexis_markets.serve.quality_persist._meta_row", lambda _pg, _sid: meta)
    monkeypatch.setattr(
        "lexis_markets.serve.quality_persist.read_series_cache",
        lambda *_a, **_k: pd.DataFrame({"ts": [date(2020, 1, 2)]}),
    )
    monkeypatch.setattr(
        "lexis_markets.serve.quality_persist.apply_series_quality_policy",
        lambda *_a, **kw: (q, {"trimmed": False}),
    )

    def _apply(_pg, updates):
        captured["updates"] = updates

    monkeypatch.setattr("lexis_markets.serve.quality_persist.apply_quality_updates", _apply)
    out = persist_l3_quality(cfg, "equity:AAPL", date(2020, 1, 1), date(2024, 12, 31))
    assert out["full"] is True
    assert captured["updates"][0] == quality_update_tuple(q, "equity:AAPL")


@pytest.mark.unit
def test_persist_tip_does_not_zero_history(monkeypatch):
    cfg = MagicMock()
    meta = {
        "series_id": "equity:TENX",
        "calendar_id": "nyse",
        "status": "ACTIVE",
        "first_seen": date(2010, 1, 1),
        "last_seen": date(2024, 12, 31),
        "gap_count": 10,
        "disagreement_count": 0,
        "suspicious_count": 875,
        "quality_score": 0.1,
        "flag_linear_ramp": 800,
        "flag_sparse_bridge": 0,
        "flag_flat_close": 0,
        "flag_ohlc_violation": 0,
        "flag_non_positive_close": 0,
        "flag_duplicate_ts": 0,
        "flag_extreme_return": 0,
    }
    q = {
        "gap_count": 0,
        "disagreement_count": 0,
        "suspicious_count": 0,
        "quality_score": 1.0,
        "flag_counts": {},
    }
    captured = {}
    monkeypatch.setattr("lexis_markets.serve.quality_persist.LakeStore", lambda _cfg: MagicMock())
    monkeypatch.setattr(
        "lexis_markets.serve.quality_persist.PgClient",
        lambda *_a, **_k: MagicMock(),
    )
    monkeypatch.setattr("lexis_markets.serve.quality_persist._meta_row", lambda _pg, _sid: meta)
    monkeypatch.setattr(
        "lexis_markets.serve.quality_persist.read_series_cache",
        lambda *_a, **_k: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "lexis_markets.serve.quality_persist.apply_series_quality_policy",
        lambda *_a, **kw: (q, {}),
    )

    def _apply(_pg, updates):
        captured["updates"] = updates

    monkeypatch.setattr("lexis_markets.serve.quality_persist.apply_quality_updates", _apply)
    out = persist_l3_quality(cfg, "equity:TENX", date(2024, 11, 1), date(2024, 12, 31))
    assert out["full"] is False
    assert captured["updates"][0][2] == 875
