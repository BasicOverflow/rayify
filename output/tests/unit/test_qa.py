"""Unit tests: hash-sample QA pool helpers."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from lexis_markets.cli.qa import _parse_classes, _pick_rows, list_pool
from lexis_markets.serve.client import hash_sample


@pytest.mark.unit
def test_hash_sample_stable_and_sized():
    rows = [{"series_id": f"equity:S{i}"} for i in range(100)]
    a = hash_sample(rows, 10, seed=42)
    b = hash_sample(rows, 10, seed=42)
    c = hash_sample(rows, 10, seed=7)
    assert [r["series_id"] for r in a] == [r["series_id"] for r in b]
    assert len(a) == 10
    assert [r["series_id"] for r in a] != [r["series_id"] for r in c]


@pytest.mark.unit
def test_parse_classes():
    assert _parse_classes(None) == ["equity", "etf"]
    assert _parse_classes("macro") == ["macro"]
    with pytest.raises(ValueError):
        _parse_classes("fx")


@pytest.mark.unit
def test_list_pool_flagged_sql():
    pg = MagicMock()
    pg.fetchall.return_value = []
    list_pool(pg, "equity", flagged=True)
    sql = pg.fetchall.call_args[0][0]
    assert "suspicious_count" in sql


@pytest.mark.unit
def test_pick_rows_explicit_series():
    pg = MagicMock()
    pg.fetchall.return_value = [
        {
            "series_id": "equity:AAPL",
            "canonical_symbol": "AAPL",
            "first_seen": date(2020, 1, 1),
            "last_seen": date(2024, 1, 1),
            "asset_class": "equity",
        }
    ]
    rows = _pick_rows(
        pg,
        series=["equity:AAPL"],
        asset_classes=["equity"],
        n=50,
        equity_n=None,
        etf_n=None,
        macro_n=None,
        seed=42,
        status="ACTIVE",
        flagged=False,
    )
    assert [r["series_id"] for r in rows] == ["equity:AAPL"]
