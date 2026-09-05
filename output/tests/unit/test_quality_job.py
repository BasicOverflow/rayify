"""Unit tests: quality job payload / row limiting."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lexis_markets.jobs.dispatch import _quality_rows, run_quality_job


@pytest.mark.unit
def test_quality_rows_respects_limit():
    pg = MagicMock()
    pg.fetchall.return_value = [
        {"series_id": f"equity:S{i}", "calendar_id": "us_equity", "first_seen": None, "last_seen": None}
        for i in range(20)
    ]
    rows = _quality_rows(pg, {"limit": 5, "asset_classes": ["equity"]})
    assert len(rows) == 5


@pytest.mark.unit
def test_run_quality_job_passes_limit(monkeypatch):
    cfg = MagicMock()
    cfg.postgres_url = "postgresql://test"
    captured = {}

    def fake_rows(_pg, payload):
        captured["payload"] = payload
        return []

    monkeypatch.setattr("lexis_markets.jobs.dispatch._quality_rows", fake_rows)
    monkeypatch.setattr("lexis_markets.jobs.dispatch.PgClient", lambda _url: MagicMock())
    out = run_quality_job(cfg, {"limit": 5, "asset_classes": ["equity"]})
    assert out == {"series": 0, "gap_nonzero": 0, "gap_total": 0}
    assert captured["payload"]["limit"] == 5
