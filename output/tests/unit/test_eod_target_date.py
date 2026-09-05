"""Unit guard: equity ingest_eod must honor cron target_date."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from lexis_markets.eod import ingest as eod_ingest


@pytest.mark.unit
def test_ingest_eod_uses_explicit_target_date(monkeypatch):
    target = date(2025, 3, 10)
    captured: dict = {}

    monkeypatch.setattr(eod_ingest, "ensure_eod_aliases", lambda _pg: None)
    monkeypatch.setattr(
        eod_ingest,
        "resolve_eod_targets",
        lambda pg, **kw: captured.update(kw) or [],
    )
    monkeypatch.setattr(eod_ingest, "fetch_nasdaq_directory", lambda: MagicMock(symbols=set()))
    monkeypatch.setattr(eod_ingest, "run_entity_detect", lambda *a, **k: {"registered": 0})
    monkeypatch.setattr(eod_ingest, "run_eod_gap_scan", lambda *a, **k: {"rewound": 0})
    lake = MagicMock()
    lake.exists.return_value = False
    cfg = MagicMock()
    cfg.test.enabled = False

    out = eod_ingest.ingest_eod(cfg, lake, MagicMock(), yf_gate=MagicMock(), target_date=target)
    assert captured.get("target_date") == target
    assert out.get("target_date") == target.isoformat()


@pytest.mark.unit
def test_as_of_today_is_day_after_target():
    assert eod_ingest._as_of_today(date(2025, 3, 10)) == date(2025, 3, 11)
