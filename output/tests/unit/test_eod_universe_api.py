"""Unit tests: EOD snapshot, universe listing, dataset mode normalization."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException

from lexis_markets.domain.canonical_bar import CANONICAL_BAR_COLUMNS
from lexis_markets.registry.meta import list_series_meta
from lexis_markets.serve.handlers import (
    MarketsService,
    normalize_query_spec,
    parse_dataset_mode,
    resolve_eod_snapshot_day,
    snapshot_last_bars,
)
from tests.unit.api_fixtures import ERA_END, FilterPg, build_api_test_universe


def _bars_two_series() -> pd.DataFrame:
    rows = []
    for sid, closes in [
        ("equity:E000", [(date(2020, 12, 28), 10.0), (date(2020, 12, 29), 11.0), (date(2020, 12, 30), 12.0)]),
        ("etf:F000", [(date(2020, 12, 24), 20.0), (date(2020, 12, 28), 21.0)]),
    ]:
        for ts, close in closes:
            rows.append(
                {
                    "series_id": sid,
                    "ts": ts,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000.0,
                    "adj_close": close,
                    "source": "yfinance",
                    "source_count": 1,
                    "data_quality": "ok",
                }
            )
    return pd.DataFrame(rows)[CANONICAL_BAR_COLUMNS]


@pytest.mark.unit
def test_parse_dataset_mode():
    assert parse_dataset_mode(None) == "range_panel"
    assert parse_dataset_mode("range") == "range_panel"
    assert parse_dataset_mode("EOD_SNAPSHOT") == "eod_snapshot"
    assert parse_dataset_mode("wide_matrix") == "wide_matrix"
    with pytest.raises(ValueError, match="mode must be"):
        parse_dataset_mode("weekly")


@pytest.mark.unit
def test_normalize_range_requires_bounds():
    with pytest.raises(ValueError, match="start and end"):
        normalize_query_spec({"mode": "range"})


@pytest.mark.unit
def test_normalize_range_defaults_l3_latest():
    out = normalize_query_spec(
        {"mode": "range_panel", "start": "2020-01-01", "end": "2020-06-01"}
    )
    assert out["mode"] == "range_panel"
    assert out["revision_mode"] == "latest"


@pytest.mark.unit
def test_normalize_eod_snapshot_fills_window(monkeypatch):
    monkeypatch.setattr(
        "lexis_markets.serve.handlers.eod_target_date",
        lambda today=None: date(2020, 12, 30),
    )
    out = normalize_query_spec({"mode": "eod_snapshot", "asset_classes": ["equity"]})
    assert out["mode"] == "eod_snapshot"
    assert out["end"] == "2020-12-30"
    assert out["start"] == "2020-12-16"
    assert out["granularity"] == "daily"
    assert out["_snapshot_as_of"] == "2020-12-30"
    assert out["revision_mode"] == "latest"


@pytest.mark.unit
def test_normalize_eod_snapshot_respects_override_date():
    out = normalize_query_spec(
        {
            "mode": "eod_snapshot",
            "eod_date": "2020-06-15",
            "eod_lookback_days": 7,
            "revision_mode": "as_of",
        }
    )
    assert out["end"] == "2020-06-15"
    assert out["start"] == "2020-06-08"
    assert out["revision_mode"] == "as_of"


@pytest.mark.unit
def test_normalize_bad_lookback():
    with pytest.raises(ValueError, match="eod_lookback_days"):
        normalize_query_spec({"mode": "eod_snapshot", "eod_date": "2020-01-01", "eod_lookback_days": 0})


@pytest.mark.unit
def test_snapshot_last_bars_picks_on_or_before_as_of():
    bars = _bars_two_series()
    out = snapshot_last_bars(bars, as_of=date(2020, 12, 29))
    assert set(out["series_id"]) == {"equity:E000", "etf:F000"}
    e = out[out.series_id == "equity:E000"].iloc[0]
    f = out[out.series_id == "etf:F000"].iloc[0]
    assert e["ts"] == date(2020, 12, 29) and float(e["close"]) == 11.0
    assert f["ts"] == date(2020, 12, 28) and float(f["close"]) == 21.0


@pytest.mark.unit
def test_snapshot_last_bars_empty_when_all_after_as_of():
    bars = _bars_two_series()
    out = snapshot_last_bars(bars, as_of=date(2020, 1, 1))
    assert out.empty


@pytest.mark.unit
def test_resolve_eod_snapshot_day(monkeypatch):
    monkeypatch.setattr(
        "lexis_markets.serve.handlers.eod_target_date",
        lambda today=None: date(2020, 12, 30),
    )
    assert resolve_eod_snapshot_day(None) == date(2020, 12, 30)
    assert resolve_eod_snapshot_day("2020-06-01") == date(2020, 6, 1)


@pytest.mark.unit
def test_list_series_meta_filters():
    universe = build_api_test_universe(100)
    pg = FilterPg(universe)
    rows = list_series_meta(
        pg,
        {
            "asset_classes": ["equity"],
            "statuses": ["ACTIVE"],
            "max_gap_count": 25,
            "start": "2010-01-01",
            "end": ERA_END.isoformat(),
            "include_partial_coverage": True,
        },
    )
    assert rows
    assert all(r["asset_class"] == "equity" for r in rows)
    assert all(r["status"] == "ACTIVE" for r in rows)
    assert all("canonical_symbol" in r for r in rows)


@pytest.mark.unit
def test_list_universe_service_pagination():
    universe = build_api_test_universe(40)
    svc = MarketsService.__new__(MarketsService)
    svc.pg = FilterPg(universe)
    out = svc.list_universe(asset_classes=["macro"], statuses=["ACTIVE"], limit=5, offset=0)
    assert out["count"] == 5
    assert out["total"] >= 5
    assert out["series"][0]["series_id"].startswith("macro:")


@pytest.mark.unit
def test_list_universe_bad_status_http_400():
    svc = MarketsService.__new__(MarketsService)
    svc.pg = FilterPg(build_api_test_universe(20))
    with pytest.raises(HTTPException) as ei:
        svc.list_universe(statuses=["NOPE"])
    assert ei.value.status_code == 400


@pytest.mark.unit
def test_query_series_eod_returns_single_bar():
    svc = MarketsService.__new__(MarketsService)
    svc.pg = MagicMock()
    svc.pg.fetchone.return_value = {
        "series_id": "equity:E000",
        "canonical_symbol": "E000",
        "asset_class": "equity",
        "calendar_id": "nyse",
        "gap_count": 0,
        "disagreement_count": 0,
        "suspicious_count": 0,
        "quality_score": 1.0,
        "first_seen": date(2010, 1, 1),
        "last_seen": date(2020, 12, 30),
        "status": "ACTIVE",
        "extras": {"eod_filled_through": "2020-12-30"},
    }
    bars = _bars_two_series()
    bars = bars[bars.series_id == "equity:E000"]
    with (
        patch("lexis_markets.serve.handlers.resolve_series_ids", return_value=["equity:E000"]),
        patch.object(MarketsService, "query_stitched_bars", return_value=bars.iloc[[-1]]),
    ):
        out = svc.query_series_eod("equity:E000", eod_date="2020-12-30")
    assert out["mode"] == "eod_snapshot"
    assert out["eod_date"] == "2020-12-30"
    assert out["bar"]["series_id"] == "equity:E000"
    assert len(out["rows"]) == 1


@pytest.mark.unit
def test_query_series_eod_404_when_no_bar():
    svc = MarketsService.__new__(MarketsService)
    svc.pg = MagicMock()
    svc.pg.fetchone.return_value = {
        "series_id": "equity:E000",
        "status": "ACTIVE",
        "calendar_id": "nyse",
        "first_seen": date(2010, 1, 1),
        "last_seen": date(2020, 12, 30),
        "extras": {},
    }
    with (
        patch("lexis_markets.serve.handlers.resolve_series_ids", return_value=["equity:E000"]),
        patch.object(MarketsService, "query_stitched_bars", return_value=pd.DataFrame()),
    ):
        with pytest.raises(HTTPException) as ei:
            svc.query_series_eod("equity:E000", eod_date="2020-12-30")
    assert ei.value.status_code == 404


@pytest.mark.unit
def test_create_dataset_eod_snapshot_mode_normalizes_before_queue():
    cfg = MagicMock()
    cfg.to_dict.return_value = {}
    svc = MarketsService.__new__(MarketsService)
    svc.cfg = cfg
    svc.pg = MagicMock()
    svc.lake = MagicMock()
    svc.lake.bucket = "lexis-markets"

    with (
        patch("lexis_markets.serve.handlers.eod_target_date", return_value=date(2020, 12, 30)),
        patch("lexis_markets.serve.handlers.task_build_dataset") as task,
    ):
        task.remote = MagicMock()
        out = svc.create_dataset(
            {
                "mode": "eod_snapshot",
                "asset_classes": ["equity", "etf", "macro"],
                "statuses": ["ACTIVE"],
            },
            sync=False,
        )
    assert out["mode"] == "eod_snapshot"
    assert out["files"] == ["dataset.parquet"]
    stored = svc.pg.execute.call_args.args[1][1]
    import json

    spec = json.loads(stored)
    assert spec["mode"] == "eod_snapshot"
    assert spec["end"] == "2020-12-30"
    assert spec["_snapshot_as_of"] == "2020-12-30"


@pytest.mark.unit
def test_dataset_spec_pydantic_range_requires_bounds():
    from lexis_markets.serve.app import DatasetSpec
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DatasetSpec(mode="range", asset_classes=["equity"])
    ok = DatasetSpec(mode="eod_snapshot", asset_classes=["equity"])
    assert ok.mode == "eod_snapshot"
    ok2 = DatasetSpec(mode="range", start="2020-01-01", end="2020-06-01")
    assert ok2.start == "2020-01-01"
