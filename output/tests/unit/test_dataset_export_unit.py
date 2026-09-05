"""Unit tests: dataset export is one parquet; filters feed the export path."""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch
from uuid import UUID

import pandas as pd
import pytest
from fastapi import HTTPException

from lexis_markets.domain.canonical_bar import CANONICAL_BAR_COLUMNS
from lexis_markets.serve.export import build_dataset_parquet
from lexis_markets.serve.handlers import MarketsService
from tests.unit.api_fixtures import ERA_END, FilterPg, build_api_test_universe, effective_last


def _multi_series_bars(series_ids: list[str], *, start: str = "2020-01-02", n_days: int = 20) -> pd.DataFrame:
    start_d = date.fromisoformat(start)
    rows = []
    for sid in series_ids:
        for i in range(n_days):
            ts = date.fromordinal(start_d.toordinal() + i)
            if ts.weekday() >= 5:
                continue
            rows.append(
                {
                    "series_id": sid,
                    "ts": ts,
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 1000.0,
                    "adj_close": 1.5,
                    "source": "yfinance",
                    "source_count": 1,
                    "data_quality": "ok",
                }
            )
    return pd.DataFrame(rows)[CANONICAL_BAR_COLUMNS]


@pytest.mark.unit
def test_dataset_export_writes_single_part_parquet():
    """Dataset contract: one giant dataset.parquet (+ manifest), not sharded parts."""
    cfg = MagicMock()
    cfg.to_dict.return_value = {}
    job_id = "11111111-1111-1111-1111-111111111111"
    series_ids = ["equity:E000", "etf:F000", "macro:DGS10"]
    bars = _multi_series_bars(series_ids)

    pg = MagicMock()
    lake = MagicMock()
    lake.bucket = "lexis-markets"
    lake.uri.side_effect = lambda p: f"s3://lexis-markets/{p}"

    with (
        patch("lexis_markets.serve.export.PgClient", return_value=pg),
        patch("lexis_markets.serve.export.LakeStore", return_value=lake),
        patch("lexis_markets.serve.handlers.MarketsService") as svc_cls,
        patch("lexis_markets.serve.export.utcnow", return_value=date(2020, 12, 31)),
        patch("lexis_markets.serve.export.put_json") as put_json,
    ):
        svc_cls.return_value.query_stitched_bars.return_value = bars
        manifest = build_dataset_parquet(
            cfg,
            job_id,
            {
                "series_ids": series_ids,
                "start": "2020-01-01",
                "end": "2020-12-31",
                "revision_mode": "latest",
            },
        )

    assert manifest["files"] == ["dataset.parquet"]
    assert manifest["series_count"] == 3
    assert manifest["row_count"] == len(bars)
    assert manifest["status"] == "complete"
    put_keys = [c.args[0] for c in lake.put_df_parquet.call_args_list]
    assert put_keys == [f"layer_3/outputs/{job_id}/dataset.parquet"]
    assert put_json.call_args.args[1].endswith("manifest.json")
    assert len(put_keys) == 1  # single parquet only


@pytest.mark.unit
def test_dataset_export_empty_bars_still_one_file():
    cfg = MagicMock()
    job_id = "22222222-2222-2222-2222-222222222222"
    pg = MagicMock()
    lake = MagicMock()
    lake.bucket = "lexis-markets"
    lake.uri.side_effect = lambda p: f"s3://{p}"

    empty = pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    with (
        patch("lexis_markets.serve.export.PgClient", return_value=pg),
        patch("lexis_markets.serve.export.LakeStore", return_value=lake),
        patch("lexis_markets.serve.handlers.MarketsService") as svc_cls,
        patch("lexis_markets.serve.export.utcnow", return_value=date(2020, 12, 31)),
        patch("lexis_markets.serve.export.put_json"),
    ):
        svc_cls.return_value.query_stitched_bars.return_value = empty
        manifest = build_dataset_parquet(
            cfg,
            job_id,
            {"series_ids": ["equity:MISSING"], "start": "1990-01-01", "end": "1990-12-31"},
        )

    assert manifest["files"] == ["dataset.parquet"]
    assert manifest["series_count"] == 0
    assert manifest["row_count"] == 0
    assert lake.put_df_parquet.call_count == 1


@pytest.mark.unit
def test_create_dataset_sync_propagates_value_error_as_http_400():
    cfg = MagicMock()
    cfg.to_dict.return_value = {}
    svc = MarketsService.__new__(MarketsService)
    svc.cfg = cfg
    svc.pg = MagicMock()
    svc.lake = MagicMock()
    svc.lake.bucket = "lexis-markets"

    with patch("lexis_markets.serve.export.build_dataset_parquet", side_effect=ValueError("bad revision")):
        with pytest.raises(HTTPException) as ei:
            svc.create_dataset(
                {"series_ids": ["equity:E000"], "start": "2020-01-01", "end": "2020-06-01", "revision_mode": "nope"},
                sync=True,
            )
    assert ei.value.status_code == 400


@pytest.mark.unit
def test_create_dataset_async_queues_single_part_name():
    cfg = MagicMock()
    cfg.to_dict.return_value = {"ray_namespace": "lexis-markets"}
    svc = MarketsService.__new__(MarketsService)
    svc.cfg = cfg
    svc.pg = MagicMock()
    svc.lake = MagicMock()
    svc.lake.bucket = "lexis-markets"

    with patch("lexis_markets.serve.handlers.task_build_dataset") as task:
        task.remote = MagicMock(return_value=MagicMock())
        out = svc.create_dataset(
            {
                "series_ids": ["equity:E000", "etf:F000"],
                "start": "2015-01-01",
                "end": "2018-12-31",
                "statuses": ["ACTIVE"],
                "min_volume": 1000,
                "features": ["sma_20"],
                "revision_mode": "latest",
            },
            sync=False,
        )

    assert out["status"] == "queued"
    assert out["files"] == ["dataset.parquet"]
    assert UUID(out["job_id"])
    inserted = svc.pg.execute.call_args.args
    spec = json.loads(inserted[1][1])
    assert spec["min_volume"] == 1000
    assert spec["features"] == ["sma_20"]


@pytest.mark.unit
def test_dataset_spec_filter_combos_resolve_like_experiment():
    """Mirror the manual dataset experiment: mixed classes, past windows, filter gates."""
    universe = build_api_test_universe(100)
    pg = FilterPg(universe)

    # 2024–2025 style window (past-dated stand-in: 2019–2020)
    recent = resolve_like(
        pg,
        {
            "asset_classes": ["equity", "etf", "macro"],
            "statuses": ["ACTIVE"],
            "start": "2019-01-01",
            "end": "2020-12-31",
            "include_partial_coverage": True,
            "max_suspicious_count": 5000,
        },
    )
    assert len(recent) >= 10

    # 6-month style: stricter coverage through end
    tip = resolve_like(
        pg,
        {
            "statuses": ["ACTIVE"],
            "start": "2020-06-01",
            "end": ERA_END.isoformat(),
            "include_partial_coverage": False,
        },
    )
    assert tip
    for sid in tip:
        row = next(r for r in universe if r["series_id"] == sid)
        assert effective_last(row) >= ERA_END

    # Long history + low gap
    long = resolve_like(
        pg,
        {
            "asset_classes": ["equity"],
            "statuses": ["ACTIVE"],
            "start": "1990-01-01",
            "end": "2000-12-31",
            "max_gap_count": 30,
            "include_partial_coverage": True,
        },
    )
    assert long

    # Contradictory: DELISTED ids + ACTIVE-only
    delisted = next(r for r in universe if r["status"] == "DELISTED")
    assert (
        resolve_like(
            pg,
            {
                "series_ids": [delisted["series_id"]],
                "statuses": ["ACTIVE"],
                "start": "2010-01-01",
                "end": "2020-01-01",
            },
        )
        == []
    )


def resolve_like(pg: FilterPg, spec: dict) -> list[str]:
    from lexis_markets.registry.meta import resolve_series_ids

    return resolve_series_ids(pg, spec)
