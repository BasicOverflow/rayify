"""Unit tests: JW/YF month align is independent per month and fans out unbounded."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from lexis_markets.domain.raw_bar import RAW_BAR_COLUMNS
from lexis_markets.eod.align import _align_month, _coverage_months, resolve_align_months
from lexis_markets.eod.l3_warm import align_symbols
from lexis_markets.jobs.scheduler import plan_task_resources
from lexis_markets.kaggle.compact import compacted_key
from lexis_markets.lake.local import LocalDirLake


def _bar(source: str, symbol: str, ts: date, close: float) -> dict:
    row = {c: None for c in RAW_BAR_COLUMNS}
    row.update(
        {
            "source": source,
            "source_symbol": symbol,
            "series_type": "equity",
            "ts": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "adj_close": close,
            "currency": "USD",
            "fetched_at": "2026-01-15T00:00:00+00:00",
        }
    )
    return row


class FakePg:
    def __init__(self, fetch_results=None):
        self.fetch_results = list(fetch_results or [])
        self.executemany_calls = 0
        self.execute_calls = 0

    def fetchall(self, sql, params=None):
        if not self.fetch_results:
            return []
        return self.fetch_results.pop(0)

    def executemany(self, sql, rows):
        self.executemany_calls += 1

    def execute(self, sql, params=None):
        self.execute_calls += 1


def _write_month(lake: LocalDirLake, year: int, month: int, rows: list[dict]) -> str:
    key = compacted_key(year, month, "test")
    lake.put_df_parquet(key, pd.DataFrame(rows))
    return key


@pytest.mark.unit
def test_align_month_scales_jw_to_yf_overlap(tmp_path):
    lake = LocalDirLake(tmp_path)
    day = date(2020, 3, 2)
    _write_month(
        lake,
        2020,
        3,
        [
            _bar("jakewright", "AAPL", day, 10.0),
            _bar("yfinance", "AAPL", day, 20.0),
        ],
    )
    pg = FakePg()
    patched = _align_month(
        lake, pg, 2020, 3, [{"jw_sym": "AAPL", "yf_sym": "AAPL"}]
    )
    assert patched == 1
    assert pg.execute_calls == 1
    frames = [
        lake.get_df_parquet(k)
        for k in lake.list_keys("layer_1/")
        if "/compacted-" in k
    ]
    jw = frames[0][frames[0]["source"] == "jakewright"].iloc[0]
    assert abs(float(jw["close"]) - 20.0) < 1e-9
    assert abs(float(jw["open"]) - 20.0) < 1e-9


@pytest.mark.unit
def test_align_month_skips_persist_when_no_overlap(tmp_path):
    lake = LocalDirLake(tmp_path)
    key = _write_month(
        lake,
        2020,
        4,
        [_bar("jakewright", "AAPL", date(2020, 4, 1), 10.0)],
    )
    pg = FakePg()
    patched = _align_month(
        lake, pg, 2020, 4, [{"jw_sym": "AAPL", "yf_sym": "AAPL"}]
    )
    assert patched == 0
    assert pg.execute_calls == 0
    assert lake.exists(key)


@pytest.mark.unit
def test_align_month_order_does_not_matter(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lake = LocalDirLake(tmp_path)
    _write_month(
        lake,
        2020,
        2,
        [
            _bar("jakewright", "MSFT", date(2020, 2, 3), 5.0),
            _bar("yfinance", "MSFT", date(2020, 2, 3), 10.0),
        ],
    )
    _write_month(
        lake,
        2020,
        1,
        [
            _bar("jakewright", "MSFT", date(2020, 1, 2), 4.0),
            _bar("yfinance", "MSFT", date(2020, 1, 2), 8.0),
        ],
    )
    pg_a, pg_b = FakePg(), FakePg()
    targets = [{"jw_sym": "MSFT", "yf_sym": "MSFT"}]
    start = threading.Barrier(2)

    def run(pg, year, month):
        start.wait()
        return _align_month(lake, pg, year, month, targets)

    with ThreadPoolExecutor(max_workers=2) as pool:
        later = pool.submit(run, pg_a, 2020, 2)
        earlier = pool.submit(run, pg_b, 2020, 1)
        assert later.result() == 1
        assert earlier.result() == 1
    by_month = {}
    for key in lake.list_keys("layer_1/"):
        if "/compacted-" not in key:
            continue
        df = lake.get_df_parquet(key)
        jw = df[df["source"] == "jakewright"].iloc[0]
        ts = pd.to_datetime(jw["ts"])
        by_month[int(ts.month)] = float(jw["close"])
    assert by_month[2] == 10.0
    assert by_month[1] == 8.0


@pytest.mark.unit
def test_coverage_months_uses_distinct_rows_then_bounds():
    pg = FakePg(
        fetch_results=[
            [{"year": 2021, "month": 12}, {"year": 2021, "month": 11}],
        ]
    )
    months = _coverage_months(pg, [{"jw_sym": "A", "yf_sym": "A"}])
    assert months == [(2021, 11), (2021, 12)]


@pytest.mark.unit
def test_resolve_align_months_empty_without_targets():
    pg = FakePg(fetch_results=[[]])
    targets, months = resolve_align_months(pg, ["ZZZ"])
    assert targets == []
    assert months == []


@pytest.mark.unit
def test_align_symbols_fans_one_task_per_month_unlimited(monkeypatch):
    captured: dict = {}
    targets = [{"jw_sym": "AAPL", "yf_sym": "AAPL"}]
    cfg_d = {"scratch": True}

    class FakeMonthTask:
        @staticmethod
        def remote(passed_cfg, year, month, passed_targets):
            captured.setdefault("calls", []).append(
                (passed_cfg, int(year), int(month), list(passed_targets))
            )
            return {"patched_rows": year + month, "year": year, "month": month}

    def fake_run_batches(label, items, submit_fn, shape, **_kw):
        captured["label"] = label
        captured["items"] = list(items)
        captured["shape"] = shape
        assert shape.batch_size == 1
        assert shape.max_in_flight == 0
        return [submit_fn([item]) for item in items]

    monkeypatch.setattr("lexis_markets.eod.l3_warm.task_align_yf_month", FakeMonthTask)
    monkeypatch.setattr("lexis_markets.eod.l3_warm.run_batches", fake_run_batches)
    monkeypatch.setattr(
        "lexis_markets.eod.l3_warm.cfg_d_with_scratch",
        lambda _cfg: cfg_d,
    )
    monkeypatch.setattr("lexis_markets.eod.l3_warm.PgClient", lambda _url: FakePg())
    monkeypatch.setattr(
        "lexis_markets.eod.l3_warm.resolve_align_months",
        lambda _pg, names: (targets, [(2020, 1), (2020, 2)]),
    )
    cfg = SimpleNamespace(postgres_url="postgresql://unused")
    out = align_symbols(cfg, ["aapl", "AAPL"])
    assert captured["label"] == "yf_align"
    assert captured["items"] == [(2020, 1), (2020, 2)]
    assert captured["calls"] == [
        (cfg_d, 2020, 1, targets),
        (cfg_d, 2020, 2, targets),
    ]
    assert plan_task_resources(batch_size=1, max_in_flight=0).max_in_flight == 0
    assert out["patched_rows"] == (2020 + 1) + (2020 + 2)
    assert out["batches"] == 2
    assert out["symbols"] == 1
