"""Unit tests: even L1/series shards and in-memory stitch+score."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lexis_markets.eod.materialize import (
    attach_series_ids,
    decode_shard_parts,
    encode_shard_frames,
    even_shards,
    list_l1_parquet_keys,
    persist_l3_bars,
    shard_count,
    shard_index,
    stitch_and_score,
)


@pytest.mark.unit
def test_even_shards_sizes_differ_by_at_most_one():
    shards = even_shards(list(range(10)), 3)
    assert len(shards) == 3
    assert sorted(len(s) for s in shards) == [3, 3, 4]
    assert sorted(x for s in shards for x in s) == list(range(10))


@pytest.mark.unit
def test_even_shards_empty_and_single():
    assert even_shards([], 4) == [[], [], [], []]
    assert even_shards([1, 2, 3], 1) == [[1, 2, 3]]


@pytest.mark.unit
def test_shard_index_stable_and_covers_range():
    n = 8
    hits = {shard_index(f"equity:S{i}", n) for i in range(40)}
    assert hits <= set(range(n))
    assert len(hits) > 1
    assert shard_index("equity:AAPL", n) == shard_index("equity:AAPL", n)


@pytest.mark.unit
def test_shard_count_caps_at_items():
    assert shard_count(n_items=0) == 1
    assert shard_count(n_items=5, requested=20) == 5
    assert shard_count(n_items=100, requested=4) == 4


@pytest.mark.unit
def test_attach_series_ids_joins_and_respects_valid_window():
    obs = pd.DataFrame(
        {
            "source": ["jakewright", "jakewright", "yfinance"],
            "source_symbol": ["aaa", "AAA", "AAA"],
            "ts": [date(2020, 1, 1), date(2020, 6, 1), date(2020, 6, 2)],
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [1, 1, 1],
            "adj_close": [1.0, 2.0, 3.0],
        }
    )
    segs = pd.DataFrame(
        {
            "series_id": ["equity:AAA"],
            "source": ["jakewright"],
            "source_symbol": ["AAA"],
            "valid_from": [date(2020, 3, 1)],
            "valid_to": [None],
        }
    )
    out = attach_series_ids(obs, segs)
    assert list(out["ts"]) == [date(2020, 6, 1)]
    assert out.iloc[0]["series_id"] == "equity:AAA"


@pytest.mark.unit
def test_list_l1_parquet_keys_strips_prefix_and_skips_non_parquet():
    lake = MagicMock()
    lake.prefix = "test/"
    lake.list_keys.return_value = [
        "test/layer_1/year=2020/month=01/compacted-a.parquet",
        "test/layer_1/year=2020/month=01/notes.json",
        "test/layer_1/year=2020/month=02/part-x.parquet",
    ]
    keys = list_l1_parquet_keys(lake)
    assert keys == [
        "layer_1/year=2020/month=01/compacted-a.parquet",
        "layer_1/year=2020/month=02/part-x.parquet",
    ]


@pytest.mark.unit
def test_stitch_and_score_from_local_obs_no_cfg():
    obs = pd.DataFrame(
        {
            "source": ["jakewright", "jakewright"],
            "source_symbol": ["T", "T"],
            "ts": [date(2020, 1, 2), date(2020, 1, 3)],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.1],
            "volume": [100, 110],
            "adj_close": [10.2, 11.1],
        }
    )
    segs = pd.DataFrame(
        {
            "series_id": ["equity:T"],
            "source": ["jakewright"],
            "source_symbol": ["T"],
            "valid_from": [None],
            "valid_to": [None],
        }
    )
    bars, q, effects = stitch_and_score(
        "equity:T",
        date(2020, 1, 2),
        date(2020, 1, 3),
        segs,
        obs,
        calendar_id="nyse",
        status="ACTIVE",
        cfg=None,
    )
    assert len(bars) == 2
    assert "quality_score" in q
    assert effects == {}


@pytest.mark.unit
def test_persist_l3_bars_skips_write_when_policy_purged(monkeypatch):
    from lexis_markets.eod import materialize as mat

    wrote = {"n": 0}

    def _boom(*_a, **_k):
        wrote["n"] += 1
        raise AssertionError("should not write L3 after purge")

    monkeypatch.setattr(mat, "write_parquet_lake", _boom)
    n = persist_l3_bars(
        MagicMock(),
        "equity:T",
        date(2020, 1, 2),
        date(2020, 1, 3),
        pd.DataFrame(),
        {"purged_l3": True, "rebuilt": False},
    )
    assert n == 0
    assert wrote["n"] == 0


@pytest.mark.unit
def test_finish_wave_aligns_when_warm_off(monkeypatch):
    from lexis_markets.eod import ingest as eod_ingest

    aligned: list[str] = []
    warmed: list[str] = []

    monkeypatch.setattr(
        "lexis_markets.eod.l3_warm.align_symbols",
        lambda _cfg, symbols: aligned.extend(symbols) or {"symbols": len(symbols)},
    )
    monkeypatch.setattr(
        "lexis_markets.eod.l3_warm.warm_series_ids",
        lambda *_a, **_k: warmed.append("hit"),
    )
    monkeypatch.setattr(
        "lexis_markets.registry.patch_eod_registry",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "lexis_markets.registry.seed_default_stitch",
        lambda *_a, **_k: 0,
    )
    details = [{"source": "yfinance", "series_id": "equity:A", "symbol": "A", "rows": 10}]
    eod_ingest._finish_wave_l3(MagicMock(), MagicMock(), details, mode="off", align=True)
    assert aligned == ["A"]
    assert warmed == []


@pytest.mark.unit
def test_encode_shard_frames_is_parquet_bytes_not_frames():
    left = pd.DataFrame({"series_id": ["equity:A"], "close": [1.0]})
    right = pd.DataFrame({"series_id": ["equity:B"], "close": [2.0]})
    blobs = encode_shard_frames([[left], [right], []])
    assert len(blobs) == 3
    assert all(isinstance(b, (bytes, bytearray)) for b in blobs)
    assert blobs[2] == b""
    got = decode_shard_parts(blobs)
    assert sorted(got["series_id"].tolist()) == ["equity:A", "equity:B"]
    assert sorted(got["close"].tolist()) == [1.0, 2.0]


@pytest.mark.unit
def test_decode_shard_parts_empty_is_empty_frame():
    got = decode_shard_parts((b"", b""))
    assert got.empty
