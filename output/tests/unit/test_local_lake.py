"""Unit tests: worker-local lake and MinIO flush key filter."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lexis_markets.lake.cluster import (
    ClusterLake,
    _parquet_bytes,
    _read_parquet_bytes,
    is_flush_key,
    owner_index,
    worker_actor_name,
)
from lexis_markets.lake.local import LocalDirLake


@pytest.mark.unit
def test_local_dir_lake_roundtrip(tmp_path: Path):
    lake = LocalDirLake(tmp_path)
    df = pd.DataFrame({"source": ["jakewright"], "close": [1.5]})
    key = "layer_1/year=2020/month=01/part-ab.parquet"
    lake.put_df_parquet(key, df)
    assert lake.exists(key)
    got = lake.get_df_parquet(key)
    assert list(got["close"]) == [1.5]
    assert key in lake.list_keys("layer_1/")
    prefixes = lake.list_common_prefixes("layer_1/")
    assert prefixes == ["layer_1/year=2020/"]
    lake.delete_keys([key])
    assert not lake.exists(key)


@pytest.mark.unit
def test_local_dir_lake_respects_prefix(tmp_path: Path):
    lake = LocalDirLake(tmp_path, prefix="test/")
    key = "layer_3/cache/equity__A.parquet"
    lake.put_bytes(key, b"abc")
    assert lake.exists(key)
    listed = lake.list_keys("layer_3/")
    assert listed == ["test/layer_3/cache/equity__A.parquet"]


@pytest.mark.unit
def test_owner_index_colocates_month_keys():
    a = "layer_1/year=2020/month=03/part-aaa.parquet"
    b = "layer_1/year=2020/month=03/compacted-bbb.parquet"
    c = "layer_1/year=2020/month=04/part-ccc.parquet"
    n = 8
    assert owner_index(a, n) == owner_index(b, n)
    assert owner_index(a, n) != owner_index(c, n) or n == 1
    assert owner_index(a, n, prefix="p/") == owner_index("p/" + a, n, prefix="p/")


@pytest.mark.unit
def test_flush_key_keeps_compacted_l1_and_l3_skips_parts_and_ops():
    assert is_flush_key("layer_1/year=2020/month=01/compacted-x.parquet")
    assert is_flush_key("layer_3/cache/equity__AAPL.parquet")
    assert not is_flush_key("layer_1/year=2020/month=01/part-ab.parquet")
    assert not is_flush_key("ops/staging/jakewright/parts/part-00000.parquet")
    assert not is_flush_key("ops/cache/jakewright.zip")
    assert not is_flush_key("ops/markers/seed_progress.json")
    assert is_flush_key("test/layer_1/year=2020/month=01/compacted-x.parquet", prefix="test/")
    assert not is_flush_key("test/layer_1/year=2020/month=01/part-z.parquet", prefix="test/")


@pytest.mark.unit
def test_parquet_bytes_roundtrip_avoids_dataframe_pickle():
    df = pd.DataFrame({"source": ["yfinance"], "close": [10.25], "volume": [3]})
    blob = _parquet_bytes(df)
    assert isinstance(blob, (bytes, bytearray))
    got = _read_parquet_bytes(blob, columns=["source", "close"])
    assert list(got["close"]) == [10.25]
    assert list(got.columns) == ["source", "close"]


@pytest.mark.unit
def test_worker_actor_name_fences_generation_from_legacy():
    assert worker_actor_name(0) == "lexis-seed-w-0"
    assert worker_actor_name(3, "a1b2c3d4") == "lexis-seed-a1b2c3d4-w-3"
    assert worker_actor_name(0, "a1b2c3d4") != worker_actor_name(0)


@pytest.mark.unit
def test_cluster_lake_put_df_sends_bytes_not_frame(monkeypatch):
    from lexis_markets.lake import cluster as cl

    sent = {}

    class Handle:
        def remote(self, key, data):
            sent["key"] = key
            sent["data"] = data
            return "ref"

    class Worker:
        put_df = Handle()

    lake = object.__new__(ClusterLake)
    lake.n = 1
    lake.prefix = ""
    lake._workers = [Worker()]
    monkeypatch.setattr(cl.ray, "get", lambda x: x)
    lake.put_df_parquet("k.parquet", pd.DataFrame({"close": [1.0]}))
    assert isinstance(sent["data"], (bytes, bytearray))
    assert list(_read_parquet_bytes(sent["data"])["close"]) == [1.0]


@pytest.mark.unit
def test_cluster_lake_get_df_decodes_bytes(monkeypatch):
    from lexis_markets.lake import cluster as cl

    blob = _parquet_bytes(pd.DataFrame({"close": [2.0]}))

    class Handle:
        def remote(self, *args):
            return blob

    class Worker:
        get_df = Handle()

    lake = object.__new__(ClusterLake)
    lake.n = 1
    lake.prefix = ""
    lake._workers = [Worker()]
    monkeypatch.setattr(cl.ray, "get", lambda x: x)
    got = lake.get_df_parquet("k.parquet")
    assert list(got["close"]) == [2.0]


@pytest.mark.unit
def test_open_lake_without_scratch_stays_minio_even_if_actors_exist(monkeypatch):
    from lexis_markets.lake import cluster as cl

    sentinel = object()
    monkeypatch.setattr(cl, "seed_lake_worker_count", lambda **_k: 8)
    monkeypatch.setattr(cl, "LakeStore", lambda _cfg: sentinel)
    assert cl.open_lake(object(), scratch=False) is sentinel


@pytest.mark.unit
def test_open_lake_scratch_raises_if_actors_down(monkeypatch):
    from lexis_markets.lake import cluster as cl

    monkeypatch.setattr(cl, "seed_lake_worker_count", lambda **_k: 0)
    cfg = type("Cfg", (), {"ray_namespace": "lexis-markets"})()
    with pytest.raises(RuntimeError, match="seed lake actors are down"):
        cl.open_lake(cfg, scratch=True)


@pytest.mark.unit
def test_lake_from_cfg_d_does_not_stick_scratch_flag(monkeypatch):
    from lexis_markets.lake import cluster as cl

    seen: dict = {}

    def fake_open(_cfg, *, scratch=None):
        seen["scratch"] = scratch
        return "lake"

    monkeypatch.setattr(cl, "open_lake", fake_open)
    monkeypatch.setattr(
        cl, "MarketsConfig", type("MC", (), {"from_dict": staticmethod(lambda d: object())})
    )
    cl._scratch.set(False)
    assert cl.lake_from_cfg_d({cl.SEED_SCRATCH_KEY: True}) == "lake"
    assert seen["scratch"] is True
    assert cl._scratch.get() is False
    d = cl.cfg_d_with_scratch(type("Cfg", (), {"to_dict": lambda self: {}})())
    assert d[cl.SEED_SCRATCH_KEY] is False


@pytest.mark.unit
def test_open_lake_without_actors_is_minio(monkeypatch):
    from lexis_markets.lake import cluster as cl

    sentinel = object()
    monkeypatch.setattr(cl, "seed_lake_worker_count", lambda **_k: 0)
    monkeypatch.setattr(cl, "LakeStore", lambda _cfg: sentinel)
    assert cl.open_lake(object()) is sentinel
