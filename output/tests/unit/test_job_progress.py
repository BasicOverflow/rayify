"""Unit tests: durable job progress resume."""
from __future__ import annotations

import json

import pytest

from lexis_markets.jobs.progress import (
    JobProgress,
    _expand_done_keys,
    scope_fingerprint,
    stamp_extra_key,
)


class _MemLake:
    def __init__(self):
        self._obj: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self._obj

    def put_bytes(self, key: str, data: bytes, _ctype: str = "") -> None:
        self._obj[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self._obj[key]

    def delete_keys(self, keys: list[str]) -> None:
        for k in keys:
            self._obj.pop(k, None)


@pytest.mark.unit
def test_scope_fingerprint_stable():
    a = scope_fingerprint({"statuses": ["ACTIVE", "DELISTED"], "limit": None})
    b = scope_fingerprint({"limit": None, "statuses": ["ACTIVE", "DELISTED"]})
    assert a == b
    assert len(a) == 16


@pytest.mark.unit
def test_expand_done_keys_aliases():
    keys = _expand_done_keys(["equity:AAPL", "deep:etf:SPY"])
    assert "equity:AAPL" in keys
    assert "tip:equity:AAPL" in keys
    assert "deep:equity:AAPL" in keys
    assert "etf:SPY" in keys
    assert "tip:etf:SPY" in keys


@pytest.mark.unit
def test_job_progress_mark_and_resume(monkeypatch):
    from lexis_markets.jobs import progress as prog_mod

    lake = _MemLake()
    monkeypatch.setattr(prog_mod, "put_json", lambda lk, key, obj: lk.put_bytes(key, json.dumps(obj).encode()))
    monkeypatch.setattr(prog_mod, "get_json", lambda lk, key: json.loads(lk.get_bytes(key)))
    monkeypatch.setattr(prog_mod, "utcnow", lambda: __import__("datetime").datetime(2026, 1, 2, tzinfo=__import__("datetime").timezone.utc))

    p1 = JobProgress(lake, job="cache_fill", scope={"phase": "tip"}, pg=None)
    p1.load()
    rows = [{"series_id": f"equity:S{i}"} for i in range(5)]
    pending = p1.pending(rows)
    assert len(pending) == 5
    p1.mark_done(["equity:S0", "equity:S1"], stamp_pg=False)
    assert lake.exists(p1.key)

    p2 = JobProgress(lake, job="cache_fill", scope={"phase": "tip"}, pg=None)
    p2.load()
    left = p2.pending(rows)
    assert [r["series_id"] for r in left] == ["equity:S2", "equity:S3", "equity:S4"]
    p2.complete()
    assert p2.is_complete()

    p3 = JobProgress(lake, job="cache_fill", scope={"phase": "tip"}, pg=None)
    p3.load()
    assert p3.is_complete()


@pytest.mark.unit
def test_stamp_extra_key_format():
    sid = scope_fingerprint({"a": 1})
    assert stamp_extra_key("cache_fill", sid).startswith("job_done__cache_fill__")
