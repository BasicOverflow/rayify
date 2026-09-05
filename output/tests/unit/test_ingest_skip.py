"""Unit tests: L1 ingest entrypoints skip when markers exist."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lexis_markets.kaggle import ingest as ingest_l1
from lexis_markets.kaggle.ingest import (
    JC_MARKER,
    JW_MARKER,
    ingest_jacksoncrow,
    ingest_jakewright,
)


@pytest.mark.unit
def test_ingest_jakewright_skips_when_marker_present(monkeypatch):
    monkeypatch.setattr(
        ingest_l1,
        "get_json",
        lambda _lake, _key: {"source": "jakewright", "rows": 9, "chunks": 1, "symbols": 1},
    )
    monkeypatch.setattr(ingest_l1, "clear_staging_after_ingest", lambda *_a, **_k: 2)
    prepare = MagicMock()
    monkeypatch.setattr(ingest_l1, "prepare_jakewright_staging", prepare)

    lake = MagicMock()
    lake.exists.side_effect = lambda key: key == JW_MARKER
    out = ingest_jakewright(MagicMock(), lake)
    assert out["rows"] == 9
    prepare.remote.assert_not_called()


@pytest.mark.unit
def test_ingest_jacksoncrow_skips_when_marker_present(monkeypatch):
    monkeypatch.setattr(
        ingest_l1,
        "get_json",
        lambda _lake, _key: {"source": "jacksoncrow", "rows": 7, "chunks": 1, "symbols": 1},
    )
    monkeypatch.setattr(ingest_l1, "clear_staging_after_ingest", lambda *_a, **_k: 1)
    prepare = MagicMock()
    monkeypatch.setattr(ingest_l1, "prepare_jacksoncrow_staging", prepare)

    lake = MagicMock()
    lake.exists.side_effect = lambda key: key == JC_MARKER
    out = ingest_jacksoncrow(MagicMock(), lake)
    assert out["rows"] == 7
    prepare.remote.assert_not_called()
