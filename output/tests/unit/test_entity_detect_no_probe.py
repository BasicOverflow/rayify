"""Discovery registers MP+NASDAQ symbols; YF support is decided on history pull."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from lexis_markets.eod.entity import MpSymbolSpan, detect_new_entities
from lexis_markets.registry.nasdaq import NasdaqDirectory, NasdaqEntry
from lexis_markets.registry.universe import patch_yf_skip_failures


def _directory(*symbols: str) -> NasdaqDirectory:
    by_symbol = {
        s: NasdaqEntry(symbol=s, exchange="Q", etf=False)
        for s in symbols
    }
    return NasdaqDirectory(by_symbol=by_symbol, symbols=set(symbols))


@pytest.mark.unit
def test_detect_new_entities_skips_yf_probe():
    lake = MagicMock()
    pg = MagicMock()
    pg.fetchall.return_value = []  # no known symbols
    cfg = MagicMock()
    cfg.yf_backfill_limit.return_value = None

    spans = {
        "AAA": MpSymbolSpan("AAA", "equity", date(2024, 1, 2), date(2024, 1, 10)),
        "BBB": MpSymbolSpan("BBB", "equity", date(2024, 1, 3), date(2024, 1, 9)),
    }
    with patch("lexis_markets.eod.entity.scan_mp_window", return_value=spans):
        with patch("yfinance.download") as yf_dl:
            out = detect_new_entities(
                lake, pg, _directory("AAA", "BBB"), cfg=cfg, today=date(2024, 1, 11)
            )
            yf_dl.assert_not_called()

    assert {c["symbol"] for c in out} == {"AAA", "BBB"}
    assert all(c["yf_backfill"] is True for c in out)


@pytest.mark.unit
def test_detect_new_entities_requires_nasdaq_listing():
    lake = MagicMock()
    pg = MagicMock()
    pg.fetchall.return_value = []
    cfg = MagicMock()
    cfg.yf_backfill_limit.return_value = None
    spans = {
        "AAA": MpSymbolSpan("AAA", "equity", date(2024, 1, 2), date(2024, 1, 10)),
        "ZZZ": MpSymbolSpan("ZZZ", "equity", date(2024, 1, 2), date(2024, 1, 10)),
    }
    with patch("lexis_markets.eod.entity.scan_mp_window", return_value=spans):
        out = detect_new_entities(lake, pg, _directory("AAA"), cfg=cfg)

    assert [c["symbol"] for c in out] == ["AAA"]


@pytest.mark.unit
def test_patch_yf_skip_marks_mp_discovery_unsupported():
    pg = MagicMock()
    pg.fetchone.return_value = {"n": 1}
    details = [
        {"source": "eod_failed", "series_id": "equity:NEW", "error": "empty history"},
        {"source": "eod_ok", "series_id": "equity:OK"},
    ]
    n = patch_yf_skip_failures(pg, details)
    assert n == 1
    # mark_yf_skip + UNSUPPORTED update for marketparquet discoveries
    assert pg.fetchone.called
    assert pg.execute.called
    sql = pg.execute.call_args[0][0]
    assert "UNSUPPORTED" in sql
    assert "marketparquet" in sql
    assert pg.execute.call_args[0][1] == (["equity:NEW"],)
