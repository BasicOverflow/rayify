"""Unit tests: FRED revision_mode collapse."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from lexis_markets.fred.alfred import collapse_fred_vintages
from lexis_markets.serve.revision import resolve_collapse_as_of


def _fred_row(
    ts: str,
    value: float,
    *,
    realtime_start: str,
    realtime_end: str,
) -> dict:
    return {
        "source": "fred",
        "source_symbol": "DGS10",
        "ts": date.fromisoformat(ts),
        "close": value,
        "realtime_start": date.fromisoformat(realtime_start),
        "realtime_end": date.fromisoformat(realtime_end),
    }


@pytest.mark.unit
def test_as_of_picks_vintage_at_or_before_cutoff():
    # ALFRED: first vintage live 2020-01-15..2020-01-31; revised from 2020-02-01.
    df = pd.DataFrame(
        [
            _fred_row(
                "2020-01-01",
                1.0,
                realtime_start="2020-01-15",
                realtime_end="2020-01-31",
            ),
            _fred_row(
                "2020-01-01",
                1.5,
                realtime_start="2020-02-01",
                realtime_end="9999-12-31",
            ),
        ]
    )
    as_of = resolve_collapse_as_of(
        revision_mode="as_of", request_end=date(2020, 12, 31), as_of=date(2020, 1, 20)
    )
    out = collapse_fred_vintages(df, as_of=as_of)
    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 1.0


@pytest.mark.unit
def test_latest_picks_newest_vintage():
    df = pd.DataFrame(
        [
            _fred_row(
                "2020-01-01",
                1.0,
                realtime_start="2020-01-15",
                realtime_end="2020-01-31",
            ),
            _fred_row(
                "2020-01-01",
                1.5,
                realtime_start="2020-02-01",
                realtime_end="9999-12-31",
            ),
        ]
    )
    as_of = resolve_collapse_as_of(
        revision_mode="latest", request_end=date(2020, 12, 31), as_of=None
    )
    out = collapse_fred_vintages(df, as_of=as_of)
    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 1.5
