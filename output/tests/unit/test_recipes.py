"""Unit tests: dataset recipes and NaN policies."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from lexis_markets.domain.recipes import (
    bars_to_wide_matrix,
    canonicalize_recipe_mode,
    parse_nan_policy,
)


@pytest.mark.unit
def test_canonicalize_recipe_aliases():
    assert canonicalize_recipe_mode("range") == "range_panel"
    assert canonicalize_recipe_mode(None) == "range_panel"


@pytest.mark.unit
def test_parse_nan_policy():
    assert parse_nan_policy(None) == "keep"
    assert parse_nan_policy("ffill") == "ffill"
    with pytest.raises(ValueError):
        parse_nan_policy("mean")


@pytest.mark.unit
def test_wide_matrix_keep_and_drop():
    bars = pd.DataFrame(
        {
            "series_id": ["equity:A", "equity:A", "equity:B"],
            "ts": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 1)],
            "close": [1.0, 2.0, 10.0],
        }
    )
    keep = bars_to_wide_matrix(bars, nan_policy="keep")
    assert list(keep.columns) == ["ts", "equity:A", "equity:B"]
    assert len(keep) == 2
    assert pd.isna(keep.loc[keep["ts"] == date(2020, 1, 2), "equity:B"].iloc[0])

    dropped = bars_to_wide_matrix(bars, nan_policy="drop_rows")
    assert len(dropped) == 1
    assert dropped.iloc[0]["ts"] == date(2020, 1, 1)


@pytest.mark.unit
def test_wide_matrix_ffill():
    bars = pd.DataFrame(
        {
            "series_id": ["equity:A", "equity:A", "equity:B", "equity:B"],
            "ts": [
                date(2020, 1, 1),
                date(2020, 1, 3),
                date(2020, 1, 1),
                date(2020, 1, 2),
            ],
            "close": [1.0, 3.0, 10.0, 11.0],
        }
    )
    wide = bars_to_wide_matrix(bars, nan_policy="ffill")
    row2 = wide.loc[wide["ts"] == date(2020, 1, 2)].iloc[0]
    assert row2["equity:A"] == 1.0  # ffilled from Jan 1
    assert row2["equity:B"] == 11.0
