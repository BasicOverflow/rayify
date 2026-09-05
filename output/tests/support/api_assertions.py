"""Assertions for live Serve API responses."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from lexis_markets.domain.quality import series_quality_window


def parse_rows(body: dict) -> pd.DataFrame:
    rows = body.get("rows") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"]).dt.date
    return df


def assert_range(df: pd.DataFrame, start: str, end: str) -> None:
    assert not df.empty, "expected non-empty series"
    lo, hi = date.fromisoformat(start), date.fromisoformat(end)
    assert df["ts"].min() >= lo, f"first ts {df['ts'].min()} before {lo}"
    assert df["ts"].max() <= hi, f"last ts {df['ts'].max()} after {hi}"


def assert_ohlcv_sanity(df: pd.DataFrame) -> None:
    for _, r in df.iterrows():
        o, h, l, c = r.get("open"), r.get("high"), r.get("low"), r.get("close")
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            continue
        assert l <= min(o, c) + 1e-6, f"low violation at {r.get('ts')}"
        assert h + 1e-6 >= max(o, c), f"high violation at {r.get('ts')}"
        if "volume" in r and r["volume"] is not None and not pd.isna(r["volume"]):
            assert float(r["volume"]) >= 0


def assert_quality(df: pd.DataFrame, calendar_id: str, *, min_score: float = 0.5) -> dict:
    from lexis_markets.domain.quality import series_quality_window

    q = series_quality_window(df, calendar_id)
    assert q["quality_score"] >= min_score, (
        f"quality_score {q['quality_score']} below {min_score} (gaps={q['gap_count']})"
    )
    return q


def assert_features(body: dict, feature_names: list[str]) -> None:
    rows = body.get("rows") or []
    assert rows, "no rows for feature check"
    for name in feature_names:
        assert name in rows[0], f"missing feature column {name}"


def assert_revision_differs(latest_body: dict, as_of_body: dict) -> None:
    latest = parse_rows(latest_body)
    as_of = parse_rows(as_of_body)
    assert not latest.empty and not as_of.empty, "macro revision bodies empty"
    merged = latest.merge(as_of, on="ts", suffixes=("_latest", "_asof"))
    assert not merged.empty, "no overlapping timestamps between latest and as_of"
    assert "close_latest" in merged.columns and "close_asof" in merged.columns
    diff = (merged["close_latest"] - merged["close_asof"]).abs()
    assert (diff > 1e-9).any(), "expected at least one close to differ between latest and as_of"
