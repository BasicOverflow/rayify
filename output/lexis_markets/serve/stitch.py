"""Merge multi-source L1 bars into canonical daily series.

Stitch policy (single version, no rule table):
  1. Load source rows for a series from L1 via ``stitch_segments`` (alias materialization).
  2. For each (series_id, ts), pick the row from the highest-priority source in
     ``SOURCE_PRIORITY`` (lower index wins).
  3. When the winning source changes and the new segment is marketparquet/yfinance,
     scale OHLC onto the prior segment using median close ratio over overlap days.
     Reject the fill if overlap is too short or the ratio is outside calibration bounds.

``merge_canonical_cache`` prepends/appends stitched spans into the per-series L3 parquet.
"""
from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from lexis_markets.config import (
    CALIBRATION_MAX_RATIO,
    CALIBRATION_MIN_OVERLAP_DAYS,
    CALIBRATION_MIN_RATIO,
    FILL_SOURCES,
    PRICE_COLS,
    SOURCE_PRIORITY,
)
from lexis_markets.domain.canonical_bar import CANONICAL_BAR_COLUMNS
from lexis_markets.fred.alfred import collapse_fred_vintages
from lexis_markets.serve.revision import RevisionMode, resolve_collapse_as_of
from lexis_markets.domain.raw_bar import RAW_BAR_COLUMNS

_PRIORITY = {s: i for i, s in enumerate(SOURCE_PRIORITY)}


def month_bounds(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, monthrange(y, m)[1])


def merge_canonical_cache(existing: pd.DataFrame | None, new_bars: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = new_bars.copy()
    elif new_bars.empty:
        out = existing.copy()
    else:
        out = pd.concat([existing, new_bars], ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    out = out.sort_values(["series_id", "ts"])
    out = out.drop_duplicates(subset=["series_id", "ts"], keep="last")
    return out.reset_index(drop=True)


def load_stitch_segments(pg, series_ids: list[str] | None) -> pd.DataFrame:
    cols = ["series_id", "source", "source_symbol", "valid_from", "valid_to"]
    if series_ids is not None and not series_ids:
        return pd.DataFrame(columns=cols)
    if series_ids:
        rows = pg.fetchall(
            """
            SELECT series_id, source, source_symbol, valid_from, valid_to
            FROM stitch_segments WHERE series_id = ANY(%s)
            """,
            (series_ids,),
        )
    else:
        rows = pg.fetchall(
            "SELECT series_id, source, source_symbol, valid_from, valid_to FROM stitch_segments"
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _coverage_months(pg, seg_df: pd.DataFrame, start: date, end: date) -> list[tuple[int, int]]:
    from lexis_markets.lake import month_in_range, months_in_range

    symbols = sorted({str(s).upper() for s in seg_df["source_symbol"]})
    rows = pg.fetchall(
        """
        SELECT DISTINCT year, month FROM symbol_month_coverage
        WHERE UPPER(source_symbol) = ANY(%s)
        """,
        (symbols,),
    )
    if rows:
        return sorted(
            (r["year"], r["month"])
            for r in rows
            if month_in_range(r["year"], r["month"], start, end)
        )
    return months_in_range(start, end)


def _read_l1_month(lake, pg, y: int, m: int, symbols: set[str]) -> pd.DataFrame | None:
    from lexis_markets.lake import compacted_data_key, month_prefix

    obs_cols = [
        "source",
        "source_symbol",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "realtime_start",
        "realtime_end",
    ]
    filters = [("source_symbol", "in", sorted(symbols))]
    key = compacted_data_key(lake, pg, y, m)
    if key:
        df = lake.get_df_parquet(key, columns=obs_cols, filters=filters)
        return df if not df.empty else None
    keys = [k for k in lake.list_keys(month_prefix(y, m)) if k.endswith(".parquet")]
    if not keys:
        return None
    frames = [lake.get_df_parquet(k, columns=obs_cols, filters=filters) for k in keys]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_source_bars(
    lake,
    pg,
    seg_df: pd.DataFrame,
    start: date,
    end: date,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    if seg_df.empty:
        return pd.DataFrame()
    symbols = sorted({str(s).upper() for s in seg_df["source_symbol"]})
    sym_set = set(symbols)
    months = _coverage_months(pg, seg_df, start, end)
    workers = min(16, len(months) or 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        frames = [
            f
            for f in ex.map(
                lambda ym: _read_l1_month(lake, pg, ym[0], ym[1], sym_set),
                months,
            )
            if f is not None
        ]
    if not frames:
        return pd.DataFrame()
    obs = pd.concat(frames, ignore_index=True)
    obs["source_symbol"] = obs["source_symbol"].astype(str).str.upper()
    obs["ts"] = pd.to_datetime(obs["ts"]).dt.date
    obs = obs[(obs["ts"] >= start) & (obs["ts"] <= end)]
    segments = seg_df.copy()
    segments["source_symbol"] = segments["source_symbol"].astype(str).str.upper()
    merged = obs.merge(segments, on=["source", "source_symbol"], how="inner")
    if merged.empty:
        return merged
    merged = merged[merged["valid_from"].isna() | (merged["ts"] >= merged["valid_from"])]
    merged = merged[merged["valid_to"].isna() | (merged["ts"] <= merged["valid_to"])]
    return collapse_fred_vintages(merged, as_of=as_of)


def pick_winners(merged: pd.DataFrame) -> pd.DataFrame:
    """One bar per (series_id, ts): lowest ``SOURCE_PRIORITY`` index wins."""
    empty = pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    if merged.empty:
        return empty
    merged = merged[merged["close"].notna() & (merged["close"] > 0)].copy()
    if merged.empty:
        return empty
    merged["_pri"] = merged["source"].map(lambda s: _PRIORITY.get(s, 99))
    merged = merged.sort_values(["series_id", "ts", "_pri"])
    counts = merged.groupby(["series_id", "ts"])["source"].nunique().rename("source_count")
    winners = merged.groupby(["series_id", "ts"], as_index=False).first()
    winners = winners.merge(counts.reset_index(), on=["series_id", "ts"])
    bars = winners[
        [
            "series_id", "ts", "open", "high", "low", "close", "volume", "adj_close",
            "source", "source_count",
        ]
    ].copy()
    bars["data_quality"] = "ok"
    return bars


def _segment_runs(bars: pd.DataFrame) -> list[dict]:
    if bars.empty:
        return []
    runs = []
    start = 0
    for i in range(1, len(bars)):
        if bars.iloc[i]["source"] != bars.iloc[i - 1]["source"]:
            runs.append({"start": start, "end": i - 1, "source": bars.iloc[start]["source"]})
            start = i
    runs.append({"start": start, "end": len(bars) - 1, "source": bars.iloc[start]["source"]})
    return runs


def _overlap_calibration_factor(
    merged: pd.DataFrame, series_id: str, primary_source: str, fill_source: str
) -> float | None:
    sub = merged[
        (merged["series_id"] == series_id) & merged["close"].notna() & (merged["close"] > 0)
    ]
    pri = sub[sub["source"] == primary_source][["ts", "close"]]
    fil = sub[sub["source"] == fill_source][["ts", "close"]]
    both = pri.merge(fil, on="ts", suffixes=("_p", "_f"))
    if len(both) < CALIBRATION_MIN_OVERLAP_DAYS:
        return None
    return float((both["close_p"] / both["close_f"]).median())


def calibrate_fill_gaps(bars: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Scale fill-source OHLC onto the prior segment; drop runs that fail calibration."""
    if bars.empty:
        return bars
    parts = []
    for _, grp in bars.groupby("series_id", sort=False):
        g = grp.sort_values("ts").reset_index(drop=True)
        runs = _segment_runs(g)
        for i in range(1, len(runs)):
            fill_src = runs[i]["source"]
            if fill_src not in FILL_SOURCES:
                continue
            pri_src = runs[i - 1]["source"]
            sl = slice(runs[i]["start"], runs[i]["end"] + 1)
            pe = runs[i - 1]["end"]
            pri_close = float(g.iloc[pe]["close"])
            series_id = g.iloc[0]["series_id"]
            k = _overlap_calibration_factor(merged, series_id, pri_src, fill_src)
            if k is None:
                fill_first_ts = g.iloc[runs[i]["start"]]["ts"]
                raw = merged[
                    (merged["series_id"] == series_id)
                    & (merged["source"] == fill_src)
                    & (merged["ts"] == fill_first_ts)
                ]
                if raw.empty or float(raw.iloc[0]["close"]) <= 0 or pri_close <= 0:
                    g.iloc[sl, g.columns.get_loc("data_quality")] = "stitch_break"
                    continue
                k = pri_close / float(raw.iloc[0]["close"])
            if k < CALIBRATION_MIN_RATIO or k > CALIBRATION_MAX_RATIO:
                g.iloc[sl, g.columns.get_loc("data_quality")] = "stitch_break"
                continue
            for col in PRICE_COLS:
                g.iloc[sl, g.columns.get_loc(col)] = g.iloc[sl, g.columns.get_loc(col)] * k
        g = g[g["data_quality"] != "stitch_break"]
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)


def stitch_series(
    lake,
    pg,
    series_id: str,
    start: date,
    end: date,
    *,
    revision_mode: RevisionMode = "latest",
    as_of: date | None = None,
) -> pd.DataFrame:
    seg_df = load_stitch_segments(pg, [series_id])
    collapse_as_of = resolve_collapse_as_of(
        revision_mode=revision_mode,
        request_end=end,
        as_of=as_of,
    )
    merged = load_source_bars(lake, pg, seg_df, start, end, as_of=collapse_as_of)
    bars = pick_winners(merged)
    return calibrate_fill_gaps(bars, merged)
