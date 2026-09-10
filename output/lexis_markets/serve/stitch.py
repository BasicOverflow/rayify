"""Merge multi-source L1 bars into canonical daily series.

Stitch policy (single version, no rule table):
  1. Load source rows for a series from L1 via ``stitch_segments`` (alias materialization).
  2. Drop jakewright bars in a long identical-close or linear-ramp run when another
     source disagrees on those dates (halt/copy-forward / synthetic stairs losing to
     a moving tape).
  3. For each (series_id, ts), pick the row from the highest-priority source in
     ``SOURCE_PRIORITY`` (lower index wins).
  4. When the winning source changes and the new segment is a fill source, scale OHLC
     onto the prior segment (raw overlap ratio × prior segment scale, or junction chain).
     Reject the fill if the ratio is outside calibration bounds.
  5. Collapse interior bars of a long identical-close run (halt / copy-forward) to endpoints.

``merge_canonical_cache`` prepends/appends stitched spans into the per-series L3 parquet.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import math
import os

import pandas as pd

from lexis_markets.config import (
    CALIBRATION_MAX_RATIO,
    CALIBRATION_MIN_OVERLAP_DAYS,
    CALIBRATION_MIN_RATIO,
    CALIBRATION_SPLIT_FACTORS,
    CALIBRATION_SPLIT_TOL,
    FILL_SOURCES,
    PRICE_COLS,
    SOURCE_PRIORITY,
)
from lexis_markets.domain.canonical_bar import CANONICAL_BAR_COLUMNS
from lexis_markets.domain.quality import (
    FLAT_CLOSE_MIN_RUN,
    collapse_halt_flats,
    flat_close_mask,
    linear_ramp_mask,
)
from lexis_markets.fred.alfred import collapse_fred_vintages
from lexis_markets.serve.revision import RevisionMode, resolve_collapse_as_of
from lexis_markets.domain.raw_bar import RAW_BAR_COLUMNS

_PRIORITY = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
# Secondary source must differ by this relative close to unseat a JW flat run.
JW_FLAT_DISAGREE_REL = 0.02


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


def _read_l1_month(
    lake,
    y: int,
    m: int,
    symbols: set[str],
    compacted_key: str | None,
) -> pd.DataFrame | None:
    from lexis_markets.lake import month_prefix

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
    if compacted_key:
        df = lake.get_df_parquet(compacted_key, columns=obs_cols, filters=filters)
        return df if not df.empty else None
    keys = [k for k in lake.list_keys(month_prefix(y, m)) if k.endswith(".parquet")]
    if not keys:
        return None
    frames = [lake.get_df_parquet(k, columns=obs_cols, filters=filters) for k in keys]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_months_for_symbols(
    lake,
    pg,
    months: list[tuple[int, int]],
    symbols: list[str] | set[str],
) -> pd.DataFrame:
    """Read L1 once for a month set × symbol union (coalesced MinIO path)."""
    if not months or not symbols:
        return pd.DataFrame()
    from lexis_markets.lake import compacted_data_keys

    sym_set = {str(s).upper() for s in symbols}
    key_by_month = compacted_data_keys(lake, pg, months)
    workers = min(int(os.environ.get("STITCH_MONTH_WORKERS", "8")), len(months) or 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        frames = [
            f
            for f in ex.map(
                lambda ym: _read_l1_month(
                    lake, ym[0], ym[1], sym_set, key_by_month.get(ym)
                ),
                months,
            )
            if f is not None
        ]
    if not frames:
        return pd.DataFrame()
    obs = pd.concat(frames, ignore_index=True)
    obs["source_symbol"] = obs["source_symbol"].astype(str).str.upper()
    obs["ts"] = pd.to_datetime(obs["ts"]).dt.date
    return obs


def merge_obs_with_segments(
    obs: pd.DataFrame,
    seg_df: pd.DataFrame,
    start: date,
    end: date,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Join preloaded L1 observations to stitch segments (no MinIO)."""
    if obs is None or obs.empty or seg_df.empty:
        return pd.DataFrame()
    symbols = {str(s).upper() for s in seg_df["source_symbol"]}
    part = obs[obs["source_symbol"].isin(symbols)].copy()
    if part.empty:
        return pd.DataFrame()
    part = part[(part["ts"] >= start) & (part["ts"] <= end)]
    segments = seg_df.copy()
    segments["source_symbol"] = segments["source_symbol"].astype(str).str.upper()
    merged = part.merge(segments, on=["source", "source_symbol"], how="inner")
    if merged.empty:
        return merged
    merged = merged[merged["valid_from"].isna() | (merged["ts"] >= merged["valid_from"])]
    merged = merged[merged["valid_to"].isna() | (merged["ts"] <= merged["valid_to"])]
    return collapse_fred_vintages(merged, as_of=as_of)


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
    months = _coverage_months(pg, seg_df, start, end)
    obs = load_months_for_symbols(lake, pg, months, symbols)
    return merge_obs_with_segments(obs, seg_df, start, end, as_of=as_of)


def drop_contradicted_jw_flats(
    merged: pd.DataFrame,
    *,
    min_run: int = FLAT_CLOSE_MIN_RUN,
    rel: float = JW_FLAT_DISAGREE_REL,
) -> pd.DataFrame:
    """Drop jakewright bars in a long flat or linear-ramp run when another source disagrees."""
    if merged.empty or "source" not in merged.columns or "series_id" not in merged.columns:
        return merged
    parts = []
    for _, grp in merged.groupby("series_id", sort=False):
        parts.append(_drop_contradicted_jw_flats_one(grp, min_run=min_run, rel=rel))
    return pd.concat(parts, ignore_index=True) if parts else merged


def _drop_contradicted_jw_flats_one(
    grp: pd.DataFrame, *, min_run: int, rel: float
) -> pd.DataFrame:
    jw = grp[grp["source"] == "jakewright"]
    if jw.empty:
        return grp
    jw = jw.sort_values("ts")
    close = pd.to_numeric(jw["close"], errors="coerce").to_numpy(dtype=float)
    mask = flat_close_mask(close, min_run=min_run) | linear_ramp_mask(close, min_run=min_run)
    if not mask.any():
        return grp
    other = grp[grp["source"] != "jakewright"]
    if other.empty:
        return grp
    jw_flat = jw.loc[mask, ["ts", "close"]].rename(columns={"close": "jw_close"})
    hit = other.merge(jw_flat, on="ts")
    if hit.empty:
        return grp
    hit = hit.copy()
    hit["close"] = pd.to_numeric(hit["close"], errors="coerce")
    hit["jw_close"] = pd.to_numeric(hit["jw_close"], errors="coerce")
    ok = hit["close"].notna() & hit["jw_close"].notna() & (hit["jw_close"].abs() > 0)
    hit = hit.loc[ok]
    if hit.empty:
        return grp
    disagree = (hit["close"] - hit["jw_close"]).abs() / hit["jw_close"].abs() > rel
    bad_ts = set(hit.loc[disagree, "ts"])
    if not bad_ts:
        return grp
    drop = (grp["source"] == "jakewright") & grp["ts"].isin(bad_ts)
    return grp.loc[~drop]


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


def _raw_close_on(
    merged: pd.DataFrame, series_id: str, source: str, ts
) -> float | None:
    sub = merged[
        (merged["series_id"] == series_id)
        & (merged["source"] == source)
        & (merged["ts"] == ts)
        & merged["close"].notna()
        & (merged["close"] > 0)
    ]
    if sub.empty:
        return None
    return float(sub.iloc[0]["close"])


def plausible_calibration_ratio(k: float | None) -> bool:
    """True when ``k`` is inside the default band or a known split factor."""
    if k is None or k <= 0 or not math.isfinite(k):
        return False
    if CALIBRATION_MIN_RATIO <= k <= CALIBRATION_MAX_RATIO:
        return True
    for n in CALIBRATION_SPLIT_FACTORS:
        target = float(n)
        inv = 1.0 / target
        if abs(k - target) / target <= CALIBRATION_SPLIT_TOL:
            return True
        if abs(k - inv) / inv <= CALIBRATION_SPLIT_TOL:
            return True
    return False


def _fill_scale_onto_prior(
    merged: pd.DataFrame,
    series_id: str,
    pri_src: str,
    fill_src: str,
    pri_close: float,
    pri_last_ts,
    fill_first_ts,
) -> float | None:
    """Scale factor mapping fill OHLC onto the already-calibrated prior segment.

    Raw overlap ratios alone are wrong after a prior hop (e.g. JC→YF then YF→MP):
    YF may already be scaled onto JC while MP/YF tip overlap is ~1.0, which leaves an
    unscaled MP stub. Multiply the raw overlap ratio by the prior segment's current
    scale (calibrated last close / raw prior close). With no overlap, chain at the
    junction: ``pri_close / fill_first_raw``.
    """
    if pri_close <= 0:
        return None
    fill_raw = _raw_close_on(merged, series_id, fill_src, fill_first_ts)
    if fill_raw is None or fill_raw <= 0:
        return None
    k_raw = _overlap_calibration_factor(merged, series_id, pri_src, fill_src)
    pri_raw = _raw_close_on(merged, series_id, pri_src, pri_last_ts)
    if k_raw is not None and pri_raw is not None and pri_raw > 0:
        return k_raw * (pri_close / pri_raw)
    return pri_close / fill_raw


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
            k = _fill_scale_onto_prior(
                merged,
                series_id,
                pri_src,
                fill_src,
                pri_close,
                g.iloc[pe]["ts"],
                g.iloc[runs[i]["start"]]["ts"],
            )
            if not plausible_calibration_ratio(k):
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
    seg_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if seg_df is None:
        seg_df = load_stitch_segments(pg, [series_id])
    else:
        seg_df = seg_df[seg_df["series_id"] == series_id]
    collapse_as_of = resolve_collapse_as_of(
        revision_mode=revision_mode,
        request_end=end,
        as_of=as_of,
    )
    merged = load_source_bars(lake, pg, seg_df, start, end, as_of=collapse_as_of)
    merged = drop_contradicted_jw_flats(merged)
    bars = pick_winners(merged)
    bars = calibrate_fill_gaps(bars, merged)
    return collapse_halt_flats(bars)


def stitch_series_from_obs(
    series_id: str,
    start: date,
    end: date,
    seg_df: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    revision_mode: RevisionMode = "latest",
    as_of: date | None = None,
) -> pd.DataFrame:
    """Stitch one series from a preloaded multi-symbol L1 observation frame."""
    if seg_df is None or seg_df.empty:
        return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    seg = seg_df[seg_df["series_id"] == series_id]
    if seg.empty:
        return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    collapse_as_of = resolve_collapse_as_of(
        revision_mode=revision_mode,
        request_end=end,
        as_of=as_of,
    )
    merged = merge_obs_with_segments(obs, seg, start, end, as_of=collapse_as_of)
    merged = drop_contradicted_jw_flats(merged)
    bars = pick_winners(merged)
    bars = calibrate_fill_gaps(bars, merged)
    return collapse_halt_flats(bars)
