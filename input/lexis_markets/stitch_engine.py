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
)
from lexis_markets.storage import (
    LakeStore,
    PgClient,
    STITCHED_BAR_COLUMNS,
    compacted_data_key,
    month_in_range,
    month_prefix,
    months_in_range,
)

MONTH_READ_WORKERS = 16
ROLE_PRIORITY = {"primary": 0, "fill": 1, "cross_check": 2}


def month_bounds(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, monthrange(y, m)[1])


def clip_range(pg: PgClient, series_ids: list[str], start: date, end: date) -> tuple[date, date]:
    if len(series_ids) != 1:
        return start, end
    row = pg.fetchone(
        "SELECT first_seen, last_seen FROM series_meta WHERE series_id = %s",
        (series_ids[0],),
    )
    if not row:
        return start, end
    return max(start, row["first_seen"]), min(end, row["last_seen"])


def coverage_months(
    pg: PgClient, seg_df: pd.DataFrame, start: date, end: date
) -> list[tuple[int, int]]:
    symbols = sorted({str(s).upper() for s in seg_df["source_symbol"]})
    rows = pg.fetchall(
        """
        SELECT DISTINCT year, month
        FROM symbol_month_coverage
        WHERE UPPER(source_symbol) = ANY(%s)
        """,
        (symbols,),
    )
    if rows:
        return sorted((r["year"], r["month"]) for r in rows if month_in_range(r["year"], r["month"], start, end))
    return [(y, m) for y, m in months_in_range(start, end)]


def read_l1_month(
    lake: LakeStore,
    pg: PgClient,
    y: int,
    m: int,
    obs_cols: list[str],
    symbols: set[str],
) -> pd.DataFrame | None:
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


def load_stitch_segments(
    pg: PgClient,
    series_ids: list[str] | None,
) -> pd.DataFrame:
    cols = ["series_id", "source", "source_symbol", "valid_from", "valid_to", "role"]
    if series_ids is not None and not series_ids:
        return pd.DataFrame(columns=cols)
    if series_ids:
        rows = pg.fetchall(
            """
            SELECT series_id, source, source_symbol, valid_from, valid_to, role
            FROM stitch_segments
            WHERE series_id = ANY(%s)
            """,
            (series_ids,),
        )
    else:
        rows = pg.fetchall(
            """
            SELECT series_id, source, source_symbol, valid_from, valid_to, role
            FROM stitch_segments
            """
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def load_obs_for_segments(
    lake: LakeStore,
    pg: PgClient,
    start: date,
    end: date,
    seg_df: pd.DataFrame,
    *,
    revision_mode: str = "as_of",
) -> pd.DataFrame:
    if seg_df.empty:
        return pd.DataFrame()
    symbols = sorted({str(s).upper() for s in seg_df["source_symbol"]})
    obs_cols = [
        "source", "source_symbol", "ts", "open", "high", "low", "close", "volume", "adj_close",
    ]
    if revision_mode != "as_of":
        obs_cols.append("realtime_end")
    sym_set = set(symbols)
    months = coverage_months(pg, seg_df, start, end)
    with ThreadPoolExecutor(max_workers=min(MONTH_READ_WORKERS, len(months) or 1)) as ex:
        frames = [
            f
            for f in ex.map(
                lambda ym: read_l1_month(lake, pg, ym[0], ym[1], obs_cols, sym_set),
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
    if "valid_from" in merged.columns:
        merged = merged[merged["valid_from"].isna() | (merged["ts"] >= merged["valid_from"])]
    if "valid_to" in merged.columns:
        merged = merged[merged["valid_to"].isna() | (merged["ts"] <= merged["valid_to"])]
    if revision_mode != "as_of":
        fred = merged[merged["source"] == "fred"]
        other = merged[merged["source"] != "fred"]
        if not fred.empty:
            fred = fred.sort_values("realtime_end").groupby(["source_symbol", "ts"], as_index=False).tail(1)
        merged = pd.concat([other, fred], ignore_index=True)
    return merged


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
        (merged["series_id"] == series_id)
        & merged["close"].notna()
        & (merged["close"] > 0)
    ]
    pri = sub[sub["source"] == primary_source][["ts", "close"]]
    fil = sub[sub["source"] == fill_source][["ts", "close"]]
    both = pri.merge(fil, on="ts", suffixes=("_p", "_f"))
    if len(both) < CALIBRATION_MIN_OVERLAP_DAYS:
        return None
    return float((both["close_p"] / both["close_f"]).median())


def _apply_fill_calibration(bars: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
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
    return pd.concat(parts, ignore_index=True)


def stitch_merged(merged: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=STITCHED_BAR_COLUMNS)
    if merged.empty:
        return empty
    merged = merged[merged["close"].notna() & (merged["close"] > 0)].copy()
    if merged.empty:
        return empty
    merged["_pri"] = merged["role"].map(lambda r: ROLE_PRIORITY.get(r, 99))
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
    bars = _apply_fill_calibration(bars, merged)
    return bars


def resample_bars(bars: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "daily" or bars.empty:
        return bars
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"])
    rule = {"weekly": "W-FRI", "monthly": "ME"}[granularity]
    bars = (
        bars.set_index("ts")
        .groupby("series_id")
        .resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "adj_close": "last",
                "source": "last",
                "source_count": "max",
                "data_quality": "last",
            }
        )
        .reset_index()
    )
    bars["ts"] = bars["ts"].dt.date
    return bars


def stitch_live(
    lake: LakeStore,
    pg: PgClient,
    series_ids: list[str],
    start: date,
    end: date,
    *,
    revision_mode: str = "as_of",
) -> pd.DataFrame:
    seg_df = load_stitch_segments(pg, series_ids)
    merged = load_obs_for_segments(lake, pg, start, end, seg_df, revision_mode=revision_mode)
    return stitch_merged(merged)
