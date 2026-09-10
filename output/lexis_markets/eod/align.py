"""Ray remote task: scale jakewright OHLC onto yfinance adj close on overlap days (seed)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import ray

from lexis_markets.config import PRICE_COLS
from lexis_markets.lake import LakeStore, PgClient, months_in_range, lake_from_cfg_d
from lexis_markets.kaggle.compact import load_month_raw_bars, persist_month_frame
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.runtime import DEFAULT_REMOTE_OPTS

logger = get_logger("eod.align")


def _overlap_targets(pg: PgClient, symbols: list[str]) -> list[dict]:
    if not symbols:
        return []
    upper = [s.upper() for s in symbols]
    return pg.fetchall(
        """
        SELECT jw.series_id,
               UPPER(jw.source_symbol) AS jw_sym,
               UPPER(yf.source_symbol) AS yf_sym
        FROM symbol_aliases jw
        JOIN symbol_aliases yf ON jw.series_id = yf.series_id
        WHERE jw.source = 'jakewright'
          AND yf.source = 'yfinance'
          AND UPPER(jw.source_symbol) = ANY(%s)
        """,
        (upper,),
    )


def _coverage_months(pg: PgClient, targets: list[dict]) -> list[tuple[int, int]]:
    if not targets:
        return []
    rows = pg.fetchall(
        """
        SELECT DISTINCT year, month FROM symbol_month_coverage
        WHERE source IN ('jakewright', 'yfinance')
          AND UPPER(source_symbol) = ANY(%s)
        """,
        ([t["jw_sym"] for t in targets] + [t["yf_sym"] for t in targets],),
    )
    months = [(int(r["year"]), int(r["month"])) for r in rows]
    if months:
        return sorted(set(months))
    jw_syms = {t["jw_sym"] for t in targets}
    bounds = pg.fetchall(
        """
        SELECT MIN(first_seen) AS lo, MAX(last_seen) AS hi
        FROM series_meta m
        JOIN symbol_aliases a ON a.series_id = m.series_id
        WHERE a.source = 'jakewright' AND UPPER(a.source_symbol) = ANY(%s)
        """,
        (list(jw_syms),),
    )
    if bounds and bounds[0]["lo"] and bounds[0]["hi"]:
        return months_in_range(bounds[0]["lo"], bounds[0]["hi"])
    return []


def resolve_align_months(pg: PgClient, symbols: list[str]) -> tuple[list[dict], list[tuple[int, int]]]:
    """JW/YF overlap targets and the months those symbols cover. Months are independent."""
    targets = _overlap_targets(pg, symbols)
    if not targets:
        return [], []
    return targets, _coverage_months(pg, targets)


def _yf_close_lookup(yf_df: pd.DataFrame) -> dict[date, float]:
    """Prefer adj_close (already scaled into close when YF was ingested via map_ohlcv)."""
    out: dict[date, float] = {}
    if yf_df.empty:
        return out
    yf_df = yf_df.copy()
    yf_df["ts"] = pd.to_datetime(yf_df["ts"]).dt.date
    for _, row in yf_df.iterrows():
        val = row.get("close")
        if pd.isna(val) or float(val) <= 0:
            val = row.get("adj_close")
        if pd.notna(val) and float(val) > 0:
            out[row["ts"]] = float(val)
    return out


def _align_month(lake: LakeStore, pg: PgClient, year: int, month: int, targets: list[dict]) -> int:
    df = load_month_raw_bars(lake, year, month)
    if df.empty:
        return 0

    df = df.copy()
    df["source_symbol"] = df["source_symbol"].astype(str).str.upper()
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    patched = 0

    for t in targets:
        jw_sym, yf_sym = t["jw_sym"], t["yf_sym"]
        yf_rows = df[(df["source"] == "yfinance") & (df["source_symbol"] == yf_sym)]
        yf_map = _yf_close_lookup(yf_rows)
        if not yf_map:
            continue
        mask = (df["source"] == "jakewright") & (df["source_symbol"] == jw_sym)
        idx = df.index[mask & df["ts"].isin(yf_map.keys())]
        for i in idx:
            ts = df.at[i, "ts"]
            jw_close = df.at[i, "close"]
            if pd.isna(jw_close) or float(jw_close) <= 0:
                continue
            factor = yf_map[ts] / float(jw_close)
            for col in PRICE_COLS:
                val = df.at[i, col]
                if pd.notna(val):
                    df.at[i, col] = float(val) * factor
            if "adj_close" in df.columns:
                df.at[i, "adj_close"] = yf_map[ts]
            patched += 1

    if patched:
        persist_month_frame(lake, pg, year, month, df)
    return patched


@ray.remote(**DEFAULT_REMOTE_OPTS)
def task_align_yf_month(cfg_d: dict, year: int, month: int, targets: list[dict]) -> dict:
    """One month, one writer. Safe to run many of these in parallel."""
    from lexis_markets.config import MarketsConfig

    cfg = MarketsConfig.from_dict(cfg_d)
    lake = lake_from_cfg_d(cfg_d)
    pg = PgClient(cfg.postgres_url)
    patched = _align_month(lake, pg, int(year), int(month), targets)
    logger.info(
        "yf_align month=%s-%02d targets=%s patched=%s",
        year,
        month,
        len(targets),
        patched,
    )
    return {
        "year": int(year),
        "month": int(month),
        "targets": len(targets),
        "patched_rows": patched,
    }
