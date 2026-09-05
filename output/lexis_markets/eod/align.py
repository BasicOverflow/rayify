"""Ray remote task: patch jakewright closes to yfinance truth on overlap days (seed pass)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import ray

from lexis_markets.lake import LakeStore, PgClient, compacted_data_key, months_in_range, write_parquet_lake
from lexis_markets.domain.raw_bar import RAW_BAR_COLUMNS
from lexis_markets.logging_setup import get_logger

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


def _yf_close_lookup(yf_df: pd.DataFrame) -> dict[date, float]:
    out: dict[date, float] = {}
    if yf_df.empty:
        return out
    yf_df = yf_df.copy()
    yf_df["ts"] = pd.to_datetime(yf_df["ts"]).dt.date
    for _, row in yf_df.iterrows():
        val = row.get("adj_close")
        if pd.isna(val):
            val = row.get("close")
        if pd.notna(val) and float(val) > 0:
            out[row["ts"]] = float(val)
    return out


def has_yfinance_coverage(pg: PgClient) -> bool:
    """True when any yfinance month coverage exists (EOD has written into L1)."""
    row = pg.fetchone(
        """
        SELECT 1 AS ok
        FROM symbol_month_coverage
        WHERE source = 'yfinance'
        LIMIT 1
        """
    )
    return bool(row)


def _align_month(lake: LakeStore, pg: PgClient, year: int, month: int, targets: list[dict]) -> int:
    key = compacted_data_key(lake, pg, year, month)
    if not key:
        return 0
    jw_syms = {t["jw_sym"] for t in targets}
    yf_syms = {t["yf_sym"] for t in targets}
    all_syms = sorted(jw_syms | yf_syms)
    df = lake.get_df_parquet(key, columns=RAW_BAR_COLUMNS, filters=[("source_symbol", "in", all_syms)])
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
            df.at[i, "close"] = yf_map[ts]
            patched += 1

    if patched:
        write_parquet_lake(lake, key, df[RAW_BAR_COLUMNS], sort_by=["source", "source_symbol", "ts"])
    return patched


@ray.remote
def task_align_yf_batch(cfg_d: dict, symbols: list[str]) -> dict:
    from lexis_markets.config import MarketsConfig

    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    targets = _overlap_targets(pg, symbols)
    if not targets:
        return {"symbols": len(symbols), "targets": 0, "patched_rows": 0}

    rows = pg.fetchall(
        """
        SELECT DISTINCT year, month FROM symbol_month_coverage
        WHERE source IN ('jakewright', 'yfinance')
          AND UPPER(source_symbol) = ANY(%s)
        """,
        ([t["jw_sym"] for t in targets] + [t["yf_sym"] for t in targets],),
    )
    months = [(r["year"], r["month"]) for r in rows]
    if not months:
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
            months = months_in_range(bounds[0]["lo"], bounds[0]["hi"])

    patched = sum(_align_month(lake, pg, y, m, targets) for y, m in months)
    logger.info("seed_align symbols=%s targets=%s patched=%s", len(symbols), len(targets), patched)
    return {"symbols": len(symbols), "targets": len(targets), "patched_rows": patched}
