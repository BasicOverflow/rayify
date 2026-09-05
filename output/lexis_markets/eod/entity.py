"""Detect new listings from the MarketParquet free window and rewind stale EOD pointers.

Discovery does not call yfinance. New NASDAQ-listed MP symbols are registered and
validated on the later EOD history pull; empty yfinance results mark them UNSUPPORTED.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from lexis_markets.lake import LakeStore, PgClient
from lexis_markets.registry.nasdaq import NasdaqDirectory, fetch_nasdaq_directory
from lexis_markets.registry import register_discovered_entities, seed_default_stitch
from lexis_markets.registry.universe import EOD_ELIGIBLE_WHERE, extras_date
from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import get_logger

logger = get_logger("pipelines.entity_detect")


@dataclass
class MpSymbolSpan:
    symbol: str
    series_type: str
    first: date
    last: date


def scan_mp_window(lake: LakeStore, today: date | None = None) -> dict[str, MpSymbolSpan]:
    from lexis_markets.eod.ingest import fetch_mp_daily, mp_cutoff

    today = today or date.today()
    cutoff = mp_cutoff(today)
    spans: dict[str, MpSymbolSpan] = {}
    day = cutoff
    while day <= today - timedelta(days=1):
        raw = fetch_mp_daily(day, lake)
        if not raw.empty and "symbol" in raw.columns:
            raw = raw.copy()
            raw["symbol"] = raw["symbol"].astype(str).str.upper()
            raw["date"] = pd.to_datetime(raw["date"]).dt.date
            if "asset_type" not in raw.columns:
                groups = [(raw, "equity")]
            else:
                groups = [
                    (raw[raw["asset_type"] == asset], series_type)
                    for asset, series_type in (("Stock", "equity"), ("ETF", "etf"))
                ]
            for sub, series_type in groups:
                if sub.empty:
                    continue
                for sym, grp in sub.groupby("symbol"):
                    first = grp["date"].min()
                    last = grp["date"].max()
                    prev = spans.get(sym)
                    if prev is None:
                        spans[sym] = MpSymbolSpan(sym, series_type, first, last)
                    else:
                        spans[sym] = MpSymbolSpan(
                            sym,
                            series_type,
                            min(prev.first, first),
                            max(prev.last, last),
                        )
        day += timedelta(days=1)
    return spans


def known_equity_symbols(pg: PgClient) -> set[str]:
    rows = pg.fetchall(
        """
        SELECT UPPER(canonical_symbol) AS sym
        FROM series_meta
        WHERE asset_class IN ('equity', 'etf')
        """
    )
    return {r["sym"] for r in rows}


def detect_new_entities(
    lake: LakeStore,
    pg: PgClient,
    directory: NasdaqDirectory,
    *,
    cfg: MarketsConfig | None = None,
    today: date | None = None,
) -> list[dict]:
    """MP + NASDAQ discovery only. YF support is decided by the later history pull."""
    mp = scan_mp_window(lake, today=today)
    known = known_equity_symbols(pg)
    pending: list[tuple[str, MpSymbolSpan]] = []
    for sym, span in mp.items():
        if sym in known:
            continue
        if sym not in directory.symbols:
            continue
        pending.append((sym, span))

    cfg = cfg or MarketsConfig.from_env()
    limit = cfg.yf_backfill_limit()
    if limit and len(pending) > limit:
        sampled = cfg.sample_yf_targets([{"symbol": sym, "series_id": sym} for sym, _ in pending])
        keep = {t["symbol"] for t in sampled}
        pending = [(sym, span) for sym, span in pending if sym in keep]
        logger.info(
            "entity_detect: yf limit capped mp candidates %s -> %s",
            len(mp) - len(known),
            len(pending),
        )

    candidates: list[dict] = []
    for sym, span in pending:
        entry = directory.by_symbol[sym]
        series_type = "etf" if entry.etf or span.series_type == "etf" else span.series_type
        candidates.append(
            {
                "symbol": sym,
                "series_type": series_type,
                "exchange": entry.exchange,
                "first_seen": span.first,
                "last_seen": span.last,
                "mp_first": span.first,
                "mp_last": span.last,
                # Always attempt YF on the main EOD history pull; empty → UNSUPPORTED.
                "yf_backfill": True,
            }
        )
    return candidates


def run_eod_gap_scan(
    pg: PgClient,
    lake: LakeStore,
    directory: NasdaqDirectory,
    *,
    target_date: date | None = None,
    today: date | None = None,
) -> dict:
    from lexis_markets.eod.ingest import eod_target_date

    target = target_date or eod_target_date(today)
    mp = scan_mp_window(lake, today=today)
    rows = pg.fetchall(
        f"""
        SELECT series_id, UPPER(canonical_symbol) AS sym, extras
        FROM series_meta m
        WHERE m.status = 'ACTIVE'
          AND m.asset_class IN ('equity', 'etf')
          AND {EOD_ELIGIBLE_WHERE}
        """
    )
    rewound = 0
    mp_seen = 0
    yf_gap = 0
    rewinds: list[tuple[str, str]] = []
    for r in rows:
        sym = r["sym"]
        if sym not in directory.symbols:
            continue
        extras = r.get("extras")
        primary_last = extras_date(extras, "primary_last_seen")
        eod_through = extras_date(extras, "eod_filled_through")
        span = mp.get(sym)
        if span:
            mp_seen += 1
            if span.last > (eod_through or date.min) and (not eod_through or eod_through < target):
                rewind = primary_last or (
                    eod_through - timedelta(days=1) if eod_through else span.first - timedelta(days=1)
                )
                rewinds.append((rewind.isoformat(), r["series_id"]))
                rewound += 1
                continue
        if (not eod_through or eod_through < target) and (primary_last and primary_last < target):
            yf_gap += 1

    if rewinds:
        pg.executemany(
            """
            UPDATE series_meta SET
                extras = COALESCE(extras, '{}'::jsonb)
                    || jsonb_build_object('eod_filled_through', %s::text)
            WHERE series_id = %s
            """,
            rewinds,
        )

    logger.info(
        "gap_scan: mp_symbols=%s eligible=%s mp_seen=%s rewound=%s yf_stale=%s target=%s",
        len(mp),
        len(rows),
        mp_seen,
        rewound,
        yf_gap,
        target,
    )
    return {"rewound": rewound, "mp_seen": mp_seen, "yf_stale": yf_gap, "target": target.isoformat()}


def run_entity_detect(
    cfg: MarketsConfig,
    lake: LakeStore,
    pg: PgClient,
    *,
    directory: NasdaqDirectory | None = None,
    yf_gate=None,
    today: date | None = None,
) -> dict:
    from lexis_markets.eod.ingest import MP_FREE_DAYS

    directory = directory or fetch_nasdaq_directory()
    candidates = detect_new_entities(
        lake,
        pg,
        directory,
        cfg=cfg,
        today=today,
    )
    if not candidates:
        return {"registered": 0, "yf_backfill": 0, "mp_only": 0, "symbols": []}

    n = register_discovered_entities(pg, candidates)
    stitch_n = seed_default_stitch(pg)
    logger.info(
        "entity_detect: mp_window=%sd nasdaq=%s new=%s registered=%s stitch=%s (yf via eod history)",
        MP_FREE_DAYS,
        len(directory.symbols),
        len(candidates),
        n,
        stitch_n,
    )
    for c in candidates[:20]:
        logger.info(
            "  + %s %s exch=%s seen=%s..%s",
            c["symbol"],
            c["series_type"],
            c["exchange"],
            c["first_seen"],
            c["last_seen"],
        )
    if len(candidates) > 20:
        logger.info("  ... +%s more", len(candidates) - 20)
    return {
        "registered": n,
        "yf_backfill": n,
        "mp_only": 0,
        "symbols": [c["symbol"] for c in candidates],
    }
