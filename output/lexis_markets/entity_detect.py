from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from lexis_markets.config import MarketsConfig
from lexis_markets.nasdaq_symbols import NasdaqDirectory, fetch_nasdaq_directory
from lexis_markets.registry import register_discovered_entities, seed_default_stitch
from lexis_markets.storage import LakeStore, PgClient
from lexis_markets.universe import EOD_ELIGIBLE_WHERE, parse_extras


@dataclass
class MpSymbolSpan:
    symbol: str
    series_type: str
    first: date
    last: date


def scan_mp_window(lake: LakeStore, today: date | None = None) -> dict[str, MpSymbolSpan]:
    from lexis_markets.eod import MP_FREE_DAYS, fetch_mp_daily, mp_cutoff

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
            for asset, series_type in (("Stock", "equity"), ("ETF", "etf")):
                sub = raw[raw["asset_type"] == asset]
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


def probe_yf_history(symbol: str, min_days: int) -> tuple[date, date] | None:
    raw = yf.download(
        symbol,
        period="max",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw.empty or len(raw) < min_days:
        return None
    idx = pd.to_datetime(raw.index).date
    return idx.min(), idx.max()


def detect_new_entities(
    lake: LakeStore,
    pg: PgClient,
    directory: NasdaqDirectory,
    *,
    min_history_days: int,
) -> list[dict]:
    mp = scan_mp_window(lake)
    known = known_equity_symbols(pg)
    candidates: list[dict] = []
    for sym, span in mp.items():
        if sym in known:
            continue
        if sym not in directory.symbols:
            continue
        entry = directory.by_symbol[sym]
        series_type = "etf" if entry.etf or span.series_type == "etf" else span.series_type
        hist = probe_yf_history(sym, min_history_days)
        if hist:
            first_seen, last_seen = hist[0], max(hist[1], span.last)
            backfill = True
        else:
            first_seen, last_seen = span.first, span.last
            backfill = False
        candidates.append(
            {
                "symbol": sym,
                "series_type": series_type,
                "exchange": entry.exchange,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "mp_first": span.first,
                "mp_last": span.last,
                "yf_backfill": backfill,
            }
        )
    return candidates


def _extras_date(extras, key: str) -> date | None:
    raw = parse_extras(extras).get(key)
    return date.fromisoformat(raw) if raw else None


def run_eod_gap_scan(
    pg: PgClient,
    lake: LakeStore,
    directory: NasdaqDirectory,
) -> dict:
    from lexis_markets.eod import eod_target_date

    target = eod_target_date()
    mp = scan_mp_window(lake)
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
        primary_last = _extras_date(extras, "primary_last_seen")
        eod_through = _extras_date(extras, "eod_filled_through")
        span = mp.get(sym)
        if span:
            mp_seen += 1
            if span.last > (eod_through or date.min) and (not eod_through or eod_through < target):
                rewind = primary_last or (eod_through - timedelta(days=1) if eod_through else span.first - timedelta(days=1))
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

    print(
        f"gap_scan: mp_symbols={len(mp)} eligible={len(rows)} mp_seen={mp_seen} "
        f"rewound={rewound} yf_stale={yf_gap} target={target}",
        flush=True,
    )
    return {"rewound": rewound, "mp_seen": mp_seen, "yf_stale": yf_gap, "target": target.isoformat()}


def run_entity_detect(
    cfg: MarketsConfig,
    lake: LakeStore,
    pg: PgClient,
    *,
    directory: NasdaqDirectory | None = None,
) -> dict:
    from lexis_markets.eod import MP_FREE_DAYS

    directory = directory or fetch_nasdaq_directory()
    min_days = cfg.discover_min_history_days
    candidates = detect_new_entities(lake, pg, directory, min_history_days=min_days)
    if not candidates:
        return {"registered": 0, "yf_backfill": 0, "mp_only": 0, "symbols": []}

    n = register_discovered_entities(pg, candidates)
    stitch_n = seed_default_stitch(pg)
    yf_n = sum(1 for c in candidates if c["yf_backfill"])
    print(
        f"entity_detect: mp_window={MP_FREE_DAYS}d nasdaq={len(directory.symbols)} "
        f"new={len(candidates)} registered={n} stitch={stitch_n} yf_backfill={yf_n}",
        flush=True,
    )
    for c in candidates[:20]:
        print(
            f"  + {c['symbol']} {c['series_type']} exch={c['exchange']} "
            f"seen={c['first_seen']}..{c['last_seen']} yf_backfill={c['yf_backfill']}",
            flush=True,
        )
    if len(candidates) > 20:
        print(f"  ... +{len(candidates) - 20} more", flush=True)
    return {
        "registered": n,
        "yf_backfill": yf_n,
        "mp_only": len(candidates) - yf_n,
        "symbols": [c["symbol"] for c in candidates],
    }
