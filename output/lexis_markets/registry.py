from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from lexis_markets.storage import PgClient
from lexis_markets.universe import LIVE_L2_WHERE

KAGGLE_SOURCES = ("jakewright", "jacksoncrow")
EOD_SOURCES = ("marketparquet", "yfinance")


def _series_id(asset_class: str, symbol: str) -> str:
    return f"{asset_class}:{symbol.upper()}"


def _span_days(first: date, last: date, calendar_id: str) -> int:
    if calendar_id == "fred_native":
        return (last - first).days + 1
    return int(np.busday_count(first, last)) + 1


def _stitched_days(dets: list[dict]) -> int:
    by_source = {d["source"]: int(d.get("unique_days") or d.get("rows") or 0) for d in dets}
    jw = by_source.get("jakewright", 0)
    jc = by_source.get("jacksoncrow", 0)
    fred = by_source.get("fred", 0)
    if fred:
        return fred
    if jw and jc:
        return jw + jc - min(jw, jc)
    return jw or jc or 0


def _quality(dets: list[dict], first: date, last: date, calendar_id: str) -> tuple[int, int, float]:
    span = _span_days(first, last, calendar_id)
    if span <= 0:
        return 0, 0, 1.0
    days = min(span, _stitched_days(dets))
    gap = max(0, span - days)
    by_source = {d["source"]: int(d.get("unique_days") or d.get("rows") or 0) for d in dets}
    disagreement = min(by_source.get("jakewright", 0), by_source.get("jacksoncrow", 0))
    return gap, disagreement, days / span


def seed_from_details(pg: PgClient, details: list[dict]) -> dict:
    meta_rows = []
    alias_rows = []
    details_by_sid: dict[str, list[dict]] = defaultdict(list)
    for d in details:
        if not d.get("symbol") or not d.get("first"):
            continue
        symbol = str(d["symbol"]).upper()
        source = d["source"]
        series_type = d.get("series_type") or ("macro" if source == "fred" else "equity")
        if source == "fred":
            asset_class, calendar_id = "macro", "fred_native"
            sid = _series_id("macro", symbol)
        elif series_type == "etf":
            asset_class, calendar_id = "etf", "nyse"
            sid = _series_id("etf", symbol)
        else:
            asset_class, calendar_id = "equity", "nyse"
            sid = _series_id("equity", symbol)
        first_seen = date.fromisoformat(d["first"])
        last_seen = date.fromisoformat(d["last"]) if d.get("last") else first_seen
        meta_rows.append((sid, symbol, asset_class, series_type, calendar_id, "ACTIVE", first_seen, last_seen))
        alias_rows.append((sid, source, symbol, first_seen, None))
        if asset_class in ("equity", "etf") and source == "jacksoncrow":
            alias_rows.append((sid, "yfinance", symbol, first_seen, None))
        details_by_sid[sid].append(d)

    by_id: dict[str, tuple] = {}
    for row in meta_rows:
        sid = row[0]
        if sid not in by_id:
            by_id[sid] = row
        else:
            prev = by_id[sid]
            by_id[sid] = (
                sid,
                prev[1],
                prev[2],
                prev[3],
                prev[4],
                prev[5],
                min(prev[6], row[6]),
                max(prev[7], row[7]),
            )

    quality_rows = []
    for sid, row in by_id.items():
        dets = details_by_sid[sid]
        gap, disagreement, score = _quality(dets, row[6], row[7], row[4])
        primary_dets = [d for d in dets if d["source"] in KAGGLE_SOURCES and d.get("last")]
        eod_dets = [d for d in dets if d["source"] in EOD_SOURCES and d.get("last")]
        extras_patch: dict[str, str] = {}
        if primary_dets:
            extras_patch["primary_last_seen"] = max(
                date.fromisoformat(d["last"]) for d in primary_dets
            ).isoformat()
        if eod_dets:
            extras_patch["eod_filled_through"] = max(
                date.fromisoformat(d["last"]) for d in eod_dets
            ).isoformat()
        quality_rows.append(
            (
                sid,
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                gap,
                disagreement,
                0,
                score,
                json.dumps(extras_patch),
            )
        )

    pg.executemany(
        """
        INSERT INTO series_meta
            (series_id, canonical_symbol, asset_class, series_type, calendar_id, status,
             first_seen, last_seen, gap_count, disagreement_count, suspicious_count, quality_score, extras)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (series_id) DO UPDATE SET
            first_seen = LEAST(series_meta.first_seen, EXCLUDED.first_seen),
            last_seen = GREATEST(series_meta.last_seen, EXCLUDED.last_seen),
            gap_count = EXCLUDED.gap_count,
            disagreement_count = EXCLUDED.disagreement_count,
            quality_score = EXCLUDED.quality_score,
            extras = COALESCE(series_meta.extras, '{}'::jsonb) || EXCLUDED.extras::jsonb
        """,
        quality_rows,
    )
    pg.executemany(
        """
        INSERT INTO symbol_aliases (series_id, source, source_symbol, valid_from, valid_to)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (series_id, source, source_symbol) DO NOTHING
        """,
        alias_rows,
    )
    gaps = [r[8] for r in quality_rows]
    return {
        "series": len(by_id),
        "aliases": len(alias_rows),
        "gap_nonzero": sum(1 for g in gaps if g > 0),
        "gap_total": sum(gaps),
    }


def ensure_eod_aliases(pg: PgClient) -> int:
    pg.execute(
        f"""
        INSERT INTO symbol_aliases (series_id, source, source_symbol, valid_from, valid_to)
        SELECT m.series_id, 'yfinance', m.canonical_symbol, m.first_seen, NULL
        FROM series_meta m
        WHERE m.asset_class IN ('equity', 'etf')
          AND ({LIVE_L2_WHERE.strip()} OR m.extras->>'discovered_via' = 'marketparquet')
        ON CONFLICT (series_id, source, source_symbol) DO NOTHING
        """
    )
    row = pg.fetchone("SELECT COUNT(*) AS n FROM symbol_aliases WHERE source = 'yfinance'")
    return int(row["n"]) if row else 0


def seed_default_stitch(pg: PgClient) -> int:
    aliases = pg.fetchall("SELECT series_id, source, source_symbol, valid_from, valid_to FROM symbol_aliases")
    priority = {s: i for i, s in enumerate(("jakewright", "jacksoncrow", "marketparquet", "yfinance", "fred"))}
    rows = []
    for a in aliases:
        role = "primary" if priority.get(a["source"], 99) == 0 else "cross_check"
        if a["source"] in ("marketparquet", "yfinance"):
            role = "fill"
        rows.append(
            (
                a["series_id"],
                a["source"],
                a["source_symbol"],
                a["valid_from"],
                a["valid_to"],
                role,
                None,
            )
        )
    pg.execute("DELETE FROM stitch_segments")
    pg.executemany(
        """
        INSERT INTO stitch_segments
            (series_id, source, source_symbol, valid_from, valid_to, role, method)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )
    return len(rows)


def register_discovered_entities(pg: PgClient, candidates: list[dict]) -> int:
    if not candidates:
        return 0
    meta_rows = []
    alias_rows = []
    for c in candidates:
        symbol = str(c["symbol"]).upper()
        series_type = c.get("series_type") or "equity"
        asset_class = "etf" if series_type == "etf" else "equity"
        sid = _series_id(asset_class, symbol)
        first_seen = c["first_seen"]
        last_seen = c["last_seen"]
        primary_last = first_seen - timedelta(days=1)
        extras = {
            "discovered_via": "marketparquet",
            "listing_exchange": c.get("exchange") or "",
            "primary_last_seen": primary_last.isoformat(),
        }
        if not c.get("yf_backfill"):
            extras["eod_filled_through"] = c["mp_last"].isoformat()
        meta_rows.append(
            (
                sid,
                symbol,
                asset_class,
                series_type,
                "nyse",
                "ACTIVE",
                first_seen,
                last_seen,
                0,
                0,
                0,
                1.0,
                json.dumps(extras),
            )
        )
        for source in ("marketparquet", "yfinance"):
            alias_rows.append((sid, source, symbol, first_seen, None))

    pg.executemany(
        """
        INSERT INTO series_meta
            (series_id, canonical_symbol, asset_class, series_type, calendar_id, status,
             first_seen, last_seen, gap_count, disagreement_count, suspicious_count, quality_score, extras)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (series_id) DO NOTHING
        """,
        meta_rows,
    )
    pg.executemany(
        """
        INSERT INTO symbol_aliases (series_id, source, source_symbol, valid_from, valid_to)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (series_id, source, source_symbol) DO NOTHING
        """,
        alias_rows,
    )
    return len(candidates)


def patch_eod_registry(pg: PgClient, details: list[dict]) -> dict:
    ok = [d for d in details if int(d.get("rows") or 0) > 0 and d.get("last")]
    if not ok:
        return {"updated": 0}

    rows: list[tuple[str, date, date, date | None]] = []
    need_extras: set[str] = set()
    for d in ok:
        sid = d.get("series_id")
        if not sid:
            symbol = str(d["symbol"]).upper()
            series_type = d.get("series_type") or "equity"
            asset = "etf" if series_type == "etf" else "equity"
            sid = f"{asset}:{symbol}"
        new_first = date.fromisoformat(d["first"])
        new_last = date.fromisoformat(d["last"])
        primary_last = d.get("primary_last")
        if primary_last:
            pl = date.fromisoformat(primary_last) if isinstance(primary_last, str) else primary_last
            rows.append((sid, new_first, new_last, pl))
        else:
            need_extras.add(sid)
            rows.append((sid, new_first, new_last, None))

    extras_by_sid: dict[str, dict] = {}
    if need_extras:
        for r in pg.fetchall(
            "SELECT series_id, extras FROM series_meta WHERE series_id = ANY(%s)",
            (list(need_extras),),
        ):
            extras = r.get("extras") or {}
            if isinstance(extras, str):
                extras = json.loads(extras)
            extras_by_sid[r["series_id"]] = extras

    fill_through: list[tuple] = []
    last_only: list[tuple] = []
    for sid, new_first, new_last, primary_last in rows:
        if primary_last is None:
            raw = (extras_by_sid.get(sid) or {}).get("primary_last_seen")
            primary_last = date.fromisoformat(raw) if raw else None
        if primary_last is None or new_first <= primary_last + timedelta(days=7):
            fill_through.append((new_last, new_last.isoformat(), sid))
        else:
            last_only.append((new_last, sid))

    if fill_through:
        pg.executemany(
            """
            UPDATE series_meta SET
                last_seen = GREATEST(last_seen, %s::date),
                extras = COALESCE(extras, '{}'::jsonb)
                    || jsonb_build_object('eod_filled_through', %s::text)
            WHERE series_id = %s
            """,
            fill_through,
        )
    if last_only:
        pg.executemany(
            """
            UPDATE series_meta SET last_seen = GREATEST(last_seen, %s::date)
            WHERE series_id = %s
            """,
            last_only,
        )
    return {"updated": len(ok)}
