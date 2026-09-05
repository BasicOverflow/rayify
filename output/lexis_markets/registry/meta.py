"""Postgres L2 registry: series metadata, aliases, cache spans, stitch segments.

``ensure_schema`` creates tables on first connect. ``seed_default_stitch`` copies
``symbol_aliases`` into ``stitch_segments`` so ``serve.stitch`` can resolve sources
per series without joining aliases at read time.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from lexis_markets.lake import PgClient, utcnow
from lexis_markets.config import SOURCE_PRIORITY, cache_object_key, series_id_to_cache_key
from lexis_markets.registry.universe import EOD_ELIGIBLE_WHERE, extras_date

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calendars (
    calendar_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    timezone TEXT NOT NULL,
    rules JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS series_meta (
    series_id TEXT PRIMARY KEY,
    canonical_symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    series_type TEXT,
    exchange TEXT,
    country TEXT,
    currency TEXT DEFAULT 'USD',
    calendar_id TEXT REFERENCES calendars(calendar_id),
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    first_seen DATE,
    last_seen DATE,
    gap_count INT NOT NULL DEFAULT 0,
    disagreement_count INT NOT NULL DEFAULT 0,
    suspicious_count INT NOT NULL DEFAULT 0,
    quality_score DOUBLE PRECISION,
    extras JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS symbol_aliases (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    source TEXT NOT NULL,
    source_symbol TEXT NOT NULL,
    valid_from DATE,
    valid_to DATE,
    UNIQUE (series_id, source, source_symbol)
);

CREATE TABLE IF NOT EXISTS series_links (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    link_type TEXT NOT NULL,
    related_series_id TEXT,
    effective_date DATE,
    note TEXT
);

CREATE TABLE IF NOT EXISTS stitch_segments (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    source TEXT NOT NULL,
    source_symbol TEXT NOT NULL,
    valid_from DATE,
    valid_to DATE
);

CREATE TABLE IF NOT EXISTS dataset_jobs (
    job_id UUID PRIMARY KEY,
    status TEXT NOT NULL,
    spec JSONB NOT NULL,
    s3_prefix TEXT,
    series_count INT,
    row_count INT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS l1_month_manifest (
    year INT NOT NULL,
    month INT NOT NULL,
    compacted_key TEXT NOT NULL,
    row_count BIGINT,
    symbol_count INT,
    compacted_at TIMESTAMPTZ,
    PRIMARY KEY (year, month)
);

CREATE TABLE IF NOT EXISTS symbol_month_coverage (
    source TEXT NOT NULL,
    source_symbol TEXT NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    PRIMARY KEY (source, source_symbol, year, month)
);

CREATE TABLE IF NOT EXISTS series_cache_meta (
    series_id TEXT PRIMARY KEY REFERENCES series_meta(series_id),
    cache_key TEXT NOT NULL,
    row_count BIGINT,
    l1_fp TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS series_cache_span (
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    span_start DATE NOT NULL,
    span_end DATE NOT NULL,
    PRIMARY KEY (series_id, span_start)
);

INSERT INTO calendars (calendar_id, name, timezone, rules)
VALUES
    ('nyse', 'NYSE equity sessions', 'America/New_York', '{"kind":"equity_session"}'::jsonb),
    ('fred_native', 'FRED native frequency', 'America/New_York', '{"kind":"fred_native"}'::jsonb)
ON CONFLICT (calendar_id) DO NOTHING;
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_stitch_segments_lookup ON stitch_segments (series_id);
CREATE INDEX IF NOT EXISTS idx_series_meta_asset_status ON series_meta (asset_class, status);
CREATE INDEX IF NOT EXISTS idx_symbol_aliases_upper_symbol ON symbol_aliases (UPPER(source_symbol));
CREATE INDEX IF NOT EXISTS idx_series_cache_span_series ON series_cache_span (series_id);
"""

MIGRATION_SQL = """
DROP TABLE IF EXISTS stitch_decisions;
DROP TABLE IF EXISTS fred_ingest_done;
DROP TABLE IF EXISTS l3_month_manifest;
ALTER TABLE stitch_segments DROP COLUMN IF EXISTS role;
ALTER TABLE stitch_segments DROP COLUMN IF EXISTS method;
ALTER TABLE stitch_segments DROP COLUMN IF EXISTS universe_version;
ALTER TABLE stitch_segments DROP COLUMN IF EXISTS rule_version;
ALTER TABLE dataset_jobs DROP COLUMN IF EXISTS spec_hash;
ALTER TABLE dataset_jobs DROP COLUMN IF EXISTS universe_version;
ALTER TABLE dataset_jobs DROP COLUMN IF EXISTS rule_version;
"""

KAGGLE_SOURCES = ("jakewright", "jacksoncrow")
EOD_SOURCES = ("marketparquet", "yfinance")


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date


def apply_schema(pg: PgClient) -> None:
    def _go():
        with pg.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
                cur.execute(MIGRATION_SQL)
                cur.execute(INDEXES_SQL)

    pg.run(_go)


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
                sid, prev[1], prev[2], prev[3], prev[4], prev[5],
                min(prev[6], row[6]), max(prev[7], row[7]),
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
        fred_dets = [d for d in dets if d["source"] == "fred" and d.get("vintage_through")]
        if fred_dets:
            extras_patch["fred_vintage_through"] = max(
                date.fromisoformat(d["vintage_through"]) for d in fred_dets
            ).isoformat()
        quality_rows.append(
            (sid, row[1], row[2], row[3], row[4], row[5], row[6], row[7],
             gap, disagreement, 0, score, json.dumps(extras_patch))
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
    """Ensure fill-source aliases exist from ``primary_last_seen + 1`` for EOD-eligible series."""
    n = 0
    for source in EOD_SOURCES:
        pg.execute(
            f"""
            INSERT INTO symbol_aliases (series_id, source, source_symbol, valid_from, valid_to)
            SELECT m.series_id, %s, m.canonical_symbol,
                   COALESCE((m.extras->>'primary_last_seen')::date + 1, m.first_seen),
                   NULL
            FROM series_meta m
            WHERE m.asset_class IN ('equity', 'etf')
              AND ({EOD_ELIGIBLE_WHERE.strip()})
            ON CONFLICT (series_id, source, source_symbol) DO UPDATE SET
              valid_from = COALESCE(EXCLUDED.valid_from, symbol_aliases.valid_from)
            """,
            (source,),
        )
        row = pg.fetchone(
            "SELECT COUNT(*) AS n FROM symbol_aliases WHERE source = %s",
            (source,),
        )
        n += int(row["n"]) if row else 0
    return n


def seed_default_stitch(pg: PgClient) -> int:
    """Copy ``symbol_aliases`` rows into ``stitch_segments`` (full refresh)."""
    aliases = pg.fetchall(
        "SELECT series_id, source, source_symbol, valid_from, valid_to FROM symbol_aliases"
    )
    rows = [
        (a["series_id"], a["source"], a["source_symbol"], a["valid_from"], a["valid_to"])
        for a in aliases
    ]
    pg.execute("DELETE FROM stitch_segments")
    pg.executemany(
        """
        INSERT INTO stitch_segments (series_id, source, source_symbol, valid_from, valid_to)
        VALUES (%s, %s, %s, %s, %s)
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
            (sid, symbol, asset_class, series_type, "nyse", "ACTIVE",
             first_seen, last_seen, 0, 0, 0, 1.0, json.dumps(extras))
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
    rows: list[tuple] = []
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
            "UPDATE series_meta SET last_seen = GREATEST(last_seen, %s::date) WHERE series_id = %s",
            last_only,
        )
    return {"updated": len(ok)}


def resolve_fred_backfill_jobs(
    pg: PgClient,
    cfg,
    *,
    vintage_end: date,
    fred_gate,
    series_ids: list[str] | None = None,
    force: bool = False,
) -> list[dict]:
    """Build one Ray job per series for the ALFRED vintage gap through ``vintage_end``."""
    from lexis_markets.fred.alfred import clamp_vintage_start
    from lexis_markets.fred.client import resolve_fred_series_ids

    floor = cfg.fred_vintage_start
    symbols = sorted({s.upper() for s in (series_ids or resolve_fred_series_ids(cfg, fred_gate))})
    if not symbols:
        return []

    meta_by_symbol: dict[str, dict] = {}
    for row in pg.fetchall(
        """
        SELECT m.series_id, a.source_symbol, m.extras, m.first_seen
        FROM series_meta m
        JOIN symbol_aliases a ON a.series_id = m.series_id AND a.source = 'fred'
        WHERE m.asset_class = 'macro'
        """
    ):
        sym = str(row["source_symbol"]).upper()
        extras = row.get("extras") or {}
        if isinstance(extras, str):
            extras = json.loads(extras)
        meta_by_symbol[sym] = {
            "series_id": row["series_id"],
            "extras": extras,
            "first_seen": row.get("first_seen"),
        }

    jobs: list[dict] = []
    for symbol in symbols:
        meta = meta_by_symbol.get(symbol) or {}
        series_id = meta.get("series_id") or f"macro:{symbol}"
        extras = meta.get("extras") or {}
        vintage_through = None if force else extras_date(extras, "fred_vintage_through")
        full_backfill = vintage_through is None
        if full_backfill:
            first_seen = meta.get("first_seen")
            raw_start = first_seen if isinstance(first_seen, date) else floor
            vintage_start = clamp_vintage_start(raw_start, floor=floor)
        else:
            vintage_start = vintage_through + timedelta(days=1)
        if vintage_start > vintage_end:
            continue
        jobs.append(
            {
                "series_id": series_id,
                "symbol": symbol,
                "vintage_start": vintage_start.isoformat(),
                "vintage_end": vintage_end.isoformat(),
                "full_backfill": full_backfill,
            }
        )
    return jobs


def resolve_fred_vintage_jobs(pg: PgClient, cfg, target: date, fred_gate) -> list[dict]:
    """Daily macro EOD: one observations query per series for the vintage gap."""
    return resolve_fred_backfill_jobs(
        pg, cfg, vintage_end=target, fred_gate=fred_gate
    )


def patch_macro_eod_registry(pg: PgClient, details: list[dict]) -> dict:
    ok = [d for d in details if d.get("vintage_through")]
    if not ok:
        return {"updated": 0}
    rows: list[tuple] = []
    for d in ok:
        sid = d.get("series_id")
        if not sid:
            symbol = str(d["symbol"]).upper()
            sid = f"macro:{symbol}"
        vintage_through = date.fromisoformat(d["vintage_through"])
        last_seen = date.fromisoformat(d["last"]) if d.get("last") else vintage_through
        rows.append((last_seen, vintage_through.isoformat(), sid))
    pg.executemany(
        """
        UPDATE series_meta SET
            last_seen = GREATEST(last_seen, %s::date),
            extras = COALESCE(extras, '{}'::jsonb)
                || jsonb_build_object('fred_vintage_through', %s::text)
        WHERE series_id = %s
        """,
        rows,
    )
    return {"updated": len(rows)}


def _merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        ls, le = merged[-1]
        if start <= le + timedelta(days=1):
            merged[-1] = (ls, max(le, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_interval(
    cached: list[tuple[date, date]], need_start: date, need_end: date
) -> list[DateRange]:
    if need_start > need_end:
        return []
    if not cached:
        return [DateRange(need_start, need_end)]
    missing: list[tuple[date, date]] = [(need_start, need_end)]
    for cs, ce in cached:
        next_missing: list[tuple[date, date]] = []
        for ms, me in missing:
            if ce < ms or cs > me:
                next_missing.append((ms, me))
                continue
            if ms < cs:
                next_missing.append((ms, cs - timedelta(days=1)))
            if me > ce:
                next_missing.append((ce + timedelta(days=1), me))
        missing = next_missing
    return [DateRange(s, e) for s, e in missing if s <= e]


def uncached_ranges(pg: PgClient, series_id: str, req_start: date, req_end: date) -> list[DateRange]:
    rows = pg.fetchall(
        """
        SELECT span_start, span_end FROM series_cache_span
        WHERE series_id = %s AND span_end >= %s AND span_start <= %s
        ORDER BY span_start
        """,
        (series_id, req_start, req_end),
    )
    cached = [(r["span_start"], r["span_end"]) for r in rows]
    cached = _merge_intervals(cached)
    return _subtract_interval(cached, req_start, req_end)


def merge_spans(pg: PgClient, series_id: str, span_start: date, span_end: date) -> None:
    rows = pg.fetchall(
        "SELECT span_start, span_end FROM series_cache_span WHERE series_id = %s ORDER BY span_start",
        (series_id,),
    )
    intervals = [(r["span_start"], r["span_end"]) for r in rows]
    intervals.append((span_start, span_end))
    merged = _merge_intervals(intervals)
    pg.execute("DELETE FROM series_cache_span WHERE series_id = %s", (series_id,))
    pg.executemany(
        "INSERT INTO series_cache_span (series_id, span_start, span_end) VALUES (%s, %s, %s)",
        [(series_id, s, e) for s, e in merged],
    )


def update_cache_meta(pg: PgClient, series_id: str, row_count: int, l1_fp: str = "") -> None:
    pg.execute(
        """
        INSERT INTO series_cache_meta (series_id, cache_key, row_count, l1_fp, updated_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (series_id) DO UPDATE SET
            row_count = EXCLUDED.row_count,
            l1_fp = EXCLUDED.l1_fp,
            updated_at = EXCLUDED.updated_at
        """,
        (series_id, series_id_to_cache_key(series_id), row_count, l1_fp, utcnow()),
    )


def _universe_where(spec: dict) -> tuple[str, list]:
    """Shared WHERE clause builder for ``resolve_series_ids`` / ``list_series_meta``."""
    from lexis_markets.registry.filters import append_series_filters, normalize_statuses

    normalize_statuses(spec.get("statuses"))
    sql = ""
    params: list = []

    if spec.get("series_ids"):
        sql += " AND m.series_id = ANY(%s)"
        params.append(list(spec["series_ids"]))
    elif spec.get("symbols"):
        sql += """
            AND EXISTS (
                SELECT 1 FROM symbol_aliases a
                WHERE a.series_id = m.series_id
                  AND UPPER(a.source_symbol) = ANY(%s)
            )
        """
        params.append([s.upper() for s in spec["symbols"]])

    if spec.get("asset_classes"):
        sql += " AND m.asset_class = ANY(%s)"
        params.append(spec["asset_classes"])

    return append_series_filters(sql, params, spec)


def resolve_series_ids(pg: PgClient, spec: dict) -> list[str]:
    where, params = _universe_where(spec)
    rows = pg.fetchall(
        f"SELECT m.series_id FROM series_meta m WHERE 1=1{where}",
        params or None,
    )
    return [r["series_id"] for r in rows]


def list_series_meta(pg: PgClient, spec: dict) -> list[dict]:
    """Return registry rows matching the same universe filters as ``resolve_series_ids``."""
    where, params = _universe_where(spec)
    rows = pg.fetchall(
        f"""
        SELECT m.series_id, m.canonical_symbol, m.asset_class, m.status,
               m.first_seen, m.last_seen, m.gap_count, m.suspicious_count,
               m.disagreement_count, m.quality_score, m.calendar_id, m.extras
        FROM series_meta m
        WHERE 1=1{where}
        ORDER BY m.asset_class, m.canonical_symbol, m.series_id
        """,
        params or None,
    )
    return list(rows or [])
