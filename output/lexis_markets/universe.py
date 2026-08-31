from __future__ import annotations

import json

from lexis_markets.storage import PgClient

US_LISTING_CODES = frozenset({"Q", "N", "P", "Z", "A"})
EXCHANGE_MAP = {"Q": "NASDAQ", "N": "NYSE", "P": "NYSE Arca", "Z": "BATS", "A": "AMEX"}

LIVE_L2_WHERE = """
    EXISTS (
        SELECT 1 FROM symbol_aliases jc
        WHERE jc.series_id = m.series_id AND jc.source = 'jacksoncrow'
    )
    AND (
        UPPER(COALESCE(m.extras->>'listing_exchange', '')) IN ('Q', 'N', 'P', 'Z', 'A')
        OR COALESCE(m.extras->>'listing_exchange', '') = ''
    )
"""

YF_SKIP_WHERE = "(m.extras->>'yf_skip') IS DISTINCT FROM 'true'"

EOD_ELIGIBLE_WHERE = f"""
    (
        {LIVE_L2_WHERE.strip()}
        OR (
            COALESCE(m.extras->>'discovered_via', '') = 'marketparquet'
            AND (
                COALESCE(m.extras->>'listing_exchange', '') = ''
                OR UPPER(COALESCE(m.extras->>'listing_exchange', '')) IN ('Q', 'N', 'P', 'Z', 'A')
            )
        )
    )
    AND {YF_SKIP_WHERE}
"""


def parse_extras(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def us_listed_extras(extras: dict) -> bool:
    code = str((extras or {}).get("listing_exchange") or "").strip().upper()
    if not code:
        return True
    return code in US_LISTING_CODES


def live_l2_eligible(*, in_jacksoncrow: bool, extras: dict) -> bool:
    return in_jacksoncrow and us_listed_extras(extras)


def enrich_universe_fields(row: dict) -> dict:
    extras = parse_extras(row.get("extras"))
    in_jc = bool(row.get("in_jacksoncrow"))
    return {
        **row,
        "extras": extras,
        "in_jacksoncrow": in_jc,
        "us_listed": us_listed_extras(extras),
        "live_l2": live_l2_eligible(in_jacksoncrow=in_jc, extras=extras),
    }


def mark_yf_skip(pg: PgClient, series_ids: list[str], reason: str, *, delist: bool = False) -> int:
    if not series_ids:
        return 0
    status_sql = ", status = 'DELISTED'" if delist else ""
    row = pg.fetchone(
        f"""
        WITH updated AS (
            UPDATE series_meta SET
                extras = COALESCE(extras, '{{}}'::jsonb)
                    || jsonb_build_object(
                        'yf_skip', 'true',
                        'yf_skip_reason', %s::text,
                        'status_source', 'yf_gate',
                        'status_note', %s::text
                    )
                {status_sql}
            WHERE series_id = ANY(%s)
              AND (extras->>'yf_skip') IS DISTINCT FROM 'true'
            RETURNING 1
        )
        SELECT COUNT(*) AS n FROM updated
        """,
        (reason, reason, series_ids),
    )
    return int(row["n"]) if row else 0


def apply_nasdaq_yf_gate(pg: PgClient, targets: list[dict], listed: set[str]) -> tuple[list[dict], int]:
    ok: list[dict] = []
    skip_ids: list[str] = []
    for t in targets:
        if t["symbol"] in listed:
            ok.append(t)
        else:
            skip_ids.append(t["series_id"])
    n = mark_yf_skip(pg, skip_ids, "not_nasdaq_listed", delist=True)
    return ok, n


def patch_yf_skip_failures(pg: PgClient, details: list[dict]) -> int:
    ids: list[str] = []
    for d in details:
        if d.get("source") != "eod_failed":
            continue
        sid = d.get("series_id")
        if not sid:
            continue
        err = str(d.get("error") or "").lower()
        if "empty" in err or "delisted" in err:
            ids.append(sid)
    return mark_yf_skip(pg, ids, "yfinance_empty")


def sync_live_universe(pg: PgClient) -> dict:
    """ACTIVE jacksoncrow US stay live; jakewright-only and non-US → UNSUPPORTED."""
    jw_only = pg.fetchone(
        f"""
        WITH updated AS (
            UPDATE series_meta m SET
                status = 'UNSUPPORTED',
                extras = COALESCE(m.extras, '{{}}'::jsonb)
                    || jsonb_build_object('status_source', 'universe', 'status_note', 'jakewright_only')
            WHERE m.asset_class IN ('equity', 'etf')
              AND m.status = 'ACTIVE'
              AND NOT EXISTS (
                  SELECT 1 FROM symbol_aliases a
                  WHERE a.series_id = m.series_id AND a.source = 'jacksoncrow'
              )
            RETURNING 1
        )
        SELECT COUNT(*) AS n FROM updated
        """
    )
    non_us = pg.fetchone(
        f"""
        WITH updated AS (
            UPDATE series_meta m SET
                status = 'UNSUPPORTED',
                extras = COALESCE(m.extras, '{{}}'::jsonb)
                    || jsonb_build_object('status_source', 'universe', 'status_note', 'non_us_listing')
            WHERE m.asset_class IN ('equity', 'etf')
              AND m.status = 'ACTIVE'
              AND EXISTS (
                  SELECT 1 FROM symbol_aliases a
                  WHERE a.series_id = m.series_id AND a.source = 'jacksoncrow'
              )
              AND COALESCE(m.extras->>'listing_exchange', '') <> ''
              AND UPPER(COALESCE(m.extras->>'listing_exchange', '')) NOT IN ('Q', 'N', 'P', 'Z', 'A')
            RETURNING 1
        )
        SELECT COUNT(*) AS n FROM updated
        """
    )
    return {
        "jakewright_only": int(jw_only["n"]) if jw_only else 0,
        "non_us_listing": int(non_us["n"]) if non_us else 0,
    }
