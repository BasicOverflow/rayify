"""Series universe filters for registry resolution and Serve queries."""
from __future__ import annotations

from datetime import date

from lexis_markets.registry.universe import parse_extras

SERIES_STATUSES = frozenset({"ACTIVE", "DELISTED", "UNSUPPORTED", "BAD_DATA"})

EFFECTIVE_LAST_SEEN_SQL = """
COALESCE(
    CASE
        WHEN NULLIF(m.extras->>'eod_filled_through', '') IS NOT NULL
        THEN GREATEST(m.last_seen, (m.extras->>'eod_filled_through')::date)
        ELSE m.last_seen
    END,
    m.last_seen
)
"""


def normalize_statuses(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    out = [s.strip().upper() for s in raw if s and s.strip()]
    if not out:
        return None
    bad = set(out) - SERIES_STATUSES
    if bad:
        raise ValueError(f"invalid statuses: {sorted(bad)}")
    return out


def effective_last_seen(row: dict) -> date | None:
    last = row.get("last_seen")
    if last is None:
        return None
    extras = parse_extras(row.get("extras"))
    raw = extras.get("eod_filled_through")
    if raw:
        last = max(last, date.fromisoformat(raw))
    return last


def covers_through_end(row: dict, end: date) -> bool:
    last = effective_last_seen(row)
    return last is not None and last >= end


def append_series_filters(sql: str, params: list, spec: dict) -> tuple[str, list]:
    statuses = normalize_statuses(spec.get("statuses"))
    if statuses:
        sql += " AND m.status = ANY(%s)"
        params.append(statuses)

    max_gap = spec.get("max_gap_count")
    if max_gap is not None:
        sql += " AND m.gap_count <= %s"
        params.append(int(max_gap))

    max_sus = spec.get("max_suspicious_count")
    if max_sus is not None:
        sql += " AND m.suspicious_count <= %s"
        params.append(int(max_sus))

    if spec.get("include_partial_coverage", True) is False:
        end_raw = spec.get("end")
        if end_raw:
            sql += f" AND {EFFECTIVE_LAST_SEEN_SQL} >= %s::date"
            params.append(end_raw)

    return sql, params
