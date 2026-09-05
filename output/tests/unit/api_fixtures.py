"""Deterministic past-dated fake series universe for Serve API unit tests."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any


# Fixed calendar anchors — all in the past so tests stay deterministic.
ERA_START = date(1990, 1, 2)
ERA_MID = date(2010, 6, 15)
ERA_END = date(2020, 12, 31)
RECENT_END = date(2020, 6, 30)  # ends before ERA_END → partial vs full coverage


def _sid(asset_class: str, symbol: str) -> str:
    return f"{asset_class}:{symbol}"


def build_api_test_universe(n: int = 100) -> list[dict[str, Any]]:
    """Build ``n`` series (capped 100) across equity / etf / macro with varied filters."""
    n = min(max(n, 10), 100)
    n_eq = max(1, n * 40 // 100)
    n_etf = max(1, n * 30 // 100)
    n_macro = n - n_eq - n_etf
    rows: list[dict[str, Any]] = []

    for i in range(n_eq):
        sym = f"E{i:03d}"
        status = "ACTIVE" if i % 10 else ("DELISTED" if i % 20 == 0 else "ACTIVE")
        if i % 17 == 0:
            status = "UNSUPPORTED"
        last = ERA_END if i % 5 else RECENT_END
        rows.append(
            {
                "series_id": _sid("equity", sym),
                "canonical_symbol": sym,
                "asset_class": "equity",
                "status": status,
                "first_seen": ERA_START + timedelta(days=i),
                "last_seen": last,
                "gap_count": (i * 7) % 50,
                "suspicious_count": (i * 3) % 20,
                "extras": {
                    "eod_filled_through": last.isoformat() if status == "ACTIVE" else None,
                    "primary_last_seen": ERA_MID.isoformat(),
                },
                "aliases": [sym, f"{sym}.US"],
            }
        )

    for i in range(n_etf):
        sym = f"F{i:03d}"
        status = "ACTIVE" if i % 8 else "DELISTED"
        last = ERA_END if i % 4 else RECENT_END
        rows.append(
            {
                "series_id": _sid("etf", sym),
                "canonical_symbol": sym,
                "asset_class": "etf",
                "status": status,
                "first_seen": date(1995, 1, 3) + timedelta(days=i * 3),
                "last_seen": last,
                "gap_count": (i * 11) % 40,
                "suspicious_count": (i * 5) % 15,
                "extras": {
                    "eod_filled_through": last.isoformat() if status == "ACTIVE" else None,
                    "primary_last_seen": "2015-01-01",
                },
                "aliases": [sym],
            }
        )

    macros = [
        "DGS10",
        "UNRATE",
        "CPIAUCSL",
        "INDPRO",
        "VIXCLS",
        "DEXUSEU",
        "PAYEMS",
        "GDPC1",
        "M2SL",
        "FEDFUNDS",
    ]
    for i in range(n_macro):
        sym = macros[i % len(macros)] if i < len(macros) else f"M{i:03d}"
        # Unique series_id even when recycling FRED names.
        sid_sym = sym if i < len(macros) else sym
        series_id = _sid("macro", sid_sym if i < len(macros) else f"M{i:03d}")
        last = ERA_END if i % 3 else date(2020, 7, 1)
        rows.append(
            {
                "series_id": series_id,
                "canonical_symbol": series_id.split(":", 1)[1],
                "asset_class": "macro",
                "status": "ACTIVE",
                "first_seen": date(1962, 1, 2) if i % 2 == 0 else date(1999, 1, 4),
                "last_seen": last,
                "gap_count": (i * 13) % 100,
                "suspicious_count": 0 if i % 2 == 0 else (i % 10),
                "extras": {},
                "aliases": [series_id.split(":", 1)[1]],
            }
        )

    assert len(rows) == n
    return rows


def effective_last(row: dict) -> date:
    last = row["last_seen"]
    raw = (row.get("extras") or {}).get("eod_filled_through")
    if raw:
        return max(last, date.fromisoformat(raw))
    return last


class FilterPg:
    """In-memory Postgres stand-in that applies ``resolve_series_ids`` SQL intent."""

    def __init__(self, universe: list[dict[str, Any]]):
        self.universe = list(universe)
        self.last_sql: str | None = None
        self.last_params: list | None = None

    def fetchall(self, sql: str, params=None) -> list[dict]:
        self.last_sql = sql
        params = list(params or [])
        rows = list(self.universe)
        pi = 0

        if "m.series_id = ANY(%s)" in sql:
            want = set(params[pi])
            pi += 1
            rows = [r for r in rows if r["series_id"] in want]
        elif "UPPER(a.source_symbol) = ANY(%s)" in sql:
            want = {s.upper() for s in params[pi]}
            pi += 1
            rows = [
                r
                for r in rows
                if any(str(a).upper() in want for a in r.get("aliases") or [])
            ]

        if "m.asset_class = ANY(%s)" in sql:
            want = set(params[pi])
            pi += 1
            rows = [r for r in rows if r["asset_class"] in want]

        if "m.status = ANY(%s)" in sql:
            want = set(params[pi])
            pi += 1
            rows = [r for r in rows if r["status"] in want]

        if "m.gap_count <=" in sql:
            cap = int(params[pi])
            pi += 1
            rows = [r for r in rows if int(r["gap_count"]) <= cap]

        if "m.suspicious_count <=" in sql:
            cap = int(params[pi])
            pi += 1
            rows = [r for r in rows if int(r["suspicious_count"]) <= cap]

        if ">= %s::date" in sql and "eod_filled_through" in sql:
            end = date.fromisoformat(params[pi]) if isinstance(params[pi], str) else params[pi]
            pi += 1
            rows = [r for r in rows if effective_last(r) >= end]

        self.last_params = params
        if "canonical_symbol" in sql:
            return [
                {
                    "series_id": r["series_id"],
                    "canonical_symbol": r.get("canonical_symbol"),
                    "asset_class": r.get("asset_class"),
                    "status": r.get("status"),
                    "first_seen": r.get("first_seen"),
                    "last_seen": r.get("last_seen"),
                    "gap_count": r.get("gap_count"),
                    "suspicious_count": r.get("suspicious_count"),
                    "disagreement_count": r.get("disagreement_count", 0),
                    "quality_score": r.get("quality_score"),
                    "calendar_id": r.get("calendar_id") or "nyse",
                    "extras": r.get("extras") or {},
                }
                for r in rows
            ]
        return [{"series_id": r["series_id"]} for r in rows]

    def fetchone(self, sql: str, params=None):
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None

    def execute(self, *args, **kwargs):
        return None
