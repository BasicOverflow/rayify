"""Shared calendar helpers for cron / EOD / FRED."""
from __future__ import annotations

from datetime import date, datetime, timedelta


def as_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    text = v.isoformat() if hasattr(v, "isoformat") else str(v)
    return date.fromisoformat(str(text)[:10])


def eod_target_date(today: date | None = None) -> date:
    today = today or date.today()
    return today - timedelta(days=1)
