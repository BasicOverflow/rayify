"""Shared calendar helpers for cron / EOD / FRED."""
from __future__ import annotations

from datetime import date, timedelta


def eod_target_date(today: date | None = None) -> date:
    today = today or date.today()
    return today - timedelta(days=1)
