"""ET calendar stepper for weeksim (no freezegun on supervisor sleep)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)


@dataclass
class SimClock:
    """Simulated 'today' in America/New_York; EOD nights run after the delay window."""

    week_start: date
    day_index: int = 0
    eod_delay_hours: float = 3.0

    @classmethod
    def from_env(cls) -> SimClock:
        raw = os.environ.get("MARKETS_TEST_WEEK_START", "2025-03-11").strip()
        days = int(os.environ.get("MARKETS_TEST_WEEK_DAYS", "7") or 7)
        start = date.fromisoformat(raw)
        clock = cls(week_start=start)
        clock._week_days = days
        return clock

    def __post_init__(self) -> None:
        if not hasattr(self, "_week_days"):
            self._week_days = 7

    @property
    def week_days(self) -> int:
        return int(getattr(self, "_week_days", 7))

    @property
    def today(self) -> date:
        return self.week_start + timedelta(days=self.day_index)

    def morning(self) -> datetime:
        return datetime.combine(self.today, time(9, 30), tzinfo=ET)

    def eod_window_now(self) -> datetime:
        close = datetime.combine(self.today, MARKET_CLOSE, tzinfo=ET)
        return close + timedelta(hours=self.eod_delay_hours, minutes=5)

    def eod_target(self) -> date:
        return self.today - timedelta(days=1)

    @staticmethod
    def is_weekend(d: date) -> bool:
        return d.weekday() >= 5

    def advance(self, days: int = 1) -> None:
        self.day_index += days

    def nights(self) -> list[date]:
        return [self.week_start + timedelta(days=i) for i in range(self.week_days)]

    def apply_eod_start_env(self) -> None:
        # Clip YF backfill so seed/EOD stay near the simulated window.
        clip = self.week_start - timedelta(days=120)
        os.environ["MARKETS_TEST_EOD_START"] = clip.isoformat()
