"""FRED / ALFRED vintage helpers (``output_type=1`` realtime periods)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from lexis_markets.lake import utcnow
from lexis_markets.logging_setup import get_logger

logger = get_logger("domain.fred_alfred")

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_VINTAGE_DATES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"
# Long-form observations with realtime_start/realtime_end/value (not wide vintage cols).
ALFRED_OUTPUT_TYPE_VINTAGE = 1
FRED_PAGE_LIMIT = 100_000
FRED_HISTORY_FLOOR = date(1970, 1, 1)


def clamp_vintage_start(raw: date, *, floor: date = FRED_HISTORY_FLOOR) -> date:
    return max(raw, floor)


def vintage_windows(start: date, end: date, *, max_days: int = 365) -> list[tuple[date, date]]:
    """Split a closed calendar range into chunks (fallback when vintagedates are empty)."""
    if start > end:
        return []
    max_days = max(1, int(max_days))
    windows: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        win_end = min(end, cur + timedelta(days=max_days - 1))
        windows.append((cur, win_end))
        cur = win_end + timedelta(days=1)
    return windows


def filter_vintage_dates(
    vintage_dates: list[date], start: date, end: date
) -> list[date]:
    """Keep revision dates inside a closed realtime range (sorted unique)."""
    if start > end:
        return []
    return sorted({d for d in vintage_dates if start <= d <= end})


def pack_vintage_date_windows(
    vintage_dates: list[date], *, max_days: int
) -> list[tuple[date, date]]:
    """Pack real ALFRED revision dates into realtime windows.

    Skips empty calendar gaps: a sparse series with revisions years apart becomes
    one window per packed group, not one HTTP call per empty multi-year span.
    """
    if not vintage_dates:
        return []
    max_days = max(1, int(max_days))
    dates = sorted(set(vintage_dates))
    windows: list[tuple[date, date]] = []
    i = 0
    n = len(dates)
    while i < n:
        start = dates[i]
        j = i
        while j + 1 < n and (dates[j + 1] - start).days + 1 <= max_days:
            j += 1
        windows.append((start, dates[j]))
        i = j + 1
    return windows


def plan_ingest_windows(
    *,
    vintage_start: date,
    vintage_end: date,
    vintage_dates: list[date],
    max_days: int,
) -> list[tuple[date, date]]:
    """Prefer vintage-date packing; fall back to calendar split when ALFRED has none."""
    dates = filter_vintage_dates(vintage_dates, vintage_start, vintage_end)
    if dates:
        return pack_vintage_date_windows(dates, max_days=max_days)
    return vintage_windows(vintage_start, vintage_end, max_days=max_days)


@dataclass
class AdaptiveWindowDays:
    """Shrink on gateway bisects; grow slowly after consecutive clean fetches."""

    current: int
    min_days: int
    max_days: int
    success_streak: int = 0
    grow_after: int = 3
    grow_factor: float = 1.25

    def __post_init__(self) -> None:
        self.min_days = max(1, int(self.min_days))
        self.max_days = max(self.min_days, int(self.max_days))
        self.current = min(self.max_days, max(self.min_days, int(self.current)))

    def preferred(self) -> int:
        return self.current

    def observe(self, *, span_days: int, bisects: int) -> int:
        span_days = max(1, int(span_days))
        bisects = max(0, int(bisects))
        if bisects > 0:
            shrunk = max(self.min_days, min(self.current, span_days // 2))
            if shrunk < self.current:
                logger.info(
                    "fred window sizer shrink %s -> %s (bisects=%s span=%s)",
                    self.current,
                    shrunk,
                    bisects,
                    span_days,
                )
            self.current = shrunk
            self.success_streak = 0
            return self.current
        self.success_streak += 1
        if self.success_streak >= self.grow_after and self.current < self.max_days:
            grown = min(
                self.max_days,
                max(int(self.current * self.grow_factor), self.current + self.min_days),
            )
            if grown > self.current:
                logger.info(
                    "fred window sizer grow %s -> %s (streak=%s)",
                    self.current,
                    grown,
                    self.success_streak,
                )
                self.current = grown
            self.success_streak = 0
        return self.current


@dataclass
class FetchStats:
    span_days: int = 0
    bisects: int = 0
    attempts: int = 0
    windows: int = 0


def observations_to_df(series_id: str, observations: list, *, mode: str) -> pd.DataFrame:
    rows = []
    for o in observations:
        if o.get("value") in (".", None, ""):
            continue
        vintage = o.get("realtime_end") or o.get("realtime_start")
        rows.append(
            {
                "source": "fred",
                "source_symbol": series_id,
                "series_type": "macro",
                "ts": o["date"],
                "open": None,
                "high": None,
                "low": None,
                "close": float(o["value"]),
                "volume": None,
                "adj_close": None,
                "dividend": None,
                "split": None,
                "currency": "USD",
                "fetched_at": utcnow(),
                "realtime_start": o.get("realtime_start", vintage),
                "realtime_end": o.get("realtime_end", vintage),
                "extras": json.dumps({"mode": mode, "vintage": vintage}),
            }
        )
    # Wrong output_type (wide vintage columns) yields obs without ``value`` — fail
    # immediately instead of burning a full multi-window fan-out for ok=0.
    if observations and not rows and "value" not in observations[0]:
        keys = sorted(observations[0].keys())
        raise ValueError(
            f"FRED observations for {series_id} missing 'value' (keys={keys}); "
            f"expected output_type={ALFRED_OUTPUT_TYPE_VINTAGE} long-form realtime periods"
        )
    return pd.DataFrame(rows)


def collapse_fred_vintages(df: pd.DataFrame, *, as_of: date | None = None) -> pd.DataFrame:
    """Pick one FRED row per (source_symbol, ts).

    ``as_of=None`` (API ``latest``): newest vintage by ``realtime_end``.
    ``as_of=T``: ALFRED point-in-time — keep vintages with
    ``realtime_start <= T <= realtime_end``, then the latest ``realtime_start``.
    """
    if df.empty or "source" not in df.columns:
        return df
    fred = df[df["source"] == "fred"].copy()
    other = df[df["source"] != "fred"]
    if fred.empty:
        return df
    if "realtime_end" not in fred.columns:
        return df
    fred["realtime_end"] = pd.to_datetime(fred["realtime_end"], errors="coerce").dt.date
    has_start = "realtime_start" in fred.columns
    if has_start:
        fred["realtime_start"] = pd.to_datetime(fred["realtime_start"], errors="coerce").dt.date
    if as_of is not None:
        if has_start:
            fred = fred[
                fred["realtime_start"].notna()
                & (fred["realtime_start"] <= as_of)
                & (fred["realtime_end"].isna() | (fred["realtime_end"] >= as_of))
            ]
        else:
            fred = fred[fred["realtime_end"].notna() & (fred["realtime_end"] >= as_of)]
    if fred.empty:
        return other.reset_index(drop=True) if not other.empty else fred
    sort_key = "realtime_start" if has_start else "realtime_end"
    fred = fred.sort_values(["source_symbol", "ts", sort_key])
    fred = fred.groupby(["source_symbol", "ts"], as_index=False).tail(1)
    if other.empty:
        return fred.reset_index(drop=True)
    return pd.concat([other, fred], ignore_index=True)
