"""Gap, disagreement, and suspicious-bar metrics for canonical series."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# Long near-linear close runs are almost never real market paths (dense fills).
LINEAR_RAMP_MIN_RUN = 21
LINEAR_RAMP_REL_TOL = 1e-5
# Consecutive bars separated by this many calendar days draw fake diagonals on plots.
SPARSE_BRIDGE_MIN_DAYS = 21
# Flat identical closes (halt / bad fill) over this many sessions.
FLAT_CLOSE_MIN_RUN = 21
# One-day |return| above this relative move is treated as a spike (equity).
EXTREME_RETURN = 5.0  # 500% day — catches garbage prints, not normal gaps


def session_span_days(first: date, last: date, calendar_id: str) -> int:
    if calendar_id == "fred_native":
        return (last - first).days + 1
    return int(np.busday_count(first, last)) + 1


def gaps_from_bars(bars: pd.DataFrame, calendar_id: str) -> tuple[int, int, float]:
    if bars.empty:
        return 0, 0, 1.0
    ts = pd.to_datetime(bars["ts"]).dt.date
    first, last = ts.min(), ts.max()
    span = session_span_days(first, last, calendar_id)
    days = int(ts.nunique())
    gap = max(0, span - days)
    disagreement = int((bars.get("source_count", 1) > 1).sum()) if "source_count" in bars.columns else 0
    score = days / span if span else 1.0
    return gap, disagreement, score


def linear_ramp_mask(
    close: np.ndarray,
    *,
    min_run: int = LINEAR_RAMP_MIN_RUN,
    rel_tol: float = LINEAR_RAMP_REL_TOL,
) -> np.ndarray:
    """True where closes sit inside a long constant-slope run."""
    n = len(close)
    flagged = np.zeros(n, dtype=bool)
    if n < min_run or close.ndim != 1:
        return flagged
    finite = np.isfinite(close)
    if not finite.any():
        return flagged
    scale = float(np.nanmedian(np.abs(close[finite])))
    scale = max(scale, 1e-8)
    d1 = np.diff(close.astype(float))
    if len(d1) < 2:
        return flagged
    d2 = np.diff(d1)
    step_ok = np.isfinite(d2) & (np.abs(d2) <= rel_tol * scale)

    i = 0
    while i < len(step_ok):
        if not step_ok[i]:
            i += 1
            continue
        j = i
        while j < len(step_ok) and step_ok[j]:
            j += 1
        run_len = j - i + 2
        if run_len >= min_run:
            flagged[i : j + 2] = True
        i = j
    return flagged


def flat_close_mask(
    close: np.ndarray,
    *,
    min_run: int = FLAT_CLOSE_MIN_RUN,
) -> np.ndarray:
    """True where closes are identical for ``min_run``+ consecutive bars."""
    n = len(close)
    flagged = np.zeros(n, dtype=bool)
    if n < min_run:
        return flagged
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.isfinite(close[i]) and close[j] == close[i]:
            j += 1
        if j - i >= min_run:
            flagged[i:j] = True
        i = j if j > i else i + 1
    return flagged


def sparse_bridge_gap_days(
    ts: np.ndarray,
    *,
    min_days: int = SPARSE_BRIDGE_MIN_DAYS,
) -> tuple[int, np.ndarray]:
    """Sum calendar days in oversized gaps; return mask of bars that touch those gaps."""
    n = len(ts)
    touch = np.zeros(n, dtype=bool)
    if n < 2:
        return 0, touch
    stamps = pd.to_datetime(ts)
    total = 0
    for i in range(1, n):
        gap = int((stamps[i] - stamps[i - 1]).days)
        if gap >= min_days:
            total += gap
            touch[i - 1] = True
            touch[i] = True
    return total, touch


def ohlc_violation_count(bars: pd.DataFrame) -> int:
    """Bars where low/high disagree with open/close (when OHLC present)."""
    cols = ("open", "high", "low", "close")
    if bars.empty or any(c not in bars.columns for c in cols):
        return 0
    o = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c)
    bad = ok & ((lo > np.minimum(o, c) + 1e-9) | (h + 1e-9 < np.maximum(o, c)))
    return int(bad.sum())


def non_positive_close_count(bars: pd.DataFrame) -> int:
    if bars.empty or "close" not in bars.columns:
        return 0
    c = pd.to_numeric(bars["close"], errors="coerce")
    return int(((c.isna()) | (c <= 0)).sum())


def duplicate_ts_count(bars: pd.DataFrame) -> int:
    if bars.empty or "ts" not in bars.columns:
        return 0
    ts = pd.to_datetime(bars["ts"]).dt.normalize()
    return int(ts.duplicated().sum())


def extreme_return_count(
    close: np.ndarray,
    *,
    threshold: float = EXTREME_RETURN,
) -> int:
    """Count bars whose |close/prev - 1| exceeds ``threshold`` (garbage prints)."""
    if len(close) < 2:
        return 0
    prev = close[:-1]
    cur = close[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(np.isfinite(prev) & (np.abs(prev) > 1e-12), np.abs(cur / prev - 1.0), 0.0)
    return int((ret >= threshold).sum())


def suspicious_from_bars(
    bars: pd.DataFrame,
    *,
    calendar_id: str | None = None,
    min_run: int = LINEAR_RAMP_MIN_RUN,
    rel_tol: float = LINEAR_RAMP_REL_TOL,
    bridge_min_days: int = SPARSE_BRIDGE_MIN_DAYS,
    flat_min_run: int = FLAT_CLOSE_MIN_RUN,
    extreme_return: float = EXTREME_RETURN,
) -> tuple[int, list[str]]:
    """Return ``(suspicious_day_count, flag_names)`` for stitched bars.

    Flags:
      - ``linear_ramp`` — dense constant-slope fills
      - ``sparse_bridge`` — multi-week calendar gaps (ITMR-style plot diagonals);
        skipped for ``fred_native``
      - ``flat_close`` — long identical-close runs
      - ``ohlc_violation`` — high/low inconsistent with open/close
      - ``non_positive_close`` — NaN or <= 0 closes
      - ``duplicate_ts`` — repeated timestamps
      - ``extreme_return`` — single-bar moves beyond ``extreme_return``
    """
    if bars.empty or "close" not in bars.columns:
        return 0, []
    df = bars.sort_values("ts") if "ts" in bars.columns else bars
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)

    ramp = linear_ramp_mask(close, min_run=min_run, rel_tol=rel_tol)
    ramp_days = int(ramp.sum())
    flat = flat_close_mask(close, min_run=flat_min_run)
    flat_days = int(flat.sum())
    # Flat identical closes are also constant-slope; prefer linear_ramp for the
    # day count when both fire, but still surface ``flat_close`` in flags.
    if ramp_days and flat_days:
        flat_for_total = 0
    else:
        flat_for_total = flat_days

    bridge_days = 0
    if calendar_id != "fred_native" and "ts" in df.columns:
        ts = pd.to_datetime(df["ts"]).to_numpy()
        bridge_days, _touch = sparse_bridge_gap_days(ts, min_days=bridge_min_days)

    ohlc_n = ohlc_violation_count(df)
    nonpos_n = non_positive_close_count(df)
    dup_n = duplicate_ts_count(df)
    extreme_n = 0
    if calendar_id != "fred_native":
        extreme_n = extreme_return_count(close, threshold=extreme_return)

    flags: list[str] = []
    if ramp_days:
        flags.append("linear_ramp")
    if bridge_days:
        flags.append("sparse_bridge")
    if flat_days >= flat_min_run:
        flags.append("flat_close")
    if ohlc_n:
        flags.append("ohlc_violation")
    if nonpos_n:
        flags.append("non_positive_close")
    if dup_n:
        flags.append("duplicate_ts")
    if extreme_n:
        flags.append("extreme_return")

    total = ramp_days + bridge_days + flat_for_total + ohlc_n + nonpos_n + dup_n + extreme_n
    return total, flags


def series_quality_window(bars: pd.DataFrame, calendar_id: str, meta: dict | None = None) -> dict:
    gap, disagreement, score = gaps_from_bars(bars, calendar_id)
    suspicious, flags = suspicious_from_bars(bars, calendar_id=calendar_id)
    return {
        "gap_count": gap,
        "disagreement_count": disagreement,
        "suspicious_count": suspicious,
        "quality_score": score,
        "flags": flags,
        "status": (meta or {}).get("status"),
    }


def apply_min_volume(bars: pd.DataFrame, min_volume: float | None) -> pd.DataFrame:
    if min_volume is None or bars.empty or "volume" not in bars.columns:
        return bars
    return bars[bars["volume"].fillna(0) >= min_volume]
