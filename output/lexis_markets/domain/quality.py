"""Gap, disagreement, and suspicious-bar metrics for canonical series."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# Long near-linear close runs are almost never real market paths (dense fills).
LINEAR_RAMP_MIN_RUN = 21
LINEAR_RAMP_REL_TOL = 1e-5
# Split-adjusted early history sits at micro prices with tick-sized steps; skip those.
LINEAR_RAMP_MIN_PRICE = 0.05
# Quiet names drifting a few percent over years are not synthetic ramps.
LINEAR_RAMP_MIN_REL_RANGE = 0.25
# Consecutive bars separated by this many calendar days draw fake diagonals on plots.
SPARSE_BRIDGE_MIN_DAYS = 21
# Flat identical closes (halt / bad fill) over this many sessions.
FLAT_CLOSE_MIN_RUN = 21
# One-day |return| above this relative move is treated as a spike (equity).
EXTREME_RETURN = 5.0  # 500% day — catches garbage prints, not normal gaps

# Per-flag day counts persisted on series_meta (plus aggregate suspicious_count).
QUALITY_FLAG_COLUMNS: tuple[str, ...] = (
    "linear_ramp",
    "sparse_bridge",
    "flat_close",
    "ohlc_violation",
    "non_positive_close",
    "duplicate_ts",
    "extreme_return",
)


def infer_native_period_days(ts: pd.Series) -> int:
    """Infer FRED-like observation period (days) from timestamps."""
    vals = sorted({d for d in pd.to_datetime(ts).dt.date.dropna().tolist()})
    if len(vals) < 2:
        return 30
    diffs = np.array([(vals[i] - vals[i - 1]).days for i in range(1, len(vals))], dtype=float)
    med = float(np.median(diffs))
    if med <= 2:
        return 1
    if med <= 10:
        return 7
    if med <= 45:
        return 30
    if med <= 120:
        return 91
    return 365


def expected_native_observations(first: date, last: date, period_days: int) -> int:
    if last < first:
        return 0
    if period_days <= 1:
        return (last - first).days + 1
    return max(1, int(round((last - first).days / float(period_days))) + 1)


def session_span_days(
    first: date,
    last: date,
    calendar_id: str,
    *,
    period_days: int | None = None,
) -> int:
    """Expected observation count in ``[first, last]`` for gap scoring."""
    if calendar_id == "fred_native":
        pdays = period_days or 30
        # Daily FRED (yields, NFCI daily cousins): weekends are not missing observations.
        if pdays <= 1:
            return int(np.busday_count(first, last)) + 1
        return expected_native_observations(first, last, pdays)
    return int(np.busday_count(first, last)) + 1


def gaps_from_bars(bars: pd.DataFrame, calendar_id: str) -> tuple[int, int, float]:
    if bars.empty:
        return 0, 0, 1.0
    ts = pd.to_datetime(bars["ts"]).dt.date
    first, last = ts.min(), ts.max()
    period = infer_native_period_days(ts) if calendar_id == "fred_native" else None
    span = session_span_days(first, last, calendar_id, period_days=period)
    days = int(ts.nunique())
    gap = max(0, span - days)
    disagreement = int((bars.get("source_count", 1) > 1).sum()) if "source_count" in bars.columns else 0
    coverage = min(1.0, days / span) if span else 1.0
    return gap, disagreement, coverage


def dense_observation_window(
    ts: pd.Series,
    *,
    max_gap_days: int = SPARSE_BRIDGE_MIN_DAYS,
    min_bars: int = 40,
) -> tuple[date, date] | None:
    """Longest run of observations with consecutive gaps ``< max_gap_days``.

    Trims phantom pre-listing / post-delist sparse prints that inflate gap scores.
    """
    vals = sorted({d.date() if hasattr(d, "date") else d for d in pd.to_datetime(ts).dropna()})
    if len(vals) < min_bars:
        if len(vals) >= 2:
            return vals[0], vals[-1]
        return (vals[0], vals[0]) if vals else None

    best: tuple[date, date] | None = None
    best_n = 0
    run_start = 0
    for i in range(1, len(vals) + 1):
        broken = i == len(vals) or (vals[i] - vals[i - 1]).days >= max_gap_days
        if broken:
            n = i - run_start
            if n > best_n:
                best_n = n
                best = (vals[run_start], vals[i - 1])
            run_start = i
    if best is None or best_n < min_bars:
        return vals[0], vals[-1]
    return best


def composite_quality_score(coverage: float, suspicious_count: int, bar_count: int) -> float:
    """Coverage penalized by fraction of suspicious bar-days (clamped to ``[0, 1]``)."""
    if bar_count <= 0:
        return float(coverage)
    flag_rate = min(1.0, float(suspicious_count) / float(bar_count))
    return max(0.0, min(1.0, float(coverage) * (1.0 - 0.5 * flag_rate)))


def quarantine_junk_rates(
    flag_counts: dict[str, int],
    bar_count: int,
    *,
    ramp_flat_frac: float = 0.35,
) -> bool:
    """True when linear_ramp or flat_close covers too much of the series (dead fills)."""
    if bar_count <= 0:
        return False
    ramp = int(flag_counts.get("linear_ramp", 0))
    flat = int(flag_counts.get("flat_close", 0))
    return max(ramp, flat) / float(bar_count) >= ramp_flat_frac


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
            sl = close[i : j + 2]
            if _ramp_run_is_material(sl):
                flagged[i : j + 2] = True
        i = j
    return flagged


def _ramp_run_is_material(close_run: np.ndarray) -> bool:
    """True for dense synthetic ramps; false for micro-adjusted / quiet drift."""
    finite = close_run[np.isfinite(close_run)]
    if len(finite) == 0:
        return False
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    med = float(np.median(np.abs(finite)))
    if hi - lo <= 1e-12:
        return False  # identical closes are flat_close / halt, not a ramp
    if med < LINEAR_RAMP_MIN_PRICE:
        return False
    return (hi - lo) / max(med, 1e-12) >= LINEAR_RAMP_MIN_REL_RANGE


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
    close: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    """Sum calendar days in oversized gaps; return mask of bars that touch those gaps.

    A long gap with the same close on both sides is a halt, not a missing-data bridge.
    """
    n = len(ts)
    touch = np.zeros(n, dtype=bool)
    if n < 2:
        return 0, touch
    stamps = pd.to_datetime(ts)
    closes = None if close is None else np.asarray(close, dtype=float)
    total = 0
    for i in range(1, n):
        gap = int((stamps[i] - stamps[i - 1]).days)
        if gap < min_days:
            continue
        if closes is not None and i < len(closes):
            c0, c1 = closes[i - 1], closes[i]
            if np.isfinite(c0) and np.isfinite(c1) and abs(c1 - c0) <= 1e-12 * max(1.0, abs(c0)):
                continue
        total += gap
        touch[i - 1] = True
        touch[i] = True
    return total, touch


def collapse_halt_flats(
    bars: pd.DataFrame, *, min_run: int = FLAT_CLOSE_MIN_RUN
) -> pd.DataFrame:
    """Keep first and last bar of a long identical-close run; drop copied interiors."""
    if bars.empty or "close" not in bars.columns:
        return bars
    if "series_id" not in bars.columns:
        return _collapse_halt_flats_one(bars, min_run=min_run)
    parts = []
    for _, grp in bars.groupby("series_id", sort=False):
        parts.append(_collapse_halt_flats_one(grp, min_run=min_run))
    return pd.concat(parts, ignore_index=True) if parts else bars


def _collapse_halt_flats_one(g: pd.DataFrame, *, min_run: int) -> pd.DataFrame:
    g = g.sort_values("ts").reset_index(drop=True)
    close = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=float)
    n = len(close)
    if n < min_run:
        return g
    keep = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.isfinite(close[i]) and close[j] == close[i]:
            j += 1
        if j - i >= min_run:
            keep[i + 1 : j - 1] = False
        i = j if j > i else i + 1
    return g.loc[keep].reset_index(drop=True)


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
    """Return ``(suspicious_day_count, flag_names)`` for stitched bars."""
    total, flags, _counts = suspicious_flag_details(
        bars,
        calendar_id=calendar_id,
        min_run=min_run,
        rel_tol=rel_tol,
        bridge_min_days=bridge_min_days,
        flat_min_run=flat_min_run,
        extreme_return=extreme_return,
    )
    return total, flags


def suspicious_flag_details(
    bars: pd.DataFrame,
    *,
    calendar_id: str | None = None,
    min_run: int = LINEAR_RAMP_MIN_RUN,
    rel_tol: float = LINEAR_RAMP_REL_TOL,
    bridge_min_days: int = SPARSE_BRIDGE_MIN_DAYS,
    flat_min_run: int = FLAT_CLOSE_MIN_RUN,
    extreme_return: float = EXTREME_RETURN,
) -> tuple[int, list[str], dict[str, int]]:
    """Return ``(suspicious_day_count, flag_names, per_flag_day_counts)``."""
    zero = {k: 0 for k in QUALITY_FLAG_COLUMNS}
    if bars.empty or "close" not in bars.columns:
        return 0, [], zero
    df = bars.sort_values("ts") if "ts" in bars.columns else bars
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)

    ramp = linear_ramp_mask(close, min_run=min_run, rel_tol=rel_tol)
    ramp_days = int(ramp.sum())
    flat = flat_close_mask(close, min_run=flat_min_run)
    flat_days = int(flat.sum())
    # Prefer linear_ramp for the day count when both fire; still surface flat_close.
    if ramp_days and flat_days:
        flat_for_total = 0
    else:
        flat_for_total = flat_days

    bridge_days = 0
    if calendar_id != "fred_native" and "ts" in df.columns:
        ts = pd.to_datetime(df["ts"]).to_numpy()
        bridge_days, _touch = sparse_bridge_gap_days(
            ts, min_days=bridge_min_days, close=close
        )

    ohlc_n = ohlc_violation_count(df)
    nonpos_n = non_positive_close_count(df)
    dup_n = duplicate_ts_count(df)
    extreme_n = 0
    if calendar_id != "fred_native":
        extreme_n = extreme_return_count(close, threshold=extreme_return)

    counts = {
        "linear_ramp": ramp_days,
        "sparse_bridge": bridge_days,
        "flat_close": flat_days if flat_days >= flat_min_run else 0,
        "ohlc_violation": ohlc_n,
        "non_positive_close": nonpos_n,
        "duplicate_ts": dup_n,
        "extreme_return": extreme_n,
    }
    flags = [k for k, v in counts.items() if v > 0]
    total = ramp_days + bridge_days + flat_for_total + ohlc_n + nonpos_n + dup_n + extreme_n
    return total, flags, counts


def series_quality_window(bars: pd.DataFrame, calendar_id: str, meta: dict | None = None) -> dict:
    gap, disagreement, coverage = gaps_from_bars(bars, calendar_id)
    suspicious, flags, flag_counts = suspicious_flag_details(bars, calendar_id=calendar_id)
    bar_count = int(len(bars)) if bars is not None else 0
    score = composite_quality_score(coverage, suspicious, bar_count)
    return {
        "gap_count": gap,
        "disagreement_count": disagreement,
        "suspicious_count": suspicious,
        "coverage_score": coverage,
        "quality_score": score,
        "flags": flags,
        "flag_counts": flag_counts,
        "status": (meta or {}).get("status"),
        "quarantine": quarantine_junk_rates(flag_counts, bar_count),
    }


def apply_min_volume(bars: pd.DataFrame, min_volume: float | None) -> pd.DataFrame:
    if min_volume is None or bars.empty or "volume" not in bars.columns:
        return bars
    return bars[bars["volume"].fillna(0) >= min_volume]
