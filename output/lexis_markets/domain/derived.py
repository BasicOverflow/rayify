"""Post-stitch derived feature columns (moving averages, returns, masks, …).

API ``features`` tokens:
  - ``sma_{window}`` / ``ema_{window}`` — moving averages of ``close``
  - ``returns`` — simple close-to-close return (first bar NaN)
  - ``log_price`` — ``log(close)`` (non-positive → NaN)
  - ``gap_mask`` — True when calendar gap to previous bar exceeds one session
  - ``is_suspicious`` — True on bars flagged by integrity / anomaly masks

RSI/MACD remain reserved for later registration.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from lexis_markets.domain.quality import (
    FLAT_CLOSE_MIN_RUN,
    LINEAR_RAMP_MIN_RUN,
    LINEAR_RAMP_REL_TOL,
    flat_close_mask,
    linear_ramp_mask,
)

MA_FEATURE_RE = re.compile(r"^(sma|ema)_(\d+)$", re.IGNORECASE)

COMMON_MA_WINDOWS: frozenset[int] = frozenset({5, 10, 20, 50, 100, 200})

SCALAR_FEATURES: frozenset[str] = frozenset(
    {"returns", "log_price", "gap_mask", "is_suspicious"}
)

PLANNED_DERIVED_FEATURES: frozenset[str] = frozenset({"rsi", "macd"})

MIN_WINDOW = 2
MAX_WINDOW = 500


def parse_ma_feature(name: str) -> tuple[str, int] | None:
    m = MA_FEATURE_RE.match(name.strip().lower())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _validate_window(window: int) -> None:
    if window < MIN_WINDOW or window > MAX_WINDOW:
        raise ValueError(f"MA window must be between {MIN_WINDOW} and {MAX_WINDOW}, got {window}")


def _sorted_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty or "series_id" not in bars.columns:
        return bars
    out = bars.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    return out.sort_values(["series_id", "ts"]).reset_index(drop=True)


def add_sma(bars: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    out = _sorted_bars(bars)
    col = f"sma_{window}"
    out[col] = out.groupby("series_id", sort=False)["close"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    return out


def add_ema(bars: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    out = _sorted_bars(bars)
    col = f"ema_{window}"
    out[col] = out.groupby("series_id", sort=False)["close"].transform(
        lambda s: s.ewm(span=window, adjust=False, min_periods=1).mean()
    )
    return out


def add_returns(bars: pd.DataFrame) -> pd.DataFrame:
    out = _sorted_bars(bars)
    out["returns"] = out.groupby("series_id", sort=False)["close"].pct_change()
    return out


def add_log_price(bars: pd.DataFrame) -> pd.DataFrame:
    out = _sorted_bars(bars)
    close = pd.to_numeric(out["close"], errors="coerce")
    out["log_price"] = np.where(close > 0, np.log(close), np.nan)
    return out


def add_gap_mask(bars: pd.DataFrame, *, max_session_days: int = 4) -> pd.DataFrame:
    """Mark bars whose gap from the previous observation exceeds ``max_session_days``.

    Equity weekends are typically 3 calendar days; ``4`` catches holiday gaps
    without flagging normal Fri→Mon. First bar per series is False.
    """
    out = _sorted_bars(bars)
    if out.empty or "ts" not in out.columns:
        out["gap_mask"] = False
        return out

    def _delta_days(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s).diff().dt.days

    deltas = out.groupby("series_id", sort=False)["ts"].transform(_delta_days)
    out["gap_mask"] = (deltas > max_session_days).fillna(False)
    return out


def _ohlc_row_mask(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    flagged = np.zeros(n, dtype=bool)
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        return flagged
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    # high < max(o,c) or low > min(o,c) or high < low
    flagged |= np.isfinite(h) & np.isfinite(low) & (h < low)
    mx = np.fmax(o, c)
    mn = np.fmin(o, c)
    flagged |= np.isfinite(h) & np.isfinite(mx) & (h + 1e-12 < mx)
    flagged |= np.isfinite(low) & np.isfinite(mn) & (low - 1e-12 > mn)
    return flagged


def add_is_suspicious(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-bar integrity / anomaly mask (linear ramp, flat close, OHLC, non-positive)."""
    out = _sorted_bars(bars)
    if out.empty:
        out["is_suspicious"] = False
        return out

    parts: list[pd.Series] = []
    for _, g in out.groupby("series_id", sort=False):
        close = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=float)
        ramp = linear_ramp_mask(close, min_run=LINEAR_RAMP_MIN_RUN, rel_tol=LINEAR_RAMP_REL_TOL)
        flat = flat_close_mask(close, min_run=FLAT_CLOSE_MIN_RUN)
        ohlc = _ohlc_row_mask(g)
        nonpos = np.isfinite(close) & (close <= 0)
        parts.append(pd.Series(ramp | flat | ohlc | nonpos, index=g.index))
    out["is_suspicious"] = pd.concat(parts).reindex(out.index).fillna(False).astype(bool)
    return out


def list_derived_feature_help() -> list[str]:
    examples = [f"sma_{w}" for w in sorted(COMMON_MA_WINDOWS)]
    examples += [f"ema_{w}" for w in sorted(COMMON_MA_WINDOWS)]
    examples += sorted(SCALAR_FEATURES)
    return examples + sorted(PLANNED_DERIVED_FEATURES)


def apply_derived_features(bars: pd.DataFrame, features: list[str] | None) -> pd.DataFrame:
    if not features:
        return bars
    if bars.empty:
        return bars

    out = bars
    for raw in features:
        name = raw.strip().lower()
        ma = parse_ma_feature(name)
        if ma is not None:
            kind, window = ma
            out = add_sma(out, window) if kind == "sma" else add_ema(out, window)
            continue
        if name == "returns":
            out = add_returns(out)
            continue
        if name in ("log_price", "log-price"):
            out = add_log_price(out)
            continue
        if name == "gap_mask":
            out = add_gap_mask(out)
            continue
        if name == "is_suspicious":
            out = add_is_suspicious(out)
            continue
        if name in PLANNED_DERIVED_FEATURES:
            raise NotImplementedError(f"derived feature {name!r} is not implemented yet")
        raise ValueError(
            f"unknown derived feature {raw!r}; use sma_<n>, ema_<n>, or "
            f"{sorted(SCALAR_FEATURES)}, e.g. {list_derived_feature_help()[:6]}"
        )
    return out
