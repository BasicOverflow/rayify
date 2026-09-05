"""Post-stitch derived feature columns (moving averages, RSI, MACD, …).

API ``features`` tokens:
  - ``sma_{window}`` — simple moving average of ``close`` (e.g. ``sma_20``, ``sma_200``)
  - ``ema_{window}`` — exponential moving average of ``close`` (e.g. ``ema_12``, ``ema_50``)

RSI/MACD remain reserved for later registration.
"""
from __future__ import annotations

import re

import pandas as pd

MA_FEATURE_RE = re.compile(r"^(sma|ema)_(\d+)$", re.IGNORECASE)

# Windows commonly used in presets / docs; any integer in [MIN_WINDOW, MAX_WINDOW] is accepted.
COMMON_MA_WINDOWS: frozenset[int] = frozenset({5, 10, 20, 50, 100, 200})

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


def list_derived_feature_help() -> list[str]:
    examples = [f"sma_{w}" for w in sorted(COMMON_MA_WINDOWS)]
    examples += [f"ema_{w}" for w in sorted(COMMON_MA_WINDOWS)]
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
        if name in PLANNED_DERIVED_FEATURES:
            raise NotImplementedError(f"derived feature {name!r} is not implemented yet")
        raise ValueError(
            f"unknown derived feature {raw!r}; use sma_<n> or ema_<n> "
            f"(window {MIN_WINDOW}-{MAX_WINDOW}), e.g. {list_derived_feature_help()[:4]}"
        )
    return out
