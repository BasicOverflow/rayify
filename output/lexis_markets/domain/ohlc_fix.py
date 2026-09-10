"""Structural OHLC repair: enforce high/low bounds around open/close."""
from __future__ import annotations

import numpy as np
import pandas as pd


def enforce_ohlc_bounds(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fix bars where high/low disagree with open/close.

    Sets ``high = max(o,h,l,c)`` and ``low = min(o,h,l,c)`` when any OHLC is finite.
    Preserves close (the research pin). Returns ``(frame, n_rows_changed)``.
    """
    cols = ("open", "high", "low", "close")
    if bars.empty or any(c not in bars.columns for c in cols):
        return bars, 0
    out = bars.copy()
    o = pd.to_numeric(out["open"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c)
    bad = ok & ((lo > np.minimum(o, c) + 1e-9) | (h + 1e-9 < np.maximum(o, c)) | (h + 1e-9 < lo))
    if not bad.any():
        return out, 0
    stack = np.column_stack([o, h, lo, c])
    new_high = np.nanmax(stack, axis=1)
    new_low = np.nanmin(stack, axis=1)
    h2 = h.copy()
    lo2 = lo.copy()
    h2[bad] = new_high[bad]
    lo2[bad] = new_low[bad]
    out["high"] = h2
    out["low"] = lo2
    return out, int(bad.sum())
