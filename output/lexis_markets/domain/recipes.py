"""Dataset recipe shapes and NaN policies for Serve exports.

Recipes (``mode`` on ``POST /v1/datasets``):

- ``eod_snapshot`` — one row per series: last daily bar on/before ``eod_date``.
- ``range_panel`` (alias ``range``) — long panel: one row per (series_id, session).
- ``wide_matrix`` — dates × tickers matrix of a single value column (default ``close``).

NaN policy (``nan_policy``, wide_matrix only):

- ``keep`` (default) — calendar union of all series; missing sessions stay NaN.
- ``drop_rows`` — drop any date where at least one series is NaN.
- ``ffill`` — forward-fill within each series column (leading NaNs remain).
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

RecipeMode = Literal["range", "range_panel", "eod_snapshot", "wide_matrix"]
NanPolicy = Literal["keep", "drop_rows", "ffill"]

DATASET_RECIPES = frozenset({"range", "range_panel", "eod_snapshot", "wide_matrix"})
NAN_POLICIES = frozenset({"keep", "drop_rows", "ffill"})

# Canonical names after normalize (``range`` → ``range_panel``).
CANONICAL_RECIPES = frozenset({"range_panel", "eod_snapshot", "wide_matrix"})


def canonicalize_recipe_mode(raw: str | None) -> str:
    mode = (raw or "range_panel").strip().lower()
    if mode == "range":
        return "range_panel"
    if mode not in DATASET_RECIPES:
        raise ValueError(f"mode must be one of {sorted(DATASET_RECIPES)}, got {raw!r}")
    return mode


def parse_nan_policy(raw: str | None) -> NanPolicy:
    policy = (raw or "keep").strip().lower()
    if policy not in NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {sorted(NAN_POLICIES)}, got {raw!r}")
    return policy  # type: ignore[return-value]


def bars_to_wide_matrix(
    bars: pd.DataFrame,
    *,
    value_col: str = "close",
    column_key: str = "series_id",
    nan_policy: NanPolicy = "keep",
) -> pd.DataFrame:
    """Pivot long bars to ``ts`` × ticker matrix.

    Prefer ``series_id`` (always on canonical bars). Pass ``canonical_symbol``
    only when that column has been joined onto the frame.
    """
    if bars is None or bars.empty:
        return pd.DataFrame()
    if value_col not in bars.columns:
        raise ValueError(f"value_col {value_col!r} not in bars columns")

    out = bars.copy()
    out["ts"] = pd.to_datetime(out["ts"]).dt.date
    key = column_key
    if key not in out.columns or out[key].isna().all():
        key = "series_id"
    if key not in out.columns:
        raise ValueError(f"column_key {column_key!r} not in bars")

    wide = out.pivot_table(
        index="ts",
        columns=key,
        values=value_col,
        aggfunc="last",
    )
    wide = wide.sort_index()
    wide.columns.name = None

    if nan_policy == "keep":
        return wide.reset_index()
    if nan_policy == "drop_rows":
        return wide.dropna(how="any").reset_index()
    if nan_policy == "ffill":
        return wide.ffill().reset_index()
    raise ValueError(f"unknown nan_policy: {nan_policy!r}")
