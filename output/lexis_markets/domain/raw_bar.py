"""L1 raw bar column schema (MinIO layer_1).

Each row is vendor-native OHLCV keyed by ``source`` + ``source_symbol`` + ``ts``.
"""
from __future__ import annotations

RAW_BAR_COLUMNS = [
    "source",
    "source_symbol",
    "series_type",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "dividend",
    "split",
    "currency",
    "fetched_at",
    "realtime_start",
    "realtime_end",
    "extras",
]

CANONICAL_BAR_COLUMNS = [
    "series_id",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "source",
    "source_count",
    "data_quality",
]
