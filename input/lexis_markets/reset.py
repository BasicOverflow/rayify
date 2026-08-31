from __future__ import annotations

from lexis_markets.storage import LakeStore, PgClient

MARKER_PREFIXES = (
    "ops/markers/",
    "ops/staging/",
    "ops/fred/done/",
    "ops/cache/",
)

L2_TABLES = (
    "stitch_decisions",
    "stitch_segments",
    "series_links",
    "symbol_aliases",
    "series_meta",
    "dataset_jobs",
    "l1_month_manifest",
    "l3_month_manifest",
    "symbol_month_coverage",
)


def clear_l2(pg: PgClient) -> int:
    tables = ", ".join(L2_TABLES)
    pg.execute(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE")
    return len(L2_TABLES)


def clear_ingest_markers(lake: LakeStore) -> int:
    n = 0
    for prefix in MARKER_PREFIXES:
        keys = lake.list_keys(prefix)
        if keys:
            lake.delete_keys(keys)
            n += len(keys)
    return n
