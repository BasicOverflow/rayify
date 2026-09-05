"""Delete superseded L1 part files after compaction."""
from __future__ import annotations

from lexis_markets.lake import LakeStore, PgClient, month_prefix
from lexis_markets.logging_setup import get_logger

logger = get_logger("kaggle.lifecycle")


def delete_l1_parts(
    lake: LakeStore,
    year: int,
    month: int,
    *,
    compacted_key: str | None = None,
) -> int:
    prefix = month_prefix(year, month)
    keys = [
        k
        for k in lake.list_keys(prefix)
        if "/part-" in k and k.endswith(".parquet") and (compacted_key is None or k != compacted_key)
    ]
    if keys:
        lake.delete_keys(keys)
    return len(keys)


def delete_l1_parts_for_month(
    lake: LakeStore,
    pg: PgClient,
    year: int,
    month: int,
) -> dict:
    row = pg.fetchone(
        "SELECT compacted_key FROM l1_month_manifest WHERE year = %s AND month = %s",
        (year, month),
    )
    compacted_key = row["compacted_key"] if row else None
    deleted = delete_l1_parts(lake, year, month, compacted_key=compacted_key)
    logger.info("delete_l1_parts year=%s month=%s deleted=%s", year, month, deleted)
    return {"year": year, "month": month, "deleted": deleted, "compacted_key": compacted_key}
