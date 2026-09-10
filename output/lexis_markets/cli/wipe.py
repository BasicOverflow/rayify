"""Dev utility: wipe lake + Postgres (full) or FRED/macro only (``--fred-only``).

CLI-only ops tool — not a supervisor/Serve job type.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lexis_markets.config import MarketsConfig
from lexis_markets.domain.raw_bar import RAW_BAR_COLUMNS
from lexis_markets.jobs.queue import wipe_supervisor_sqlite
from lexis_markets.lake import LakeStore, PgClient, ensure_schema, write_parquet_lake
from lexis_markets.logging_setup import configure_logging, get_logger

logger = get_logger("cli.wipe")

FRED_MARKERS = ("ops/markers/fred_backfill.json",)


def wipe_all(cfg: MarketsConfig | None = None) -> dict:
    configure_logging()
    cfg = cfg or MarketsConfig.from_env()
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)

    # When MARKETS_LAKE_PREFIX is set, only delete under that prefix (shared buckets).
    list_prefix = lake.prefix or None
    token = None
    total = 0
    while True:
        kw: dict = {"Bucket": lake.bucket, "MaxKeys": 1000}
        if list_prefix:
            kw["Prefix"] = list_prefix
        if token:
            kw["ContinuationToken"] = token
        resp = lake.client.list_objects_v2(**kw)
        keys = [x["Key"] for x in resp.get("Contents", [])]
        if keys:
            lake.client.delete_objects(
                Bucket=lake.bucket,
                Delete={"Objects": [{"Key": k} for k in keys]},
            )
            total += len(keys)
            if total % 50000 == 0:
                logger.info("deleted=%s", total)
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]

    check_kw: dict = {"Bucket": lake.bucket, "MaxKeys": 1}
    if list_prefix:
        check_kw["Prefix"] = list_prefix
    check = lake.client.list_objects_v2(**check_kw)
    remaining = check.get("KeyCount", 0)

    tables = pg.fetchall(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    names = [r["tablename"] for r in tables]
    for name in names:
        pg.execute(f"DROP TABLE IF EXISTS {name} CASCADE")

    ensure_schema(pg)
    left = pg.fetchall(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    wipe_supervisor_sqlite(cfg.supervisor_state_path)
    out = {
        "bucket": lake.bucket,
        "prefix": list_prefix or "",
        "lake_deleted": total,
        "lake_remaining": remaining,
        "pg_dropped": names,
        "pg_tables_after": [r["tablename"] for r in left],
        "supervisor_state": cfg.supervisor_state_path,
    }
    logger.info("wipe complete %s", out)
    return out


def _month_prefixes(lake: LakeStore) -> list[str]:
    months: list[str] = []
    for year_p in lake.list_common_prefixes("layer_1/"):
        months.extend(lake.list_common_prefixes(year_p))
    return months


def _scrub_key(lake: LakeStore, key: str) -> tuple[str, int]:
    """Return (action, fred_rows) where action is rewrite|delete|skip."""
    try:
        probe = lake.get_df_parquet(key, columns=["source"])
    except Exception as exc:
        logger.warning("wipe_fred skip read key=%s err=%s", key, exc)
        return "skip", 0
    if probe.empty or "source" not in probe.columns:
        return "skip", 0
    mask = probe["source"].astype(str) == "fred"
    n_fred = int(mask.sum())
    if n_fred == 0:
        return "skip", 0
    if n_fred == len(probe):
        lake.delete_keys([key])
        return "delete", n_fred
    df = lake.get_df_parquet(key)
    kept = df.loc[df["source"].astype(str) != "fred"]
    cols = [c for c in RAW_BAR_COLUMNS if c in kept.columns]
    write_parquet_lake(
        lake,
        key,
        kept[cols] if cols else kept,
        sort_by=[c for c in ("source", "source_symbol", "ts") if c in kept.columns],
    )
    return "rewrite", n_fred


def wipe_fred_l1(lake: LakeStore, *, workers: int = 16) -> dict:
    keys: list[str] = []
    months = _month_prefixes(lake)
    logger.info("wipe_fred scanning months=%s", len(months))
    for i, prefix in enumerate(months, 1):
        month_keys = [k for k in lake.list_keys(prefix) if k.endswith(".parquet")]
        keys.extend(month_keys)
        if i % 24 == 0 or i == len(months):
            logger.info("wipe_fred listed months %s/%s keys=%s", i, len(months), len(keys))

    rewritten = deleted = skipped = 0
    fred_rows_removed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scrub_key, lake, key): key for key in keys}
        for fut in as_completed(futures):
            action, n_fred = fut.result()
            fred_rows_removed += n_fred
            if action == "rewrite":
                rewritten += 1
            elif action == "delete":
                deleted += 1
            else:
                skipped += 1
            done += 1
            if done % 200 == 0 or done == len(keys):
                logger.info(
                    "wipe_fred scrub %s/%s rewritten=%s deleted=%s fred_rows=%s",
                    done,
                    len(keys),
                    rewritten,
                    deleted,
                    fred_rows_removed,
                )
    return {
        "months": len(months),
        "files_scanned": len(keys),
        "files_rewritten": rewritten,
        "files_deleted": deleted,
        "files_untouched": skipped,
        "fred_rows_removed": fred_rows_removed,
    }


def wipe_fred_postgres(pg: PgClient) -> dict:
    ensure_schema(pg)
    macro_ids = [
        r["series_id"]
        for r in pg.fetchall("SELECT series_id FROM series_meta WHERE asset_class = 'macro'")
    ]
    dropped = {
        "macro_series": len(macro_ids),
        "symbol_month_coverage_fred": 0,
        "markers_cleared": [],
    }
    if macro_ids:
        pg.execute("DELETE FROM series_cache_span WHERE series_id = ANY(%s)", (macro_ids,))
        pg.execute("DELETE FROM series_cache_meta WHERE series_id = ANY(%s)", (macro_ids,))
        pg.execute("DELETE FROM stitch_segments WHERE series_id = ANY(%s)", (macro_ids,))
        pg.execute(
            "DELETE FROM series_links WHERE series_id = ANY(%s) OR linked_series_id = ANY(%s)",
            (macro_ids, macro_ids),
        )
        pg.execute(
            "DELETE FROM symbol_aliases WHERE series_id = ANY(%s) OR source = 'fred'",
            (macro_ids,),
        )
        pg.execute("DELETE FROM series_meta WHERE series_id = ANY(%s)", (macro_ids,))
    else:
        pg.execute("DELETE FROM symbol_aliases WHERE source = 'fred'")

    cov = pg.fetchone(
        "SELECT COUNT(*)::int AS n FROM symbol_month_coverage WHERE source = 'fred'"
    )
    dropped["symbol_month_coverage_fred"] = int(cov["n"]) if cov else 0
    pg.execute("DELETE FROM symbol_month_coverage WHERE source = 'fred'")
    return dropped


def wipe_fred(cfg: MarketsConfig | None = None) -> dict:
    configure_logging()
    cfg = cfg or MarketsConfig.from_env()
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)

    l1 = wipe_fred_l1(lake)
    pg_out = wipe_fred_postgres(pg)

    markers_cleared = []
    for key in FRED_MARKERS:
        if lake.exists(key):
            lake.delete_keys([key])
            markers_cleared.append(key)
    pg_out["markers_cleared"] = markers_cleared

    out = {"l1": l1, "postgres": pg_out}
    logger.info("wipe_fred complete %s", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Wipe lake + Postgres (full or FRED-only)")
    p.add_argument(
        "--fred-only",
        action="store_true",
        help="Remove FRED/macro L1 + registry only; leave equity/ETF intact",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required. Without this flag the wipe is a no-op.",
    )
    args = p.parse_args()
    if not args.confirm:
        raise SystemExit("refusing to wipe without --confirm")
    if args.fred_only:
        wipe_fred()
    else:
        wipe_all()


if __name__ == "__main__":
    main()
