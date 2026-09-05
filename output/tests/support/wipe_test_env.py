"""Scoped wipe for test MinIO prefix, test Postgres, and supervisor SQLite."""
from __future__ import annotations

from lexis_markets.lake import LakeStore, PgClient, ensure_schema
from lexis_markets.config import MarketsConfig
from lexis_markets.jobs.queue import wipe_supervisor_sqlite

PROD_NAMESPACE = "lexis-markets"


def assert_test_safe(cfg: MarketsConfig) -> None:
    if not cfg.lake_prefix:
        raise RuntimeError("refusing wipe: MARKETS_LAKE_PREFIX is empty")
    if cfg.ray_namespace == PROD_NAMESPACE:
        raise RuntimeError("refusing wipe: RAY_NAMESPACE is production")
    if "lexis_markets_test" not in cfg.postgres_url:
        raise RuntimeError("refusing wipe: postgres URL is not lexis_markets_test")


TEST_PREFIX_SENTINEL = ".keep"
TEST_PREFIX_DIRS = (
    ".keep",
    "ops/.keep",
    "ops/markers/.keep",
    "layer_1/.keep",
    "layer_3/.keep",
)


def ensure_lake_prefix(lake: LakeStore) -> str:
    """Write sentinels under the lake prefix so MinIO path-based ACLs allow head/list."""
    keys: list[str] = []
    for rel in TEST_PREFIX_DIRS:
        lake.put_bytes(rel, b"", content_type="application/octet-stream")
        keys.append(lake.key(rel))
    return keys[0]


def wipe_minio_prefix(lake: LakeStore, prefix: str) -> int:
    token = None
    total = 0
    while True:
        kw: dict = {"Bucket": lake.bucket, "Prefix": prefix, "MaxKeys": 1000}
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
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return total


def wipe_postgres(pg: PgClient) -> list[str]:
    tables = pg.fetchall(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    names = [r["tablename"] for r in tables]
    for name in names:
        pg.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
    ensure_schema(pg)
    return names


def wipe_test_env(cfg: MarketsConfig) -> dict:
    assert_test_safe(cfg)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    prefix = lake.prefix or cfg.lake_prefix
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    deleted = wipe_minio_prefix(lake, prefix) if prefix else 0
    sentinel = ensure_lake_prefix(lake) if prefix else None
    dropped = wipe_postgres(pg)
    wipe_supervisor_sqlite(cfg.supervisor_state_path)
    return {"lake_deleted": deleted, "pg_dropped": dropped, "prefix": prefix, "sentinel": sentinel}
