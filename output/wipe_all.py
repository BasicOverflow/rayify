from lexis_markets.config import MarketsConfig
from lexis_markets.storage import LakeStore, PgClient


def main():
    cfg = MarketsConfig.from_env()
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)

    token = None
    total = 0
    while True:
        kw = {"Bucket": lake.bucket, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = lake.client.list_objects_v2(**kw)
        keys = [x["Key"] for x in resp.get("Contents", [])]
        if keys:
            lake.delete_keys(keys)
            total += len(keys)
            if total % 50000 == 0:
                print(f"deleted={total}")
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    check = lake.client.list_objects_v2(Bucket=lake.bucket, MaxKeys=1)
    print(
        f"lake: endpoint={cfg.s3_endpoint} bucket={lake.bucket} "
        f"wiped={total} remaining={check.get('KeyCount', 0)}"
    )

    tables = pg.fetchall(
        """
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename
        """
    )
    names = [r["tablename"] for r in tables]
    print(f"pg: tables={names}")
    for name in names:
        pg.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
    left = pg.fetchall(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    print(f"pg: remaining={[r['tablename'] for r in left]}")
    print("reset=ok")


if __name__ == "__main__":
    main()
