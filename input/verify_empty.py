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
        total += len(resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]

    print(f"lake: endpoint={cfg.s3_endpoint} bucket={lake.bucket} objects={total}")
    for x in lake.client.list_objects_v2(Bucket=lake.bucket, MaxKeys=5).get("Contents", []):
        print(f"  key: {x['Key']}")

    tables = pg.fetchall(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    print(f"pg: tables={[r['tablename'] for r in tables]}")
    for row in tables:
        n = pg.fetchone(f"SELECT COUNT(*) AS n FROM {row['tablename']}")["n"]
        print(f"  {row['tablename']} rows={n}")


if __name__ == "__main__":
    main()
