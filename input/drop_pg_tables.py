from lexis_markets.config import MarketsConfig
from lexis_markets.storage import PgClient


def main():
    cfg = MarketsConfig.from_env()
    pg = PgClient(cfg.postgres_url)
    rows = pg.fetchall(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    names = [r["tablename"] for r in rows]
    print(f"tables={names}")
    for name in names:
        pg.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
    left = pg.fetchall(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    print(f"remaining={[r['tablename'] for r in left]}")
    print("dropped=ok")


if __name__ == "__main__":
    main()
