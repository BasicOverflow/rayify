"""Smoke test MarketParquet daily download + overlap with stale registry targets."""
from __future__ import annotations

from datetime import date, timedelta

import requests

from lexis_markets.config import MarketsConfig
from lexis_markets.eod import MP_FREE_DAYS, fetch_mp_daily, mp_cutoff, resolve_eod_targets
from lexis_markets.storage import LakeStore, PgClient


def main():
    cfg = MarketsConfig.from_env()
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    targets = resolve_eod_targets(pg)
    sym_set = {t["symbol"] for t in targets}

    day = date.today() - timedelta(days=1)
    df = fetch_mp_daily(day, lake)
    mp_syms = set(df["symbol"].astype(str).str.upper())
    overlap = sym_set & mp_syms

    print(f"marketparquet test day={day.isoformat()}")
    print(f"  file rows={len(df)} symbols={len(mp_syms)}")
    print(f"  stale registry={len(sym_set)} overlap={len(overlap)}")
    print(f"  free window: {mp_cutoff()} .. yesterday ({MP_FREE_DAYS}d)")

    old = day - timedelta(days=MP_FREE_DAYS + 1)
    r = requests.get(
        f"https://marketparquet.com/api/data/download/stock_daily/{old.isoformat()}.parquet",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    print(f"  day-{MP_FREE_DAYS + 1} auth gate: HTTP {r.status_code}")

    print("  sample:", df.head(2).to_string(index=False))


if __name__ == "__main__":
    main()
