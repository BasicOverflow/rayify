"""Scan MarketParquet free window + NASDAQ directory for new L2 entities."""

from __future__ import annotations

import time

from lexis_markets.config import MarketsConfig
from lexis_markets.entity_detect import run_entity_detect, run_eod_gap_scan
from lexis_markets.nasdaq_symbols import fetch_nasdaq_directory
from lexis_markets.registry import ensure_eod_aliases
from lexis_markets.storage import LakeStore, PgClient, ensure_schema


def main():
    cfg = MarketsConfig.from_env()
    pg = PgClient(cfg.postgres_url)
    lake = LakeStore(cfg)

    t0 = time.perf_counter()
    ensure_schema(pg)
    print(f"ensure_schema: {time.perf_counter() - t0:.1f}s")

    ensure_eod_aliases(pg)
    directory = fetch_nasdaq_directory()
    t0 = time.perf_counter()
    gap = run_eod_gap_scan(pg, lake, directory)
    print(f"gap_scan: {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    out = run_entity_detect(cfg, lake, pg, directory=directory)
    print(f"entity_detect: {time.perf_counter() - t0:.1f}s")
    print(f"entity_detect=ok {out} gap={gap}", flush=True)


if __name__ == "__main__":
    main()
