"""Backfill monthly stitched L3 cache from existing L1 + L2. Run from output/."""

from __future__ import annotations

import argparse

from lexis_markets.cluster import init_ray, timed
from lexis_markets.compact import discover_l1_months
from lexis_markets.config import MarketsConfig
from lexis_markets.stitch_cache import build_stitched_months, cache_status, months_from_l1_manifest
from lexis_markets.storage import LakeStore, PgClient, ensure_schema


def main():
    p = argparse.ArgumentParser(description="Build incremental L3 stitched cache")
    p.add_argument("--rebuild-all", action="store_true", help="rebuild every month (ignore L1 fingerprint skip)")
    args = p.parse_args()

    cfg = MarketsConfig.from_env()
    init_ray(cfg)
    pg = PgClient(cfg.postgres_url)
    lake = LakeStore(cfg)

    with timed("ensure_schema"):
        ensure_schema(pg)

    months = months_from_l1_manifest(pg)
    if not months:
        months = discover_l1_months(lake)
    print(f"build_stitch_cache: {len(months)} months status={cache_status(pg)}", flush=True)
    out = build_stitched_months(cfg, months, force=args.rebuild_all, pg=pg)
    print(
        f"done: rebuilt={len(out)} rows={sum(r.get('rows', 0) for r in out)} status={cache_status(pg)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
